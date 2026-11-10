import Foundation
import MLX

/// Dequantize GGUF tensor payloads to dense float16, matching llama.cpp `dequantize_row_*`.
public enum GGUFDequant {
    public static func toMLXArray(data: Data, info: GGUFTensorInfo) throws -> MLXArray {
        let shape = info.dimensions.reversed().map { Int($0) }
        let n = Int(info.totalElements)
        switch info.type {
        case .f16:
            let values = dataToArray(data, Float16.self)
            return MLXArray(values, shape)
        case .f32:
            let values = dataToArray(data, Float.self)
            return MLXArray(values, shape).asType(.float16)
        case .f64:
            let doubles = dataToArray(data, Double.self)
            return MLXArray(doubles.map { Float($0) }, shape).asType(.float16)
        case .bf16:
            return MLXArray(dequantizeBF16(data), shape).asType(.float16)
        case .q4_0:
            return f16Array(dequantizeQ4_0(data, n), shape)
        case .q4_1:
            return f16Array(dequantizeQ4_1(data, n), shape)
        case .q5_0:
            return f16Array(dequantizeQ5_0(data, n), shape)
        case .q5_1:
            return f16Array(dequantizeQ5_1(data, n), shape)
        case .q8_0:
            return f16Array(dequantizeQ8_0(data, n), shape)
        case .q8_1:
            return f16Array(dequantizeQ8_1(data, n), shape)
        case .q2_K:
            return f16Array(dequantizeQ2_K(data, n), shape)
        case .q3_K:
            return f16Array(dequantizeQ3_K(data, n), shape)
        case .q4_K:
            return f16Array(dequantizeQ4_K(data, n), shape)
        case .q5_K:
            return f16Array(dequantizeQ5_K(data, n), shape)
        case .q6_K:
            return f16Array(dequantizeQ6_K(data, n), shape)
        case .q8_K:
            return f16Array(dequantizeQ8_K(data, n), shape)
        default:
            throw GGUFError.unsupportedTensorType(info.name, info.type.rawValue)
        }
    }
}

private func f16Array(_ values: [Float16], _ shape: [Int]) -> MLXArray {
    MLXArray(values.map { Float($0) }, shape).asType(.float16)
}

private func dataToArray<T>(_ data: Data, _ type: T.Type) -> [T] {
    let count = data.count / MemoryLayout<T>.size
    return data.withUnsafeBytes { ptr in
        Array(
            UnsafeBufferPointer(
                start: ptr.baseAddress?.assumingMemoryBound(to: T.self),
                count: count))
    }
}

private func f16(_ ptr: UnsafeRawPointer, _ offset: Int) -> Float {
    Float(Float16(bitPattern: ptr.loadUnaligned(fromByteOffset: offset, as: UInt16.self)))
}

private func dequantizeBF16(_ data: Data) -> [Float] {
    let n = data.count / 2
    var out = [Float]()
    out.reserveCapacity(n)
    data.withUnsafeBytes { raw in
        let p = raw.baseAddress!
        for i in 0..<n {
            let bits = UInt32(p.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self)) << 16
            out.append(Float(bitPattern: bits))
        }
    }
    return out
}

// MARK: - Legacy 32-wide quants
// llama.cpp stores the high nibble as the *second half* of the block, not interleaved.

private func dequantizeQ4_0(_ data: Data, _ n: Int) -> [Float16] {
    let block = 32
    let bpb = 18
    let nBlocks = (n + block - 1) / block
    var out = [Float16](repeating: 0, count: nBlocks * block)
    data.withUnsafeBytes { raw in
        let base = raw.baseAddress!
        for i in 0..<nBlocks {
            let p = base.advanced(by: i * bpb)
            let d = Float16(bitPattern: p.loadUnaligned(as: UInt16.self))
            let qs = p.advanced(by: 2)
            let dst = i * block
            for j in 0..<16 {
                let byte = qs.loadUnaligned(fromByteOffset: j, as: UInt8.self)
                out[dst + j] = d * Float16(Int(byte & 0x0F) - 8)
                out[dst + j + 16] = d * Float16(Int(byte >> 4) - 8)
            }
        }
    }
    if out.count > n { out.removeSubrange(n...) }
    return out
}

private func dequantizeQ4_1(_ data: Data, _ n: Int) -> [Float16] {
    let block = 32
    let bpb = 20
    let nBlocks = (n + block - 1) / block
    var out = [Float16](repeating: 0, count: nBlocks * block)
    data.withUnsafeBytes { raw in
        let base = raw.baseAddress!
        for i in 0..<nBlocks {
            let p = base.advanced(by: i * bpb)
            let d = Float16(bitPattern: p.loadUnaligned(as: UInt16.self))
            let m = Float16(bitPattern: p.loadUnaligned(fromByteOffset: 2, as: UInt16.self))
            let qs = p.advanced(by: 4)
            let dst = i * block
            for j in 0..<16 {
                let byte = qs.loadUnaligned(fromByteOffset: j, as: UInt8.self)
                out[dst + j] = d * Float16(byte & 0x0F) + m
                out[dst + j + 16] = d * Float16(byte >> 4) + m
            }
        }
    }
    if out.count > n { out.removeSubrange(n...) }
    return out
}

private func dequantizeQ5_0(_ data: Data, _ n: Int) -> [Float16] {
    let block = 32
    let bpb = 22
    let nBlocks = (n + block - 1) / block
    var out = [Float16](repeating: 0, count: nBlocks * block)
    data.withUnsafeBytes { raw in
        let base = raw.baseAddress!
        for i in 0..<nBlocks {
            let p = base.advanced(by: i * bpb)
            let d = Float16(bitPattern: p.loadUnaligned(as: UInt16.self))
            let qh = p.loadUnaligned(fromByteOffset: 2, as: UInt32.self)
            let qs = p.advanced(by: 6)
            let dst = i * block
            for j in 0..<16 {
                let byte = qs.loadUnaligned(fromByteOffset: j, as: UInt8.self)
                let xh0 = UInt8((qh >> j) << 4) & 0x10
                let xh1 = UInt8(qh >> (j + 12)) & 0x10
                out[dst + j] = d * Float16(Int(byte & 0x0F | xh0) - 16)
                out[dst + j + 16] = d * Float16(Int(byte >> 4 | xh1) - 16)
            }
        }
    }
    if out.count > n { out.removeSubrange(n...) }
    return out
}

private func dequantizeQ5_1(_ data: Data, _ n: Int) -> [Float16] {
    let block = 32
    let bpb = 24
    let nBlocks = (n + block - 1) / block
    var out = [Float16](repeating: 0, count: nBlocks * block)
    data.withUnsafeBytes { raw in
        let base = raw.baseAddress!
        for i in 0..<nBlocks {
            let p = base.advanced(by: i * bpb)
            let d = Float16(bitPattern: p.loadUnaligned(as: UInt16.self))
            let m = Float16(bitPattern: p.loadUnaligned(fromByteOffset: 2, as: UInt16.self))
            let qh = p.loadUnaligned(fromByteOffset: 4, as: UInt32.self)
            let qs = p.advanced(by: 8)
            let dst = i * block
            for j in 0..<16 {
                let byte = qs.loadUnaligned(fromByteOffset: j, as: UInt8.self)
                let xh0 = UInt8((qh >> j) << 4) & 0x10
                let xh1 = UInt8(qh >> (j + 12)) & 0x10
                out[dst + j] = d * Float16(byte & 0x0F | xh0) + m
                out[dst + j + 16] = d * Float16(byte >> 4 | xh1) + m
            }
        }
    }
    if out.count > n { out.removeSubrange(n...) }
    return out
}

private func dequantizeQ8_0(_ data: Data, _ n: Int) -> [Float16] {
    let block = 32
    let bpb = 34
    let nBlocks = (n + block - 1) / block
    var out = [Float16](repeating: 0, count: nBlocks * block)
    data.withUnsafeBytes { raw in
        let base = raw.baseAddress!
        for i in 0..<nBlocks {
            let p = base.advanced(by: i * bpb)
            let d = Float16(bitPattern: p.loadUnaligned(as: UInt16.self))
            let qs = p.advanced(by: 2)
            let dst = i * block
            for j in 0..<32 {
                out[dst + j] = d * Float16(qs.loadUnaligned(fromByteOffset: j, as: Int8.self))
            }
        }
    }
    if out.count > n { out.removeSubrange(n...) }
    return out
}

private func dequantizeQ8_1(_ data: Data, _ n: Int) -> [Float16] {
    let block = 32
    let bpb = 38
    let nBlocks = (n + block - 1) / block
    var out = [Float16](repeating: 0, count: nBlocks * block)
    data.withUnsafeBytes { raw in
        let base = raw.baseAddress!
        for i in 0..<nBlocks {
            let p = base.advanced(by: i * bpb)
            let d = Float16(bitPattern: p.loadUnaligned(as: UInt16.self))
            let qs = p.advanced(by: 4)
            let dst = i * block
            for j in 0..<32 {
                out[dst + j] = d * Float16(qs.loadUnaligned(fromByteOffset: j, as: Int8.self))
            }
        }
    }
    if out.count > n { out.removeSubrange(n...) }
    return out
}

// MARK: - K-quants (super-block 256)

private let qkK = 256

private func getScaleMinK4(j: Int, q: UnsafeRawPointer) -> (UInt8, UInt8) {
    if j < 4 {
        let d = q.loadUnaligned(fromByteOffset: j, as: UInt8.self) & 63
        let m = q.loadUnaligned(fromByteOffset: j + 4, as: UInt8.self) & 63
        return (d, m)
    }
    let d =
        (q.loadUnaligned(fromByteOffset: j + 4, as: UInt8.self) & 0xF)
        | ((q.loadUnaligned(fromByteOffset: j - 4, as: UInt8.self) >> 6) << 4)
    let m =
        (q.loadUnaligned(fromByteOffset: j + 4, as: UInt8.self) >> 4)
        | ((q.loadUnaligned(fromByteOffset: j, as: UInt8.self) >> 6) << 4)
    return (d, m)
}

/// block_q2_K: scales[16] + qs[64] + d + dmin = 84
private func dequantizeQ2_K(_ data: Data, _ n: Int) -> [Float16] {
    let bpb = 84
    let nBlocks = (n + qkK - 1) / qkK
    var out = [Float16](repeating: 0, count: nBlocks * qkK)
    data.withUnsafeBytes { raw in
        let base = raw.baseAddress!
        for i in 0..<nBlocks {
            let p = base.advanced(by: i * bpb)
            let scales = p
            let qs = p.advanced(by: 16)
            let d = f16(p, 80)
            let minv = f16(p, 82)
            var y = i * qkK
            var qOff = 0
            var iscale = 0
            for _ in 0..<2 {
                var shift = 0
                for _ in 0..<4 {
                    let sc0 = scales.loadUnaligned(fromByteOffset: iscale, as: UInt8.self)
                    iscale += 1
                    let dl0 = d * Float(sc0 & 0xF)
                    let ml0 = minv * Float(sc0 >> 4)
                    for l in 0..<16 {
                        let q = Int(qs.loadUnaligned(fromByteOffset: qOff + l, as: UInt8.self) >> shift) & 3
                        out[y] = Float16(dl0 * Float(q) - ml0)
                        y += 1
                    }
                    let sc1 = scales.loadUnaligned(fromByteOffset: iscale, as: UInt8.self)
                    iscale += 1
                    let dl1 = d * Float(sc1 & 0xF)
                    let ml1 = minv * Float(sc1 >> 4)
                    for l in 0..<16 {
                        let q = Int(qs.loadUnaligned(fromByteOffset: qOff + 16 + l, as: UInt8.self) >> shift) & 3
                        out[y] = Float16(dl1 * Float(q) - ml1)
                        y += 1
                    }
                    shift += 2
                }
                qOff += 32
            }
        }
    }
    if out.count > n { out.removeSubrange(n...) }
    return out
}

/// block_q3_K: hmask[32] + qs[64] + scales[12] + d = 110
private func dequantizeQ3_K(_ data: Data, _ n: Int) -> [Float16] {
    let bpb = 110
    let nBlocks = (n + qkK - 1) / qkK
    var out = [Float16](repeating: 0, count: nBlocks * qkK)
    data.withUnsafeBytes { raw in
        let base = raw.baseAddress!
        let kmask1: UInt32 = 0x0303_0303
        let kmask2: UInt32 = 0x0F0F_0F0F
        for i in 0..<nBlocks {
            let p = base.advanced(by: i * bpb)
            let hmask = p
            let qs = p.advanced(by: 32)
            let scalesRaw = p.advanced(by: 96)
            let dAll = f16(p, 108)
            var aux: [UInt32] = [
                UInt32(scalesRaw.loadUnaligned(fromByteOffset: 0, as: UInt32.self)),
                UInt32(scalesRaw.loadUnaligned(fromByteOffset: 4, as: UInt32.self)),
                UInt32(scalesRaw.loadUnaligned(fromByteOffset: 8, as: UInt32.self)),
                0,
            ]
            let tmp = aux[2]
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4)
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4)
            var scales = [Int8](repeating: 0, count: 16)
            aux.withUnsafeBytes { ab in
                for s in 0..<16 {
                    scales[s] = ab.loadUnaligned(fromByteOffset: s, as: Int8.self)
                }
            }
            var y = i * qkK
            var qOff = 0
            var iscale = 0
            var mBit: UInt8 = 1
            for _ in 0..<2 {
                var shift = 0
                for _ in 0..<4 {
                    let dl0 = dAll * Float(Int(scales[iscale]) - 32)
                    iscale += 1
                    for l in 0..<16 {
                        let q = Int(qs.loadUnaligned(fromByteOffset: qOff + l, as: UInt8.self) >> shift) & 3
                        let hm = hmask.loadUnaligned(fromByteOffset: l, as: UInt8.self)
                        out[y] = Float16(dl0 * Float(q - ((hm & mBit) != 0 ? 0 : 4)))
                        y += 1
                    }
                    let dl1 = dAll * Float(Int(scales[iscale]) - 32)
                    iscale += 1
                    for l in 0..<16 {
                        let q = Int(qs.loadUnaligned(fromByteOffset: qOff + 16 + l, as: UInt8.self) >> shift) & 3
                        let hm = hmask.loadUnaligned(fromByteOffset: 16 + l, as: UInt8.self)
                        out[y] = Float16(dl1 * Float(q - ((hm & mBit) != 0 ? 0 : 4)))
                        y += 1
                    }
                    shift += 2
                    mBit &<<= 1
                }
                qOff += 32
            }
        }
    }
    if out.count > n { out.removeSubrange(n...) }
    return out
}

/// block_q4_K: d + dmin + scales[12] + qs[128] = 144
private func dequantizeQ4_K(_ data: Data, _ n: Int) -> [Float16] {
    let bpb = 144
    let nBlocks = (n + qkK - 1) / qkK
    var out = [Float16](repeating: 0, count: nBlocks * qkK)
    data.withUnsafeBytes { raw in
        let base = raw.baseAddress!
        for i in 0..<nBlocks {
            let p = base.advanced(by: i * bpb)
            let d = f16(p, 0)
            let minv = f16(p, 2)
            let scales = p.advanced(by: 4)
            var q = p.advanced(by: 16)
            var y = i * qkK
            var iscale = 0
            for _ in 0..<(qkK / 64) {
                let (sc0, m0) = getScaleMinK4(j: iscale, q: scales)
                let (sc1, m1) = getScaleMinK4(j: iscale + 1, q: scales)
                let d1 = d * Float(sc0)
                let m1v = minv * Float(m0)
                let d2 = d * Float(sc1)
                let m2v = minv * Float(m1)
                for l in 0..<32 {
                    let byte = q.loadUnaligned(fromByteOffset: l, as: UInt8.self)
                    out[y + l] = Float16(d1 * Float(byte & 0xF) - m1v)
                    out[y + 32 + l] = Float16(d2 * Float(byte >> 4) - m2v)
                }
                q = q.advanced(by: 32)
                y += 64
                iscale += 2
            }
        }
    }
    if out.count > n { out.removeSubrange(n...) }
    return out
}

/// block_q5_K: d + dmin + scales[12] + qh[32] + qs[128] = 176
private func dequantizeQ5_K(_ data: Data, _ n: Int) -> [Float16] {
    let bpb = 176
    let nBlocks = (n + qkK - 1) / qkK
    var out = [Float16](repeating: 0, count: nBlocks * qkK)
    data.withUnsafeBytes { raw in
        let base = raw.baseAddress!
        for i in 0..<nBlocks {
            let p = base.advanced(by: i * bpb)
            let d = f16(p, 0)
            let minv = f16(p, 2)
            let scales = p.advanced(by: 4)
            let qh = p.advanced(by: 16)
            var ql = p.advanced(by: 48)
            var y = i * qkK
            var iscale = 0
            var u1: UInt8 = 1
            var u2: UInt8 = 2
            for _ in 0..<(qkK / 64) {
                let (sc0, m0) = getScaleMinK4(j: iscale, q: scales)
                let (sc1, m1) = getScaleMinK4(j: iscale + 1, q: scales)
                let d1 = d * Float(sc0)
                let m1v = minv * Float(m0)
                let d2 = d * Float(sc1)
                let m2v = minv * Float(m1)
                for l in 0..<32 {
                    let lo = ql.loadUnaligned(fromByteOffset: l, as: UInt8.self)
                    let hi = qh.loadUnaligned(fromByteOffset: l, as: UInt8.self)
                    out[y + l] = Float16(d1 * Float(Int(lo & 0xF) + (hi & u1 != 0 ? 16 : 0)) - m1v)
                    out[y + 32 + l] = Float16(d2 * Float(Int(lo >> 4) + (hi & u2 != 0 ? 16 : 0)) - m2v)
                }
                ql = ql.advanced(by: 32)
                y += 64
                iscale += 2
                u1 &<<= 2
                u2 &<<= 2
            }
        }
    }
    if out.count > n { out.removeSubrange(n...) }
    return out
}

/// block_q6_K: ql[128] + qh[64] + scales[16] + d = 210
private func dequantizeQ6_K(_ data: Data, _ n: Int) -> [Float16] {
    let bpb = 210
    let nBlocks = (n + qkK - 1) / qkK
    var out = [Float16](repeating: 0, count: nBlocks * qkK)
    data.withUnsafeBytes { raw in
        let base = raw.baseAddress!
        for i in 0..<nBlocks {
            let p = base.advanced(by: i * bpb)
            var ql = p
            var qh = p.advanced(by: 128)
            var sc = p.advanced(by: 192)
            let d = f16(p, 208)
            var y = i * qkK
            for _ in 0..<(qkK / 128) {
                for l in 0..<32 {
                    let iscale = l / 16
                    let q1 =
                        Int((ql.loadUnaligned(fromByteOffset: l, as: UInt8.self) & 0xF)
                            | ((qh.loadUnaligned(fromByteOffset: l, as: UInt8.self) >> 0) & 3) << 4) - 32
                    let q2 =
                        Int((ql.loadUnaligned(fromByteOffset: l + 32, as: UInt8.self) & 0xF)
                            | ((qh.loadUnaligned(fromByteOffset: l, as: UInt8.self) >> 2) & 3) << 4) - 32
                    let q3 =
                        Int((ql.loadUnaligned(fromByteOffset: l, as: UInt8.self) >> 4)
                            | ((qh.loadUnaligned(fromByteOffset: l, as: UInt8.self) >> 4) & 3) << 4) - 32
                    let q4 =
                        Int((ql.loadUnaligned(fromByteOffset: l + 32, as: UInt8.self) >> 4)
                            | ((qh.loadUnaligned(fromByteOffset: l, as: UInt8.self) >> 6) & 3) << 4) - 32
                    let s0 = Float(sc.loadUnaligned(fromByteOffset: iscale, as: Int8.self))
                    let s2 = Float(sc.loadUnaligned(fromByteOffset: iscale + 2, as: Int8.self))
                    let s4 = Float(sc.loadUnaligned(fromByteOffset: iscale + 4, as: Int8.self))
                    let s6 = Float(sc.loadUnaligned(fromByteOffset: iscale + 6, as: Int8.self))
                    out[y + l] = Float16(d * s0 * Float(q1))
                    out[y + l + 32] = Float16(d * s2 * Float(q2))
                    out[y + l + 64] = Float16(d * s4 * Float(q3))
                    out[y + l + 96] = Float16(d * s6 * Float(q4))
                }
                y += 128
                ql = ql.advanced(by: 64)
                qh = qh.advanced(by: 32)
                sc = sc.advanced(by: 8)
            }
        }
    }
    if out.count > n { out.removeSubrange(n...) }
    return out
}

/// block_q8_K: float d + qs[256] + bsums[16] = 292
private func dequantizeQ8_K(_ data: Data, _ n: Int) -> [Float16] {
    let bpb = 292
    let nBlocks = (n + qkK - 1) / qkK
    var out = [Float16](repeating: 0, count: nBlocks * qkK)
    data.withUnsafeBytes { raw in
        let base = raw.baseAddress!
        for i in 0..<nBlocks {
            let p = base.advanced(by: i * bpb)
            let d = p.loadUnaligned(as: Float.self)
            let qs = p.advanced(by: 4)
            let dst = i * qkK
            for j in 0..<qkK {
                out[dst + j] = Float16(d * Float(qs.loadUnaligned(fromByteOffset: j, as: Int8.self)))
            }
        }
    }
    if out.count > n { out.removeSubrange(n...) }
    return out
}
