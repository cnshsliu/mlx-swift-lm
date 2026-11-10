import Foundation

// MARK: - GGUF Types

public enum GGUFTensorType: UInt32, Sendable {
    case f32 = 0, f16 = 1
    case q4_0 = 2, q4_1 = 3
    case q5_0 = 6, q5_1 = 7
    case q8_0 = 8, q8_1 = 9
    case q2_K = 10, q3_K = 11, q4_K = 12, q5_K = 13, q6_K = 14, q8_K = 15
    case iq2_xxs = 16, iq2_xs = 17, iq3_xxs = 18, iq1_s = 19
    case iq4_nl = 20, iq3_s = 21, iq2_s = 22
    case iq4_xs = 24
    case i8 = 25, i16 = 26, i32 = 27, i64 = 28, f64 = 29
    case iq1_m = 30, bf16 = 31

    var elementSize: Int {
        switch self {
        case .f32: return 4
        case .f16: return 2
        case .bf16: return 2
        case .f64: return 8
        case .i8: return 1
        case .i16: return 2
        case .i32: return 4
        case .i64: return 8
        default: return 0
        }
    }

    var blockSize: Int {
        switch self {
        case .f32, .f16, .bf16, .f64, .i8, .i16, .i32, .i64: return 1
        case .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1: return 32
        case .q2_K, .q3_K, .q4_K, .q5_K, .q6_K, .q8_K: return 256
        default: return 256
        }
    }

    var bytesPerBlock: Int {
        switch self {
        case .f32: return 4
        case .f16: return 2
        case .bf16: return 2
        case .f64: return 8
        case .i8: return 1
        case .i16: return 2
        case .i32: return 4
        case .i64: return 8
        case .q4_0: return 18
        case .q4_1: return 20
        case .q5_0: return 22
        case .q5_1: return 24
        case .q8_0: return 34
        case .q8_1: return 38
        case .q2_K: return 84
        case .q3_K: return 110
        case .q4_K: return 144
        case .q5_K: return 176
        case .q6_K: return 210
        case .q8_K: return 292
        default: return 0
        }
    }

    var isQuantized: Bool {
        switch self {
        case .f32, .f16, .bf16, .f64, .i8, .i16, .i32, .i64: return false
        default: return true
        }
    }

    /// Dense or k-quant types we dequantize to FP16. IQ / MX types are not supported.
    public var isSupported: Bool {
        switch self {
        case .f32, .f16, .bf16, .f64,
            .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1,
            .q2_K, .q3_K, .q4_K, .q5_K, .q6_K, .q8_K:
            return true
        default:
            return false
        }
    }
}

public enum GGUFValueType: UInt32 {
    case uint8 = 0, int8 = 1
    case uint16 = 2, int16 = 3
    case uint32 = 4, int32 = 5
    case float32 = 6, bool = 7
    case string = 8, array = 9
    case uint64 = 10, int64 = 11
    case float64 = 12
}

public indirect enum GGUFMetadata: @unchecked Sendable {
    case uint8(UInt8)
    case int8(Int8)
    case uint16(UInt16)
    case int16(Int16)
    case uint32(UInt32)
    case int32(Int32)
    case float32(Float)
    case bool(Bool)
    case string(String)
    case array(GGUFValueType, [GGUFMetadata])
    case uint64(UInt64)
    case int64(Int64)
    case float64(Double)

    public var stringValue: String? {
        if case .string(let s) = self { return s }
        return nil
    }

    public var uint64Value: UInt64? {
        switch self {
        case .uint64(let v): return v
        case .uint32(let v): return UInt64(v)
        case .int32(let v): return v >= 0 ? UInt64(v) : nil
        case .int64(let v): return v >= 0 ? UInt64(v) : nil
        case .uint16(let v): return UInt64(v)
        case .uint8(let v): return UInt64(v)
        default: return nil
        }
    }

    public var intValue: Int? {
        if let u = uint64Value, u <= UInt64(Int.max) { return Int(u) }
        switch self {
        case .int32(let v): return Int(v)
        case .int64(let v): return Int(v)
        case .int16(let v): return Int(v)
        case .int8(let v): return Int(v)
        default: return nil
        }
    }

    public var floatValue: Float? {
        switch self {
        case .float32(let v): return v
        case .float64(let v): return Float(v)
        default: return nil
        }
    }

    public var boolValue: Bool? {
        if case .bool(let v) = self { return v }
        return nil
    }

    public var stringArray: [String]? {
        if case .array(_, let items) = self {
            return items.compactMap(\.stringValue)
        }
        return nil
    }

    public var floatArray: [Float]? {
        if case .array(_, let items) = self {
            return items.compactMap { item in
                switch item {
                case .float32(let v): return v
                case .float64(let v): return Float(v)
                default: return nil
                }
            }
        }
        return nil
    }
}

// MARK: - GGUF Tensor Info

public struct GGUFTensorInfo: Sendable {
    public let name: String
    public let dimensions: [UInt64]
    public let type: GGUFTensorType
    public let dataOffset: UInt64
    public let dataLength: UInt64
    public let totalElements: UInt64

    public init(
        name: String, dimensions: [UInt64], type: GGUFTensorType, dataOffset: UInt64,
        dataLength: UInt64, totalElements: UInt64
    ) {
        self.name = name
        self.dimensions = dimensions
        self.type = type
        self.dataOffset = dataOffset
        self.dataLength = dataLength
        self.totalElements = totalElements
    }
}

// MARK: - GGUF Error

public enum GGUFError: Error, LocalizedError {
    case invalidMagic(UInt32)
    case unsupportedVersion(UInt32)
    case invalidValueType(UInt32)
    case unsupportedTensorType(String, UInt32)
    case tensorNotFound(String)
    case readError(String)

    public var errorDescription: String? {
        switch self {
        case .invalidMagic(let m):
            return "Invalid GGUF magic: 0x\(String(m, radix: 16))"
        case .unsupportedVersion(let v):
            return "Unsupported GGUF version: \(v)"
        case .invalidValueType(let t):
            return "Invalid GGUF value type: \(t)"
        case .unsupportedTensorType(let name, let t):
            return "Unsupported tensor type \(t) for tensor '\(name)'"
        case .tensorNotFound(let n):
            return "Tensor not found: \(n)"
        case .readError(let m):
            return "GGUF read error: \(m)"
        }
    }
}

// MARK: - GGUF File

public struct GGUFFile: Sendable {
    public let version: UInt32
    public let metadata: [String: GGUFMetadata]
    public let tensors: [String: GGUFTensorInfo]
    public let fileURL: URL
    public let alignment: UInt64

    public static let magic: UInt32 = 0x46554747  // "GGUF" little-endian

    public static func parse(url: URL) throws -> GGUFFile {
        guard let handle = try? FileHandle(forReadingFrom: url) else {
            throw GGUFError.readError("Cannot open file: \(url.path)")
        }
        defer { try? handle.close() }

        let magic = try handle.readLE32()
        guard magic == Self.magic else {
            throw GGUFError.invalidMagic(magic)
        }

        let version = try handle.readLE32()
        guard version >= 2 && version <= 3 else {
            throw GGUFError.unsupportedVersion(version)
        }

        let nTensors = try handle.readLE64()
        let nKV = try handle.readLE64()

        var metadata: [String: GGUFMetadata] = [:]
        metadata.reserveCapacity(Int(nKV))
        for _ in 0..<nKV {
            let key = try handle.readGGUFString()
            let rawType = try handle.readLE32()
            guard let vtype = GGUFValueType(rawValue: rawType) else {
                throw GGUFError.invalidValueType(rawType)
            }
            metadata[key] = try handle.readGGUFValue(type: vtype)
        }

        let rawAlign = metadata["general.alignment"]?.uint64Value ?? 32
        let alignment = rawAlign > 0 ? rawAlign : 32

        let headerEnd = handle.offsetInFile
        let dataSectionOffset = Self.alignUp(headerEnd, to: alignment)

        var tensors: [String: GGUFTensorInfo] = [:]
        tensors.reserveCapacity(Int(nTensors))

        for _ in 0..<nTensors {
            let name = try handle.readGGUFString()
            let nDims = try handle.readLE32()
            var dims: [UInt64] = []
            dims.reserveCapacity(Int(nDims))
            for _ in 0..<nDims {
                dims.append(try handle.readLE64())
            }
            let rawType = try handle.readLE32()
            guard let type = GGUFTensorType(rawValue: rawType) else {
                throw GGUFError.unsupportedTensorType(name, rawType)
            }
            let offset = try handle.readLE64()

            let totalElements = dims.reduce(UInt64(1), *)
            let dataLength = Self.computeDataLength(elements: totalElements, type: type)

            tensors[name] = GGUFTensorInfo(
                name: name,
                dimensions: dims,
                type: type,
                dataOffset: dataSectionOffset + offset,
                dataLength: dataLength,
                totalElements: totalElements
            )
        }

        return GGUFFile(
            version: version,
            metadata: metadata,
            tensors: tensors,
            fileURL: url,
            alignment: alignment
        )
    }

    public func readTensorData(_ name: String) throws -> Data {
        guard let info = tensors[name] else {
            throw GGUFError.tensorNotFound(name)
        }
        let handle = try FileHandle(forReadingFrom: fileURL)
        defer { try? handle.close() }
        try handle.seek(toOffset: info.dataOffset)
        guard let data = try handle.read(upToCount: Int(info.dataLength)) else {
            throw GGUFError.readError("Failed to read tensor data for \(name)")
        }
        return data
    }

    public var architecture: String? {
        metadata["general.architecture"]?.stringValue
    }

    public func meta(_ key: String) -> GGUFMetadata? {
        metadata[key]
    }

    public func archMeta(_ suffix: String) -> GGUFMetadata? {
        guard let arch = architecture else { return nil }
        return metadata["\(arch).\(suffix)"]
    }

    public func archInt(_ suffix: String) -> Int? {
        archMeta(suffix)?.intValue
    }

    public func archFloat(_ suffix: String) -> Float? {
        archMeta(suffix)?.floatValue
    }

    public var expertCount: Int {
        archInt("expert_count") ?? 0
    }

    private static func computeDataLength(elements: UInt64, type: GGUFTensorType) -> UInt64 {
        if type.blockSize == 1 {
            return elements * UInt64(type.elementSize)
        }
        let nBlocks = (elements + UInt64(type.blockSize) - 1) / UInt64(type.blockSize)
        return nBlocks * UInt64(type.bytesPerBlock)
    }

    private static func alignUp(_ offset: UInt64, to alignment: UInt64) -> UInt64 {
        (offset + alignment - 1) / alignment * alignment
    }
}

// MARK: - FileHandle GGUF Extensions

extension FileHandle {
    func readLE32() throws -> UInt32 {
        guard let data = try read(upToCount: 4), data.count == 4 else {
            throw GGUFError.readError("unexpected EOF reading uint32")
        }
        return data.withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }
    }

    func readLE64() throws -> UInt64 {
        guard let data = try read(upToCount: 8), data.count == 8 else {
            throw GGUFError.readError("unexpected EOF reading uint64")
        }
        return data.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
    }

    func readGGUFString() throws -> String {
        let length = try readLE64()
        guard length < 1_000_000 else {
            throw GGUFError.readError("string too long: \(length)")
        }
        guard length > 0 else { return "" }
        guard let data = try read(upToCount: Int(length)), data.count == Int(length) else {
            throw GGUFError.readError("unexpected EOF reading string")
        }
        return String(data: data, encoding: .utf8) ?? ""
    }

    func readGGUFValue(type: GGUFValueType) throws -> GGUFMetadata {
        switch type {
        case .uint8:
            guard let d = try read(upToCount: 1), d.count == 1 else {
                throw GGUFError.readError("EOF reading uint8")
            }
            return .uint8(d[0])
        case .int8:
            guard let d = try read(upToCount: 1), d.count == 1 else {
                throw GGUFError.readError("EOF reading int8")
            }
            return .int8(d.withUnsafeBytes { $0.loadUnaligned(as: Int8.self) })
        case .uint16:
            guard let d = try read(upToCount: 2), d.count == 2 else {
                throw GGUFError.readError("EOF reading uint16")
            }
            return .uint16(d.withUnsafeBytes { $0.loadUnaligned(as: UInt16.self) })
        case .int16:
            guard let d = try read(upToCount: 2), d.count == 2 else {
                throw GGUFError.readError("EOF reading int16")
            }
            return .int16(d.withUnsafeBytes { $0.loadUnaligned(as: Int16.self) })
        case .uint32:
            return .uint32(try readLE32())
        case .int32:
            guard let d = try read(upToCount: 4), d.count == 4 else {
                throw GGUFError.readError("EOF reading int32")
            }
            return .int32(d.withUnsafeBytes { $0.loadUnaligned(as: Int32.self) })
        case .float32:
            guard let d = try read(upToCount: 4), d.count == 4 else {
                throw GGUFError.readError("EOF reading float32")
            }
            return .float32(d.withUnsafeBytes { $0.loadUnaligned(as: Float.self) })
        case .bool:
            guard let d = try read(upToCount: 1), d.count == 1 else {
                throw GGUFError.readError("EOF reading bool")
            }
            return .bool(d[0] != 0)
        case .string:
            return .string(try readGGUFString())
        case .array:
            let rawElemType = try readLE32()
            guard let etype = GGUFValueType(rawValue: rawElemType) else {
                throw GGUFError.invalidValueType(rawElemType)
            }
            let count = try readLE64()
            var items: [GGUFMetadata] = []
            items.reserveCapacity(Int(count))
            for _ in 0..<count {
                items.append(try readGGUFValue(type: etype))
            }
            return .array(etype, items)
        case .uint64:
            return .uint64(try readLE64())
        case .int64:
            guard let d = try read(upToCount: 8), d.count == 8 else {
                throw GGUFError.readError("EOF reading int64")
            }
            return .int64(d.withUnsafeBytes { $0.loadUnaligned(as: Int64.self) })
        case .float64:
            guard let d = try read(upToCount: 8), d.count == 8 else {
                throw GGUFError.readError("EOF reading float64")
            }
            return .float64(d.withUnsafeBytes { $0.loadUnaligned(as: Double.self) })
        }
    }
}
