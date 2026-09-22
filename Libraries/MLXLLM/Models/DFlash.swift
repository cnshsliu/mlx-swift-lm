import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Port of mlx-dspark `dflash_model.py` (z-lab DFlash + Inco AI DFlash 2).

public struct DFlashConfig: Sendable, Equatable {
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var intermediateSize: Int
    public var vocabSize: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var maxPositionEmbeddings: Int
    public var blockSize: Int
    public var targetLayerIds: [Int]
    public var numTargetLayers: Int
    public var maskTokenId: Int
    public var layerTypes: [String]
    public var slidingWindow: Int?
    public var finalLogitSoftcapping: Float?
    public var selectorRank: Int
    public var selectorTopK: Int
    public var convKernelSize: Int
    public var convGroupSize: Int
    public var outputMultiplier: Float

    public var isDFlash2: Bool { selectorRank > 0 && selectorTopK > 0 }

    public static func fromJSON(_ data: Data) throws -> DFlashConfig {
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw DFlashError.invalidConfig("not a JSON object")
        }
        return try fromJSONObject(json)
    }

    public static func fromJSONObject(_ json: [String: Any]) throws -> DFlashConfig {
        let dfc = json["dflash_config"] as? [String: Any] ?? [:]
        guard let blockSize = intValue(dfc["block_size"]) ?? intValue(json["block_size"]) else {
            throw DFlashError.invalidConfig("missing block_size")
        }
        let selectorRank = intValue(dfc["selector_rank"]) ?? 0
        let selectorTopK = intValue(dfc["selector_top_k"]) ?? 0
        let arches = (json["architectures"] as? [String]) ?? []
        if arches.contains("DFlash2DraftModel"), selectorRank == 0 || selectorTopK == 0 {
            throw DFlashError.invalidConfig(
                "DFlash2DraftModel requires dflash_config.selector_rank and selector_top_k")
        }
        let rope = json["rope_parameters"] as? [String: Any] ?? [:]
        let hiddenLayers = try reqInt(json, "num_hidden_layers")
        let layerTypes = (json["layer_types"] as? [String])
            ?? Array(repeating: "full_attention", count: hiddenLayers)
        let taps = (dfc["target_layer_ids"] as? [Any]) ?? (json["target_layer_ids"] as? [Any]) ?? []
        return DFlashConfig(
            hiddenSize: try reqInt(json, "hidden_size"),
            numHiddenLayers: hiddenLayers,
            numAttentionHeads: try reqInt(json, "num_attention_heads"),
            numKeyValueHeads: try reqInt(json, "num_key_value_heads"),
            headDim: try reqInt(json, "head_dim"),
            intermediateSize: try reqInt(json, "intermediate_size"),
            vocabSize: try reqInt(json, "vocab_size"),
            rmsNormEps: floatValue(json["rms_norm_eps"]) ?? 1e-6,
            ropeTheta: floatValue(json["rope_theta"]) ?? floatValue(rope["rope_theta"]) ?? 1_000_000,
            maxPositionEmbeddings: intValue(json["max_position_embeddings"]) ?? 262_144,
            blockSize: blockSize,
            targetLayerIds: taps.compactMap { intValue($0) },
            numTargetLayers: intValue(json["num_target_layers"]) ?? hiddenLayers,
            maskTokenId: intValue(dfc["mask_token_id"]) ?? intValue(json["mask_token_id"]) ?? 0,
            layerTypes: layerTypes,
            slidingWindow: intValue(json["sliding_window"]),
            finalLogitSoftcapping: floatValue(dfc["final_logit_softcapping"])
                ?? floatValue(json["final_logit_softcapping"]),
            selectorRank: selectorRank,
            selectorTopK: selectorTopK,
            convKernelSize: intValue(dfc["conv_kernel_size"]) ?? 0,
            convGroupSize: intValue(dfc["conv_group_size"]) ?? 16,
            outputMultiplier: floatValue(dfc["output_multiplier"]) ?? 1.0
        )
    }
}

public enum DFlashError: Error, LocalizedError {
    case invalidConfig(String)
    case missingBind
    case notATarget

    public var errorDescription: String? {
        switch self {
        case .invalidConfig(let s): return "DFlash config: \(s)"
        case .missingBind: return "DFlash drafter is not bound to a target embed/lm_head"
        case .notATarget: return "Loaded model is not a DFlash target"
        }
    }
}

private func reqInt(_ json: [String: Any], _ key: String) throws -> Int {
    guard let v = intValue(json[key]) else { throw DFlashError.invalidConfig("missing \(key)") }
    return v
}

private func intValue(_ any: Any?) -> Int? {
    switch any {
    case let i as Int: return i
    case let i as Int64: return Int(i)
    case let n as NSNumber: return n.intValue
    case let d as Double: return Int(d)
    default: return nil
    }
}

private func floatValue(_ any: Any?) -> Float? {
    switch any {
    case let f as Float: return f
    case let d as Double: return Float(d)
    case let i as Int: return Float(i)
    case let n as NSNumber: return n.floatValue
    default: return nil
    }
}

// MARK: - Grouped conv (DFlash 2)

public final class DFlashGroupedConv: Module {
    public let taps: Int
    public let groupSize: Int
    public let numGroups: Int

    @ParameterInfo(key: "base_kernel") public var baseKernel: MLXArray
    @ModuleInfo(key: "kernel_projection") public var kernelProjection: Linear

    public init(hiddenSize: Int, taps: Int, groupSize: Int) {
        precondition(hiddenSize % groupSize == 0, "conv_group_size must divide hidden_size")
        self.taps = taps
        self.groupSize = groupSize
        self.numGroups = hiddenSize / groupSize
        let tap0 = MLXArray.ones([2, 1, hiddenSize])
        let rest = MLXArray.zeros([2, taps - 1, hiddenSize])
        _baseKernel.wrappedValue = concatenated([tap0, rest], axis: 1)
        _kernelProjection.wrappedValue = Linear(hiddenSize, 2 * taps * numGroups, bias: false)
        super.init()
    }

    public func prepare(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let coeff = kernelProjection(x).reshaped(
            x.dim(0), x.dim(1), 2, taps, numGroups)
        let inDelta = coeff[0..., 0..., 0, 0..., 0...]
        let outDelta = coeff[0..., 0..., 1, 0..., 0...]
        return (convolve(x, delta: inDelta, side: 0), outDelta)
    }

    public func finish(_ y: MLXArray, _ delta: MLXArray) -> MLXArray {
        convolve(y, delta: delta, side: 1)
    }

    private func convolve(_ x: MLXArray, delta: MLXArray, side: Int) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let H = x.dim(2)
        let xg = x.reshaped(B, L, numGroups, groupSize)
        let base = baseKernel[side, 0..., 0...].reshaped(1, 1, taps, numGroups, groupSize)
        let coeff = base + delta.expandedDimensions(axis: -1)
        var out = coeff[0..., 0..., 0, 0..., 0...] * xg
        if taps > 1 {
            for t in 1 ..< taps {
                let pad = MLXArray.zeros([B, t, numGroups, groupSize], dtype: x.dtype)
                let body = xg[0..., ..<(L - t), 0..., 0...]
                let shifted = concatenated([pad, body], axis: 1)
                out = out + coeff[0..., 0..., t, 0..., 0...] * shifted
            }
        }
        return out.reshaped(B, L, H)
    }
}

// MARK: - Candidate selector

public final class DFlashCandidateSelector: Module {
    public let topK: Int
    public let rank: Int

    @ParameterInfo(key: "predecessor_codebook") public var predecessorCodebook: MLXArray
    @ParameterInfo(key: "successor_codebook") public var successorCodebook: MLXArray
    @ModuleInfo(key: "hidden_projection") public var hiddenProjection: Linear

    public init(hiddenSize: Int, vocabSize: Int, rank: Int, topK: Int) {
        self.topK = topK
        self.rank = rank
        _predecessorCodebook.wrappedValue = MLXArray.zeros([vocabSize, rank])
        _successorCodebook.wrappedValue = MLXArray.zeros([vocabSize, rank])
        _hiddenProjection.wrappedValue = Linear(hiddenSize, rank, bias: false)
        super.init()
    }

    public func lattice(
        candidateIds: MLXArray, unaryLogits: MLXArray, hidden: MLXArray, anchorId: Int
    ) -> MLXArray {
        let g = candidateIds.dim(0)
        let k = candidateIds.dim(1)
        let h = hiddenProjection(hidden)
        let anchor = MLXArray(Array(repeating: Int32(anchorId), count: k), [1, k])
        let predIds = concatenated([anchor, candidateIds[0 ..< (g - 1), 0...]], axis: 0)
        let predFlat = predIds.reshaped(-1)
        let sucFlat = candidateIds.reshaped(-1)
        let pre = take(predecessorCodebook, predFlat, axis: 0).reshaped(g, k, rank)
            * h.expandedDimensions(axis: 1)
        let suc = take(successorCodebook, sucFlat, axis: 0).reshaped(g, k, rank)
        let bilinear = matmul(pre, suc.transposed(0, 2, 1))
        let unary = unaryLogits.asType(.float32).expandedDimensions(axis: 1)
        return unary + bilinear.asType(.float32)
    }

    public func walkGreedy(scores: MLXArray, candidateIds: MLXArray) -> MLXArray {
        let g = scores.dim(0)
        func idAt(_ position: Int, _ index: MLXArray) -> MLXArray {
            take(candidateIds[position], index.reshaped([1]), axis: 0).reshaped([1])
        }
        var idx = argMax(scores[0, 0, 0...], axis: -1)
        var picks: [MLXArray] = [idAt(0, idx)]
        if g > 1 {
            for s in 1 ..< g {
                let row = take(scores[s], idx.reshaped([1]), axis: 0)
                idx = argMax(row, axis: -1)
                picks.append(idAt(s, idx))
            }
        }
        return concatenated(picks, axis: 0)
    }
}

// MARK: - Attention + layer

public final class DFlashCtxCache {
    var keys: MLXArray?
    var values: MLXArray?
    var offset: Int = 0
}

final class DFlashAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let scale: Float
    let isSliding: Bool
    let slidingWindow: Int?

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ config: DFlashConfig, layerIdx: Int) {
        let dim = config.hiddenSize
        self.nHeads = config.numAttentionHeads
        self.nKVHeads = config.numKeyValueHeads
        self.scale = pow(Float(config.headDim), -0.5)
        self.isSliding = layerIdx < config.layerTypes.count
            && config.layerTypes[layerIdx] == "sliding_attention"
        self.slidingWindow = isSliding ? config.slidingWindow : nil
        _qProj.wrappedValue = Linear(dim, nHeads * config.headDim, bias: false)
        _kProj.wrappedValue = Linear(dim, nKVHeads * config.headDim, bias: false)
        _vProj.wrappedValue = Linear(dim, nKVHeads * config.headDim, bias: false)
        _oProj.wrappedValue = Linear(nHeads * config.headDim, dim, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        super.init()
    }

    func appendCtx(_ xCtx: MLXArray, rope: RoPELayer, cache: DFlashCtxCache) -> (MLXArray, MLXArray) {
        var xCtx = xCtx
        var S = xCtx.dim(1)
        let B = xCtx.dim(0)
        if isSliding, let window = slidingWindow {
            let keepCtx = window - 1
            if keepCtx < S {
                let skip = S - keepCtx
                xCtx = xCtx[0..., skip..., 0...]
                S = xCtx.dim(1)
                cache.offset += skip
            }
        }
        var ctxKeys = kProj(xCtx)
        var ctxValues = vProj(xCtx)
        ctxKeys = kNorm(ctxKeys.reshaped(B, S, nKVHeads, -1)).transposed(0, 2, 1, 3)
        ctxValues = ctxValues.reshaped(B, S, nKVHeads, -1).transposed(0, 2, 1, 3)
        ctxKeys = applyRotaryPosition(rope, to: ctxKeys, offset: .scalar(cache.offset))
        if let prevK = cache.keys, let prevV = cache.values {
            cache.keys = concatenated([prevK, ctxKeys], axis: 2)
            cache.values = concatenated([prevV, ctxValues], axis: 2)
        } else {
            cache.keys = ctxKeys
            cache.values = ctxValues
        }
        cache.offset += S
        return (cache.keys!, cache.values!)
    }

    func callAsFunction(_ x: MLXArray, xCtx: MLXArray, rope: RoPELayer, cache: DFlashCtxCache)
        -> MLXArray
    {
        let B = x.dim(0)
        let L = x.dim(1)
        let (keys0, values0) = appendCtx(xCtx, rope: rope, cache: cache)
        var queries = qProj(x)
        var propKeys = kProj(x)
        var propValues = vProj(x)
        queries = qNorm(queries.reshaped(B, L, nHeads, -1)).transposed(0, 2, 1, 3)
        propKeys = kNorm(propKeys.reshaped(B, L, nKVHeads, -1)).transposed(0, 2, 1, 3)
        propValues = propValues.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        queries = applyRotaryPosition(rope, to: queries, offset: .scalar(cache.offset))
        propKeys = applyRotaryPosition(rope, to: propKeys, offset: .scalar(cache.offset))
        let keys = concatenated([keys0, propKeys], axis: 2)
        let values = concatenated([values0, propValues], axis: 2)
        let mask: MLXFast.ScaledDotProductAttentionMaskMode = isSliding ? .causal : .none
        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

final class DFlashDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DFlashAttention
    @ModuleInfo(key: "mlp") var mlp: Qwen3MLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm
    @ModuleInfo(key: "attention_conv") var attentionConv: DFlashGroupedConv?
    @ModuleInfo(key: "mlp_conv") var mlpConv: DFlashGroupedConv?

    init(_ config: DFlashConfig, layerIdx: Int) {
        _selfAttn.wrappedValue = DFlashAttention(config, layerIdx: layerIdx)
        _mlp.wrappedValue = Qwen3MLP(
            dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
        _inputLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        if config.convKernelSize > 0 {
            _attentionConv.wrappedValue = DFlashGroupedConv(
                hiddenSize: config.hiddenSize, taps: config.convKernelSize,
                groupSize: config.convGroupSize)
            _mlpConv.wrappedValue = DFlashGroupedConv(
                hiddenSize: config.hiddenSize, taps: config.convKernelSize,
                groupSize: config.convGroupSize)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, xCtx: MLXArray, rope: RoPELayer, cache: DFlashCtxCache)
        -> MLXArray
    {
        var a = inputLayernorm(x)
        var attnOutCoeff: MLXArray?
        if let conv = attentionConv {
            let prepared = conv.prepare(a)
            a = prepared.0
            attnOutCoeff = prepared.1
        }
        var attn = selfAttn(a, xCtx: xCtx, rope: rope, cache: cache)
        if let conv = attentionConv, let coeff = attnOutCoeff {
            attn = conv.finish(attn, coeff)
        }
        let h = x + attn
        var m = postAttentionLayernorm(h)
        var mlpOutCoeff: MLXArray?
        if let conv = mlpConv {
            let prepared = conv.prepare(m)
            m = prepared.0
            mlpOutCoeff = prepared.1
        }
        var mlpOut = mlp(m)
        if let conv = mlpConv, let coeff = mlpOutCoeff {
            mlpOut = conv.finish(mlpOut, coeff)
        }
        return h + mlpOut
    }
}

// MARK: - Draft model

public final class DFlashDraftModel: Module, LLMModel, KVCacheDimensionProvider {
    public let config: DFlashConfig
    public var vocabularySize: Int { config.vocabSize }
    public var kvHeads: [Int] {
        Array(repeating: config.numKeyValueHeads, count: config.numHiddenLayers)
    }
    let rope: RoPELayer

    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm
    @ModuleInfo(key: "layers") var layers: [DFlashDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "candidate_selector") var candidateSelector: DFlashCandidateSelector?

    public var embedTokens: Embedding?
    public var lmHead: ((MLXArray) -> MLXArray)?
    public var embedScale: Float = 1.0

    public init(_ config: DFlashConfig) {
        self.config = config
        _fc.wrappedValue = Linear(
            config.targetLayerIds.count * config.hiddenSize, config.hiddenSize, bias: false)
        _hiddenNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _layers.wrappedValue = (0 ..< config.numHiddenLayers).map {
            DFlashDecoderLayer(config, layerIdx: $0)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.rope = initializeRope(
            dims: config.headDim,
            base: config.ropeTheta,
            traditional: false,
            scalingConfig: nil,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )
        if config.isDFlash2 {
            _candidateSelector.wrappedValue = DFlashCandidateSelector(
                hiddenSize: config.hiddenSize, vocabSize: config.vocabSize,
                rank: config.selectorRank, topK: config.selectorTopK)
        }
        super.init()
    }

    public var loraLayers: [Module] { [] }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        MLXArray.zeros([max(inputs.dim(0), 1), 1, config.vocabSize])
    }

    public func bind(embed: Embedding, lmHead: @escaping (MLXArray) -> MLXArray, embedScale: Float = 1.0) {
        self.embedTokens = embed
        self.lmHead = lmHead
        self.embedScale = embedScale
    }

    public func makeCache() -> [DFlashCtxCache] {
        layers.map { _ in DFlashCtxCache() }
    }

    func projectCtx(_ targetHidden: MLXArray) -> MLXArray {
        hiddenNorm(fc(targetHidden))
    }

    func forwardHidden(
        _ inputs: MLXArray, targetHidden: MLXArray, cache: [DFlashCtxCache], logitsStart: Int
    ) -> MLXArray {
        guard let embedTokens else { return MLXArray.zeros([1, 1, config.hiddenSize]) }
        var h = embedTokens(inputs) * embedScale
        let hCtx = projectCtx(targetHidden)
        for (layer, c) in zip(layers, cache) {
            h = layer(h, xCtx: hCtx, rope: rope, cache: c)
        }
        if logitsStart > 0 {
            h = h[0..., logitsStart..., 0...]
        }
        return norm(h)
    }

    public func selectBlock(
        block: MLXArray, targetHidden: MLXArray, cache: [DFlashCtxCache],
        cap: Int, anchorId: Int
    ) -> MLXArray {
        let hiddenFull = forwardHidden(
            block, targetHidden: targetHidden, cache: cache, logitsStart: 1)
        // [B, L, H] → [cap, H]
        let seq = hiddenFull.dim(1)
        let takeN = Swift.min(cap, Swift.max(seq, 1))
        let hidden = hiddenFull[0, 0 ..< takeN, 0...]
        guard let sel = candidateSelector, let lmHead else {
            let logits = lmHead?(hidden) ?? hidden
            return argMax(logits, axis: -1)
        }
        var logits = lmHead(hidden)
        if logits.ndim == 3 {
            logits = logits[0, 0..., 0...]
        }
        let vocab = Swift.min(config.vocabSize, logits.dim(-1))
        if logits.dim(-1) > vocab {
            logits = logits[0..., 0 ..< vocab]
        }
        let k = Swift.min(sel.topK, vocab)
        let order = argSort(logits, axis: -1)
        let start = Swift.max(logits.dim(-1) - k, 0)
        let candidateIds = order[0..., start...].asType(.int32)
        let unary = MLX.takeAlong(logits, candidateIds, axis: -1).asType(.float32)
        let scores = sel.lattice(
            candidateIds: candidateIds, unaryLogits: unary, hidden: hidden, anchorId: anchorId)
        return sel.walkGreedy(scores: scores, candidateIds: candidateIds)
    }
}


