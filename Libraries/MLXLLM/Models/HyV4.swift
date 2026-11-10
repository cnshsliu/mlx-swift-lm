//
//  HyV4.swift
//  mlx-swift-lm
//
//  Port of mlx-lm models/hy_v4.py (Tencent Hunyuan Hy4 / HYV4ForCausalLM).
//  Gated MLA + DSA (IndexCache), iHC residuals, 256-expert MoE, native MTP.
//  Unlike mlx-lm sanitize, NovaMLX keeps `model.mtp_layers.*` and uses them
//  as an in-graph MtpDrafter.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct HyV4Configuration: Codable, Sendable {
    public var modelType: String = "hy_v4"
    public var vocabSize: Int = 120_832
    public var hiddenSize: Int = 6144
    public var intermediateSize: Int = 18432
    public var moeIntermediateSize: Int = 2048
    public var numHiddenLayers: Int = 78
    public var numAttentionHeads: Int = 64
    public var qLoraRank: Int = 2048
    public var kvLoraRank: Int = 512
    public var qkNopeHeadDim: Int = 192
    public var qkRopeHeadDim: Int = 64
    public var vHeadDim: Int = 256
    public var indexHeadDim: Int = 128
    public var indexNHeads: Int = 32
    public var indexTopk: Int = 2048
    public var indexerTypes: [String] = []
    public var mlpLayerTypes: [String] = []
    public var nRoutedExperts: Int = 256
    public var nSharedExperts: Int = 1
    public var numExpertsPerTok: Int = 8
    public var routedScalingFactor: Float = 2.827
    public var normTopkProb: Bool = true
    public var nGroup: Int = 1
    public var topkGroup: Int = 1
    public var swigluLimit: Float = 10.0
    public var hcMult: Int = 4
    public var hcMagnitude: Float = 2.0
    public var hcEps: Float = 1e-6
    public var gatingType: String = "elementwise"
    public var maxPositionEmbeddings: Int = 1_048_576
    public var rmsNormEps: Float = 1e-5
    public var ropeTheta: Float = 10_000_000
    public var ropeScaling: [String: StringOrNumber]?
    public var attentionBias: Bool = false
    public var numNextnPredictLayers: Int = 1
    public var tieWordEmbeddings: Bool = false

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case qLoraRank = "q_lora_rank"
        case kvLoraRank = "kv_lora_rank"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case indexHeadDim = "index_head_dim"
        case indexNHeads = "index_n_heads"
        case indexTopk = "index_topk"
        case indexerTypes = "indexer_types"
        case mlpLayerTypes = "mlp_layer_types"
        case nRoutedExperts = "n_routed_experts"
        case nSharedExperts = "n_shared_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case routedScalingFactor = "routed_scaling_factor"
        case normTopkProb = "norm_topk_prob"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case swigluLimit = "swiglu_limit"
        case hcMult = "hc_mult"
        case hcMagnitude = "hc_magnitude"
        case hcEps = "hc_eps"
        case gatingType = "gating_type"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case ropeParameters = "rope_parameters"
        case attentionBias = "attention_bias"
        case numNextnPredictLayers = "num_nextn_predict_layers"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "hy_v4"
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 120_832
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 6144
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 18432
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 2048
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 78
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 64
        qLoraRank = try c.decodeIfPresent(Int.self, forKey: .qLoraRank) ?? 2048
        kvLoraRank = try c.decodeIfPresent(Int.self, forKey: .kvLoraRank) ?? 512
        qkNopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkNopeHeadDim) ?? 192
        qkRopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim) ?? 64
        vHeadDim = try c.decodeIfPresent(Int.self, forKey: .vHeadDim) ?? 256
        indexHeadDim = try c.decodeIfPresent(Int.self, forKey: .indexHeadDim) ?? 128
        indexNHeads = try c.decodeIfPresent(Int.self, forKey: .indexNHeads) ?? 32
        indexTopk = try c.decodeIfPresent(Int.self, forKey: .indexTopk) ?? 2048
        indexerTypes = try c.decodeIfPresent([String].self, forKey: .indexerTypes) ?? []
        mlpLayerTypes = try c.decodeIfPresent([String].self, forKey: .mlpLayerTypes) ?? []
        nRoutedExperts = try c.decodeIfPresent(Int.self, forKey: .nRoutedExperts) ?? 256
        nSharedExperts = try c.decodeIfPresent(Int.self, forKey: .nSharedExperts) ?? 1
        numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 8
        routedScalingFactor = try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 2.827
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        nGroup = try c.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
        topkGroup = try c.decodeIfPresent(Int.self, forKey: .topkGroup) ?? 1
        swigluLimit = try c.decodeIfPresent(Float.self, forKey: .swigluLimit) ?? 10.0
        hcMult = try c.decodeIfPresent(Int.self, forKey: .hcMult) ?? 4
        hcMagnitude = try c.decodeIfPresent(Float.self, forKey: .hcMagnitude) ?? 2.0
        hcEps = try c.decodeIfPresent(Float.self, forKey: .hcEps) ?? 1e-6
        gatingType = try c.decodeIfPresent(String.self, forKey: .gatingType) ?? "elementwise"
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 1_048_576
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000_000
        ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        numNextnPredictLayers = try c.decodeIfPresent(Int.self, forKey: .numNextnPredictLayers) ?? 1
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        if let rp = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters) {
            if let theta = rp["rope_theta"]?.asFloat() {
                ropeTheta = theta
            }
            let ropeType = rp["rope_type"]?.asString() ?? rp["type"]?.asString() ?? "default"
            if ropeType != "default" {
                ropeScaling = rp
            }
        }

        if indexerTypes.isEmpty {
            indexerTypes = (0 ..< numHiddenLayers).map { i in
                (i == 0 || (i - 1) % 4 == 0) ? "full" : "shared"
            }
        }
        if mlpLayerTypes.isEmpty {
            mlpLayerTypes = ["dense"] + Array(repeating: "sparse", count: max(0, numHiddenLayers - 1))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(vocabSize, forKey: .vocabSize)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(intermediateSize, forKey: .intermediateSize)
        try c.encode(moeIntermediateSize, forKey: .moeIntermediateSize)
        try c.encode(numHiddenLayers, forKey: .numHiddenLayers)
        try c.encode(numAttentionHeads, forKey: .numAttentionHeads)
        try c.encode(qLoraRank, forKey: .qLoraRank)
        try c.encode(kvLoraRank, forKey: .kvLoraRank)
        try c.encode(qkNopeHeadDim, forKey: .qkNopeHeadDim)
        try c.encode(qkRopeHeadDim, forKey: .qkRopeHeadDim)
        try c.encode(vHeadDim, forKey: .vHeadDim)
        try c.encode(indexHeadDim, forKey: .indexHeadDim)
        try c.encode(indexNHeads, forKey: .indexNHeads)
        try c.encode(indexTopk, forKey: .indexTopk)
        try c.encode(indexerTypes, forKey: .indexerTypes)
        try c.encode(mlpLayerTypes, forKey: .mlpLayerTypes)
        try c.encode(nRoutedExperts, forKey: .nRoutedExperts)
        try c.encode(nSharedExperts, forKey: .nSharedExperts)
        try c.encode(numExpertsPerTok, forKey: .numExpertsPerTok)
        try c.encode(routedScalingFactor, forKey: .routedScalingFactor)
        try c.encode(normTopkProb, forKey: .normTopkProb)
        try c.encode(nGroup, forKey: .nGroup)
        try c.encode(topkGroup, forKey: .topkGroup)
        try c.encode(swigluLimit, forKey: .swigluLimit)
        try c.encode(hcMult, forKey: .hcMult)
        try c.encode(hcMagnitude, forKey: .hcMagnitude)
        try c.encode(hcEps, forKey: .hcEps)
        try c.encode(gatingType, forKey: .gatingType)
        try c.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try c.encode(rmsNormEps, forKey: .rmsNormEps)
        try c.encode(ropeTheta, forKey: .ropeTheta)
        try c.encodeIfPresent(ropeScaling, forKey: .ropeScaling)
        try c.encode(attentionBias, forKey: .attentionBias)
        try c.encode(numNextnPredictLayers, forKey: .numNextnPredictLayers)
        try c.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
    }

    func indexerType(at layer: Int) -> String {
        guard layer >= 0, layer < indexerTypes.count else { return "full" }
        return indexerTypes[layer]
    }

    func isDense(at layer: Int) -> Bool {
        guard layer >= 0, layer < mlpLayerTypes.count else { return layer == 0 }
        return mlpLayerTypes[layer] == "dense"
    }
}

extension StringOrNumber {
    fileprivate func asString() -> String? {
        if case .string(let s) = self { return s }
        return nil
    }
}

// MARK: - MultiLinear (MLA absorb)

class HyV4MultiLinear: Module, Quantizable {
    let inputDims: Int
    let outputDims: Int
    let numHeads: Int

    @ParameterInfo(key: "weight") var weight: MLXArray

    init(inputDims: Int, outputDims: Int, numHeads: Int) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numHeads = numHeads
        let scale = sqrt(1.0 / Float(inputDims))
        _weight.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale, [numHeads, outputDims, inputDims])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x.matmul(weight.swappedAxes(-1, -2))
    }

    func callNoTranspose(_ x: MLXArray) -> MLXArray {
        x.matmul(weight)
    }

    func toQuantized(groupSize: Int, bits: Int, mode: QuantizationMode) -> Module {
        let (qWeight, scales, biases) = MLX.quantized(
            weight, groupSize: groupSize, bits: bits, mode: mode)
        return HyV4QuantizedMultiLinear(
            weight: qWeight, scales: scales, biases: biases,
            inputDims: inputDims, outputDims: outputDims, numHeads: numHeads,
            groupSize: groupSize, bits: bits, mode: mode)
    }
}

class HyV4QuantizedMultiLinear: Module, Quantized {
    let groupSize: Int
    let bits: Int
    let mode: QuantizationMode
    let inputDims: Int
    let outputDims: Int
    let numHeads: Int

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "scales") var scales: MLXArray
    @ParameterInfo(key: "biases") var biases: MLXArray?

    init(
        weight: MLXArray, scales: MLXArray, biases: MLXArray?,
        inputDims: Int, outputDims: Int, numHeads: Int,
        groupSize: Int, bits: Int, mode: QuantizationMode
    ) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numHeads = numHeads
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        _weight.wrappedValue = weight
        _scales.wrappedValue = scales
        _biases.wrappedValue = biases
        super.init()
        freeze()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLX.quantizedMM(
            x, weight, scales: scales, biases: biases, transpose: true,
            groupSize: groupSize, bits: bits)
    }

    func callNoTranspose(_ x: MLXArray) -> MLXArray {
        MLX.quantizedMM(
            x, weight, scales: scales, biases: biases, transpose: false,
            groupSize: groupSize, bits: bits)
    }
}

func hyV4MultiLinear(_ module: Module, _ x: MLXArray, transpose: Bool = true) -> MLXArray {
    if let m = module as? HyV4MultiLinear {
        return transpose ? m(x) : m.callNoTranspose(x)
    }
    if let m = module as? HyV4QuantizedMultiLinear {
        return transpose ? m(x) : m.callNoTranspose(x)
    }
    fatalError("HyV4 expected MultiLinear, got \(type(of: module))")
}

// MARK: - Clamped SwitchGLU (routed experts)

class HyV4SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear
    let limit: Float

    init(inputDims: Int, hiddenDims: Int, numExperts: Int, limit: Float) {
        self.limit = limit
        _gateProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: false)
        _upProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: false)
        _downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])
        let doSort = indices.size >= 64
        var idx = indices
        var inverseOrder = MLXArray()
        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }
        let xUp = clip(upProj(x, idx, sortedIndices: doSort), min: -limit, max: limit)
        let xGate = silu(minimum(gateProj(x, idx, sortedIndices: doSort), MLXArray(limit)))
        x = downProj(xGate * xUp, idx, sortedIndices: doSort)
        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }
        return MLX.squeezed(x, axis: -2)
    }
}

// MARK: - Indexer (DSA)

class HyV4Indexer: Module {
    let nHeads: Int
    let headDim: Int
    let ropeHeadDim: Int
    let indexTopk: Int
    let softmaxScale: Float

    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "k_norm") var kNorm: LayerNorm
    @ModuleInfo(key: "weights_proj") var weightsProj: Linear
    let rope: RoPELayer

    init(_ args: HyV4Configuration) {
        self.nHeads = args.indexNHeads
        self.headDim = args.indexHeadDim
        self.ropeHeadDim = args.qkRopeHeadDim
        self.indexTopk = args.indexTopk
        self.softmaxScale = pow(Float(args.indexHeadDim), -0.5)
        _wqB.wrappedValue = Linear(args.qLoraRank, nHeads * headDim, bias: false)
        _wk.wrappedValue = Linear(args.hiddenSize, headDim, bias: false)
        _kNorm.wrappedValue = LayerNorm(dimensions: headDim, eps: 1e-6)
        _weightsProj.wrappedValue = Linear(args.hiddenSize, nHeads, bias: false)
        self.rope = initializeRope(
            dims: args.qkRopeHeadDim, base: args.ropeTheta, traditional: true,
            scalingConfig: args.ropeScaling, maxPositionEmbeddings: args.maxPositionEmbeddings)
        super.init()
    }

    private func applyRope(_ x: MLXArray, offset: RoPEOffset?) -> MLXArray {
        let splitX = split(x, indices: [headDim - ropeHeadDim], axis: -1)
        return concatenated([splitX[0], applyRotaryPosition(rope, to: splitX[1], offset: offset)], axis: -1)
    }

    func callAsFunction(
        _ x: MLXArray, qr: MLXArray, mask: MLXArray?, cache: KVCache?
    ) -> MLXArray? {
        let b = x.dim(0)
        let s = x.dim(1)
        var k = kNorm(wk(x)).reshaped(b, 1, s, headDim)
        let offset = cache?.ropeOffset
        k = applyRope(k, offset: offset)
        if let cache {
            (k, _) = cache.update(keys: k, values: k)
            if let simple = cache as? KVCacheSimple, simple.state.count >= 2 {
                simple.state = [simple.state[0], MLXArray.zeros(like: simple.state[0])]
            }
        }
        if k.dim(2) <= indexTopk {
            return nil
        }
        var q = wqB(qr).reshaped(b, s, nHeads, headDim).swappedAxes(1, 2)
        q = applyRope(q, offset: offset)
        var scores = q.matmul(k.swappedAxes(-1, -2))
        scores = maximum(scores, MLXArray(0, dtype: scores.dtype))
        let weights = weightsProj(x) * (pow(Float(nHeads), -0.5) * softmaxScale)
        scores = weights[.ellipsis, .newAxis, 0...].matmul(scores.transposed(0, 2, 1, 3))
        scores = scores.transposed(0, 2, 1, 3)
        if let mask {
            scores = MLX.where(mask, scores, MLXArray(-Float.infinity, dtype: scores.dtype))
        }
        let kth = scores.dim(-1) - indexTopk
        return argPartition(scores, kth: kth, axis: -1)[.ellipsis, kth...]
    }
}

// MARK: - Gated MLA + DSA attention

class HyV4Attention: Module {
    let skipTopk: Bool
    let numHeads: Int
    let qkRopeHeadDim: Int
    let kvLoraRank: Int
    let vHeadDim: Int
    let qkNopeHeadDim: Int
    let qHeadDim: Int
    let scale: Float
    let rope: RoPELayer

    @ModuleInfo(key: "q_a_proj") var qAProj: Linear
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "embed_q") var embedQ: Module
    @ModuleInfo(key: "unembed_out") var unembedOut: Module
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "linear_gate") var linearGate: Linear
    @ParameterInfo(key: "learnable_sink_param") var learnableSinkParam: MLXArray
    @ModuleInfo(key: "indexer") var indexer: HyV4Indexer?

    init(_ config: HyV4Configuration, layerIdx: Int) {
        self.skipTopk = config.indexerType(at: layerIdx) == "shared"
        self.numHeads = config.numAttentionHeads
        self.qkRopeHeadDim = config.qkRopeHeadDim
        self.kvLoraRank = config.kvLoraRank
        self.vHeadDim = config.vHeadDim
        self.qkNopeHeadDim = config.qkNopeHeadDim
        self.qHeadDim = config.qkNopeHeadDim + config.qkRopeHeadDim
        self.scale = pow(Float(qHeadDim), -0.5)

        _qAProj.wrappedValue = Linear(config.hiddenSize, config.qLoraRank, bias: config.attentionBias)
        _qALayerNorm.wrappedValue = RMSNorm(dimensions: config.qLoraRank, eps: config.rmsNormEps)
        _qBProj.wrappedValue = Linear(config.qLoraRank, numHeads * qHeadDim, bias: false)
        _kvAProjWithMqa.wrappedValue = Linear(
            config.hiddenSize, kvLoraRank + qkRopeHeadDim, bias: config.attentionBias)
        _kvALayerNorm.wrappedValue = RMSNorm(dimensions: kvLoraRank, eps: config.rmsNormEps)
        _embedQ.wrappedValue = HyV4MultiLinear(
            inputDims: qkNopeHeadDim, outputDims: kvLoraRank, numHeads: numHeads)
        _unembedOut.wrappedValue = HyV4MultiLinear(
            inputDims: kvLoraRank, outputDims: vHeadDim, numHeads: numHeads)
        _oProj.wrappedValue = Linear(numHeads * vHeadDim, config.hiddenSize, bias: config.attentionBias)
        _linearGate.wrappedValue = Linear(config.hiddenSize, numHeads * vHeadDim, bias: false)
        _learnableSinkParam.wrappedValue = MLXArray.zeros([numHeads])
        if !skipTopk {
            _indexer.wrappedValue = HyV4Indexer(config)
        }
        self.rope = initializeRope(
            dims: qkRopeHeadDim, base: config.ropeTheta, traditional: true,
            scalingConfig: config.ropeScaling, maxPositionEmbeddings: config.maxPositionEmbeddings)
        super.init()
    }

    private func kvCache(_ cache: KVCache?) -> KVCache? {
        if let list = cache as? CacheList { return list[0] }
        return cache
    }

    private func indexCache(_ cache: KVCache?) -> KVCache? {
        guard !skipTopk, let list = cache as? CacheList else { return nil }
        return list[1]
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray?,
        cache: KVCache?,
        topkIndices: MLXArray?
    ) -> (MLXArray, MLXArray?) {
        let B = x.dim(0)
        let L = x.dim(1)
        let qr = qALayerNorm(qAProj(x))
        let q = qBProj(qr).reshaped(B, L, numHeads, qHeadDim).transposed(0, 2, 1, 3)
        let splitQ = split(q, indices: [qkNopeHeadDim], axis: -1)
        var qNope = splitQ[0]
        var qPe = splitQ[1]

        var compressedKv = kvAProjWithMqa(x)
        let splitKv = split(compressedKv, indices: [kvLoraRank], axis: -1)
        compressedKv = splitKv[0]
        var kPe = splitKv[1].reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)
        var kvLatent = kvALayerNorm(compressedKv)

        let kv = kvCache(cache)
        let offset = kv?.ropeOffset
        qPe = applyRotaryPosition(rope, to: qPe, offset: offset)
        kPe = applyRotaryPosition(rope, to: kPe, offset: offset)
        kvLatent = expandedDimensions(kvLatent, axis: 1)

        if let kv {
            (kvLatent, kPe) = kv.update(keys: kvLatent, values: kPe)
        }

        var topk = topkIndices
        var attnMask = mask
        if !skipTopk, let indexer {
            topk = indexer(x, qr: qr, mask: mask, cache: indexCache(cache))
        }
        if let topk {
            if L == 1 {
                let idx = topk[0..., 0..., 0, 0..., .newAxis]
                let latentTarget = idx.shape.dropLast() + [kvLatent.dim(-1)]
                kvLatent = takeAlong(kvLatent, broadcast(idx, to: Array(latentTarget)), axis: 2)
                let peTarget = idx.shape.dropLast() + [kPe.dim(-1)]
                kPe = takeAlong(kPe, broadcast(idx, to: Array(peTarget)), axis: 2)
                if let currentMask = attnMask {
                    attnMask = takeAlong(currentMask, topk, axis: -1)
                }
            } else {
                var shape = topk.shape
                shape[shape.count - 1] = kvLatent.dim(2)
                var sparse = MLXArray.zeros(shape, dtype: .bool)
                sparse = putAlong(sparse, topk, values: MLXArray(true), axis: -1)
                if let attnMask {
                    sparse = sparse .&& attnMask
                }
                attnMask = sparse
            }
        }

        var peScores = (qPe * scale).matmul(kPe.swappedAxes(-1, -2))
        if let attnMask {
            peScores = MLX.where(
                attnMask, peScores,
                MLXArray(-Float.greatestFiniteMagnitude, dtype: peScores.dtype))
        }

        let k: MLXArray
        let v: MLXArray
        if L == 1 {
            qNope = hyV4MultiLinear(embedQ, qNope)
            k = kvLatent
            v = kvLatent
        } else {
            k = hyV4MultiLinear(embedQ, kvLatent, transpose: false)
            v = hyV4MultiLinear(unembedOut, kvLatent)
        }

        var output = MLXFast.scaledDotProductAttention(
            queries: qNope, keys: k, values: v, scale: scale,
            mask: .array(peScores),
            sinks: learnableSinkParam.asType(qNope.dtype))
        if L == 1 {
            output = hyV4MultiLinear(unembedOut, output)
        }
        output = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        output = output * sigmoid(linearGate(x))
        return (oProj(output), topk)
    }
}

// MARK: - MLP / MoE

class HyV4MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

class HyV4MoEGate: Module {
    let topK: Int
    let normTopkProb: Bool
    let nGroup: Int
    let topkGroup: Int
    let routedScalingFactor: Float

    @ParameterInfo var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ config: HyV4Configuration) {
        self.topK = config.numExpertsPerTok
        self.normTopkProb = config.normTopkProb
        self.nGroup = max(config.nGroup, 1)
        self.topkGroup = config.topkGroup
        self.routedScalingFactor = config.routedScalingFactor
        _weight.wrappedValue = MLXArray.zeros([config.nRoutedExperts, config.hiddenSize])
        _eScoreCorrectionBias.wrappedValue = MLXArray.zeros([config.nRoutedExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let orig = sigmoid(x.matmul(weight.T).asType(.float32))
        var scores = orig + eScoreCorrectionBias
        if nGroup > 1 {
            scores = unflatten(scores, axis: -1, shape: [nGroup, -1])
            let groupScores = top(scores, k: 2, axis: -1).sum(axis: -1, keepDims: true)
            let k = nGroup - topkGroup
            let groupIdx = argPartition(groupScores, kth: k - 1, axis: -2)[.ellipsis, ..<k, 0...]
            scores = putAlong(scores, stopGradient(groupIdx), values: MLXArray(0.0), axis: -2)
            scores = flattened(scores, start: -2, end: -1)
        }
        let inds = argPartition(-scores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var taken = takeAlong(orig, inds, axis: -1)
        if topK > 1, normTopkProb {
            taken = taken / taken.sum(axis: -1, keepDims: true)
        }
        taken = taken * routedScalingFactor
        return (inds, taken)
    }
}

class HyV4MoE: Module, UnaryLayer {
    @ModuleInfo(key: "switch_mlp") var switchMLP: HyV4SwitchGLU
    @ModuleInfo(key: "gate") var gate: HyV4MoEGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: HyV4MLP?

    init(_ config: HyV4Configuration) {
        _switchMLP.wrappedValue = HyV4SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts,
            limit: config.swigluLimit)
        _gate.wrappedValue = HyV4MoEGate(config)
        if config.nSharedExperts > 0 {
            _sharedExperts.wrappedValue = HyV4MLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.moeIntermediateSize * config.nSharedExperts)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, scores) = gate(x)
        var y = switchMLP(x, inds)
        y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)
        if let shared = sharedExperts {
            y = y + shared(x)
        }
        return y
    }
}

// MARK: - iHC

class HyV4HCPre: Module {
    @ParameterInfo(key: "hc_fn") var hcFn: MLXArray
    @ParameterInfo(key: "hc_scale") var hcScale: MLXArray
    @ParameterInfo(key: "hc_base") var hcBase: MLXArray

    init(_ config: HyV4Configuration) {
        let hc = config.hcMult
        _hcFn.wrappedValue = MLXArray.zeros([2 * hc, hc * config.hiddenSize], dtype: .float32)
        _hcScale.wrappedValue = MLXArray.ones([2], dtype: .float32)
        _hcBase.wrappedValue = MLXArray.zeros([2 * hc], dtype: .float32)
        super.init()
    }
}

class HyV4HCLayer: Module {
    let hcMult: Int
    let magnitude: Float
    let hcEps: Float
    let rmsNormEps: Float
    @ModuleInfo(key: "hc_pre") var hcPre: HyV4HCPre

    init(_ config: HyV4Configuration) {
        self.hcMult = config.hcMult
        self.magnitude = config.hcMagnitude
        self.hcEps = config.hcEps
        self.rmsNormEps = config.rmsNormEps
        _hcPre.wrappedValue = HyV4HCPre(config)
        super.init()
    }

    func collapse(_ h: MLXArray) -> (MLXArray, MLXArray) {
        let hf = h.asType(.float32)
        let flat = flattened(hf, start: -2, end: -1)
        let rms = rsqrt(mean(square(flat), axis: -1, keepDims: true) + rmsNormEps)
        let mixes = flat.matmul(hcPre.hcFn.T) * rms
        let pre = sigmoid(mixes[.ellipsis, ..<hcMult] * hcPre.hcScale[0] + hcPre.hcBase[..<hcMult]) + hcEps
        let post =
            magnitude
            * sigmoid(mixes[.ellipsis, hcMult...] * hcPre.hcScale[1] + hcPre.hcBase[hcMult...])
            + hcEps
        let x = (pre[.ellipsis, .newAxis] * hf).sum(axis: -2)
        return (x.asType(h.dtype), post)
    }

    func expand(out: MLXArray, residual: MLXArray, post: MLXArray) -> MLXArray {
        let y = post[.ellipsis, .newAxis] * out[.ellipsis, .newAxis, 0...].asType(.float32)
        return (y + residual.asType(.float32)).asType(out.dtype)
    }
}

class HyV4HCHead: Module {
    let hcEps: Float
    let rmsNormEps: Float
    @ParameterInfo(key: "hc_head_fn") var hcHeadFn: MLXArray
    @ParameterInfo(key: "hc_head_scale") var hcHeadScale: MLXArray
    @ParameterInfo(key: "hc_head_base") var hcHeadBase: MLXArray

    init(_ config: HyV4Configuration) {
        let hc = config.hcMult
        self.hcEps = config.hcEps
        self.rmsNormEps = config.rmsNormEps
        _hcHeadFn.wrappedValue = MLXArray.zeros([hc, hc * config.hiddenSize], dtype: .float32)
        _hcHeadScale.wrappedValue = MLXArray.ones([1], dtype: .float32)
        _hcHeadBase.wrappedValue = MLXArray.zeros([hc], dtype: .float32)
        super.init()
    }

    func callAsFunction(_ h: MLXArray) -> MLXArray {
        let hf = h.asType(.float32)
        let flat = flattened(hf, start: -2, end: -1)
        let rms = rsqrt(mean(square(flat), axis: -1, keepDims: true) + rmsNormEps)
        let mixes = flat.matmul(hcHeadFn.T) * rms
        let pre = sigmoid(mixes * hcHeadScale + hcHeadBase) + hcEps
        return (pre[.ellipsis, .newAxis] * hf).sum(axis: -2).asType(h.dtype)
    }
}

// MARK: - Decoder

class HyV4DecoderLayer: Module {
    let skipTopk: Bool
    @ModuleInfo(key: "self_attn") var selfAttn: HyV4Attention
    @ModuleInfo(key: "mlp") var mlp: Module
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "hc_attn_layer") var hcAttnLayer: HyV4HCLayer
    @ModuleInfo(key: "hc_mlp_layer") var hcMlpLayer: HyV4HCLayer

    init(_ config: HyV4Configuration, layerIdx: Int) {
        self.skipTopk = config.indexerType(at: layerIdx) == "shared"
        _selfAttn.wrappedValue = HyV4Attention(config, layerIdx: layerIdx)
        if config.isDense(at: layerIdx) {
            _mlp.wrappedValue = HyV4MLP(hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        } else {
            _mlp.wrappedValue = HyV4MoE(config)
        }
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _hcAttnLayer.wrappedValue = HyV4HCLayer(config)
        _hcMlpLayer.wrappedValue = HyV4HCLayer(config)
        super.init()
    }

    func callAsFunction(
        _ h: MLXArray, mask: MLXArray?, cache: KVCache?, topkIndices: MLXArray?
    ) -> (MLXArray, MLXArray?) {
        let residualAttn = h
        let (xAttn, postAttn) = hcAttnLayer.collapse(h)
        let (rAttn, topk) = selfAttn(
            inputLayerNorm(xAttn), mask: mask, cache: cache, topkIndices: topkIndices)
        var out = hcAttnLayer.expand(out: rAttn, residual: residualAttn, post: postAttn)

        let residualMlp = out
        let (xMlp, postMlp) = hcMlpLayer.collapse(out)
        let rMlp = (mlp as! UnaryLayer)(postAttentionLayerNorm(xMlp))
        out = hcMlpLayer.expand(out: rMlp, residual: residualMlp, post: postMlp)
        return (out, topk)
    }
}

// MARK: - Native MTP block (no iHC; DeepSeek-style eh_proj)

class HyV4MTPBlock: Module {
    @ModuleInfo(key: "enorm") var enorm: RMSNorm
    @ModuleInfo(key: "hnorm") var hnorm: RMSNorm
    @ModuleInfo(key: "eh_proj") var ehProj: Linear
    @ModuleInfo(key: "final_layernorm") var finalLayerNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var selfAttn: HyV4Attention
    @ModuleInfo(key: "mlp") var mlp: Module
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: HyV4Configuration) {
        _enorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _hnorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _ehProj.wrappedValue = Linear(2 * config.hiddenSize, config.hiddenSize, bias: false)
        _finalLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _selfAttn.wrappedValue = HyV4Attention(config, layerIdx: 0)
        if config.nRoutedExperts > 0 {
            _mlp.wrappedValue = HyV4MoE(config)
        } else {
            _mlp.wrappedValue = HyV4MLP(
                hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        }
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        tokenEmbed: MLXArray, hidden: MLXArray, mask: MLXArray?, cache: KVCache?
    ) -> MLXArray {
        var h = ehProj(concatenated([enorm(tokenEmbed), hnorm(hidden)], axis: -1))
        let (r, _) = selfAttn(inputLayerNorm(h), mask: mask, cache: cache, topkIndices: nil)
        h = h + r
        h = h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
        return finalLayerNorm(h)
    }
}

// MARK: - Inner model

public class HyV4ModelInner: Module {
    let args: HyV4Configuration
    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    var layers: [HyV4DecoderLayer]
    @ModuleInfo(key: "hc_head") var hcHead: HyV4HCHead
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "mtp_layers") var mtpLayers: [HyV4MTPBlock]

    init(_ args: HyV4Configuration) {
        self.args = args
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabSize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.numHiddenLayers).map { HyV4DecoderLayer(args, layerIdx: $0) }
        _hcHead.wrappedValue = HyV4HCHead(args)
        _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        let nMtp = max(0, args.numNextnPredictLayers)
        _mtpLayers.wrappedValue = (0 ..< nMtp).map { _ in HyV4MTPBlock(args) }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(x)
        let kv: KVCache? = {
            guard let first = cache?.first else { return nil }
            if let list = first as? CacheList { return list[0] }
            return first
        }()
        let maskMode = createAttentionMask(h: h, cache: kv, returnArray: true)
        let maskArray: MLXArray? = {
            switch maskMode {
            case .array(let a): return a
            case .none: return nil
            default:
                return createCausalMask(n: h.dim(1), offset: kv?.offset ?? 0)
            }
        }()

        let b = h.dim(0)
        let s = h.dim(1)
        h = expandedDimensions(h, axis: 2)
        h = broadcast(h, to: [b, s, args.hcMult, args.hiddenSize])

        var topk: MLXArray? = nil
        for (i, layer) in layers.enumerated() {
            (h, topk) = layer(h, mask: maskArray, cache: cache?[i], topkIndices: topk)
        }
        return norm(hcHead(h))
    }
}

// MARK: - Public model

public class HyV4Model: Module, LLMModel, KVCacheDimensionProvider, MtpTarget, MtpDrafter {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let args: HyV4Configuration
    public var nativeMtpAvailable = false

    public var model: HyV4ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    private var embedFn: ((MLXArray) -> MLXArray)?
    private var lmHeadFn: ((MLXArray) -> MLXArray)?

    public var mtpBlockSize: Int { nativeMtpAvailable ? 3 : 0 }

    public init(_ args: HyV4Configuration) {
        self.args = args
        self.vocabularySize = args.vocabSize
        self.kvHeads = Array(repeating: 1, count: args.numHiddenLayers)
        self.model = HyV4ModelInner(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        mtpLmHead(model(inputs, cache: cache))
    }

    public func mtpEmbed(_ tokens: MLXArray) -> MLXArray {
        model.embedTokens(tokens)
    }

    public func mtpLmHead(_ hidden: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hidden)
        }
        return model.embedTokens.asLinear(hidden)
    }

    public func mtpHiddenAndLogits(_ tokens: MLXArray, cache: [KVCache]?) -> (
        hidden: MLXArray, logits: MLXArray
    ) {
        let hidden = model(tokens, cache: cache)
        return (hidden, mtpLmHead(hidden))
    }

    public func bindMtp(
        embed: @escaping (MLXArray) -> MLXArray, lmHead: @escaping (MLXArray) -> MLXArray
    ) {
        embedFn = embed
        lmHeadFn = lmHead
    }

    public func mtpForward(tokenEmbed: MLXArray, hidden: MLXArray, cache: [KVCache]?) -> MLXArray {
        precondition(nativeMtpAvailable && !model.mtpLayers.isEmpty, "HyV4 native MTP weights missing")
        let kv: KVCache? = {
            guard let first = cache?.first else { return nil }
            if let list = first as? CacheList { return list[0] }
            return first
        }()
        let maskMode = createAttentionMask(h: tokenEmbed, cache: kv, returnArray: true)
        let maskArray: MLXArray? = {
            switch maskMode {
            case .array(let a): return a
            case .none: return nil
            default:
                return createCausalMask(n: tokenEmbed.dim(1), offset: kv?.offset ?? 0)
            }
        }()
        var h = hidden
        for (i, block) in model.mtpLayers.enumerated() {
            h = block(tokenEmbed: tokenEmbed, hidden: h, mask: maskArray, cache: cache?[i])
        }
        return h
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        model.layers.map { layer in
            if layer.skipTopk {
                return CacheList(KVCacheSimple())
            }
            return CacheList(KVCacheSimple(), KVCacheSimple())
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        nativeMtpAvailable = weights.keys.contains { $0.contains("mtp_layers") }

        var weights = weights.filter { key, _ in
            let parts = key.split(separator: ".")
            if parts.count >= 3, parts[0] == "model", parts[1] == "layers",
                let idx = Int(parts[2])
            {
                return idx < args.numHiddenLayers
            }
            return true
        }

        func splitKvB(prefix: String) {
            let wKey = "\(prefix).kv_b_proj.weight"
            guard let v = weights.removeValue(forKey: wKey) else { return }
            let quantized = weights["\(prefix).kv_b_proj.scales"] != nil
            var dense = v
            var bits = 4
            var groupSize = 64
            if quantized {
                let dims = args.kvLoraRank
                let scales = weights.removeValue(forKey: "\(prefix).kv_b_proj.scales")!
                let biases = weights.removeValue(forKey: "\(prefix).kv_b_proj.biases")
                bits = max(1, (v.dim(-1) * 32) / max(dims, 1))
                groupSize = max(1, dims / max(scales.dim(-1), 1))
                dense = MLX.dequantized(
                    v, scales: scales, biases: biases, groupSize: groupSize, bits: bits,
                    mode: .affine)
            }
            let headDim = args.qkNopeHeadDim + args.vHeadDim
            let reshaped = dense.reshaped(args.numAttentionHeads, headDim, -1)
            let wk = reshaped[0..., ..<args.qkNopeHeadDim, 0...].swappedAxes(-1, -2)
            let wv = reshaped[0..., args.qkNopeHeadDim..., 0...]
            if quantized {
                let (qWk, wkScales, wkBiases) = MLX.quantized(wk, groupSize: groupSize, bits: bits)
                let (qWv, wvScales, wvBiases) = MLX.quantized(wv, groupSize: groupSize, bits: bits)
                weights["\(prefix).embed_q.weight"] = qWk
                weights["\(prefix).unembed_out.weight"] = qWv
                weights["\(prefix).embed_q.scales"] = wkScales
                weights["\(prefix).unembed_out.scales"] = wvScales
                weights["\(prefix).embed_q.biases"] = wkBiases
                weights["\(prefix).unembed_out.biases"] = wvBiases
            } else {
                weights["\(prefix).embed_q.weight"] = wk
                weights["\(prefix).unembed_out.weight"] = wv
            }
        }

        func splitExperts(prefix: String) {
            if let gateUp = weights.removeValue(forKey: "\(prefix).experts.gate_up_proj") {
                let dims = args.moeIntermediateSize
                weights["\(prefix).switch_mlp.gate_proj.weight"] = gateUp[0..., ..<dims, 0...]
                weights["\(prefix).switch_mlp.up_proj.weight"] = gateUp[0..., dims..., 0...]
            }
            for suffix in ["weight", "scales", "biases"] {
                let fused = "\(prefix).experts.gate_up_proj.\(suffix)"
                if let gateUp = weights.removeValue(forKey: fused) {
                    if suffix == "weight" || suffix == "scales" || suffix == "biases" {
                        let dims: Int
                        if suffix == "weight" {
                            dims = args.moeIntermediateSize
                        } else {
                            dims = gateUp.dim(1) / 2
                        }
                        weights["\(prefix).switch_mlp.gate_proj.\(suffix)"] = gateUp[0..., ..<dims, 0...]
                        weights["\(prefix).switch_mlp.up_proj.\(suffix)"] = gateUp[0..., dims..., 0...]
                    }
                }
            }
            if let down = weights.removeValue(forKey: "\(prefix).experts.down_proj") {
                weights["\(prefix).switch_mlp.down_proj.weight"] = down
            }
            for suffix in ["weight", "scales", "biases"] {
                let src = "\(prefix).experts.down_proj.\(suffix)"
                if let down = weights.removeValue(forKey: src) {
                    weights["\(prefix).switch_mlp.down_proj.\(suffix)"] = down
                }
            }
        }

        for l in 0 ..< args.numHiddenLayers {
            splitExperts(prefix: "model.layers.\(l).mlp")
            splitKvB(prefix: "model.layers.\(l).self_attn")
        }
        for l in 0 ..< args.numNextnPredictLayers {
            splitExperts(prefix: "model.mtp_layers.\(l).mlp")
            splitKvB(prefix: "model.mtp_layers.\(l).self_attn")
        }

        if args.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }
        return weights
    }
}

extension HyV4Model: LoRAModel {
    public var loraLayers: [Module] { model.layers }
}

extension HyV4Model {
    /// Draft-only cache for native MTP (not the backbone CacheList stack).
    public func mtpNewCache(parameters: GenerateParameters?) -> [KVCache] {
        model.mtpLayers.map { _ in CacheList(KVCacheSimple(), KVCacheSimple()) }
    }
}
