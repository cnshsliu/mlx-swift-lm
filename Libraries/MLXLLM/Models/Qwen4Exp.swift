//
//  Qwen4Exp.swift
//  mlx-swift-lm
//
//  Port of oMLX mlx_vlm.models.qwen4_exp (Qwen3.8-Flash-Next / qwen4_exp).
//  Language path: Gated DeltaNet + Qwen Sparse Attention, mHC residuals,
//  512-expert MoE, and layer-1 n-gram PLE. Resident PLE only (no SSD mmap).
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct Qwen4ExpTextConfiguration: Codable, Sendable {
    public var modelType: String = "qwen4_exp_text"
    public var hiddenSize: Int = 2560
    public var hiddenLayers: Int = 48
    public var attentionHeads: Int = 24
    public var kvHeads: Int = 2
    public var headDim: Int = 256
    public var linearNumValueHeads: Int = 48
    public var linearNumKeyHeads: Int = 16
    public var linearKeyHeadDim: Int = 128
    public var linearValueHeadDim: Int = 128
    public var linearConvKernelDim: Int = 4
    public var numExperts: Int = 512
    public var numExpertsPerTok: Int = 10
    public var sharedExpertIntermediateSize: Int = 640
    public var moeIntermediateSize: Int = 640
    public var rmsNormEps: Float = 1e-6
    public var vocabularySize: Int = 248_320
    public var maxPositionEmbeddings: Int = 262_144
    public var hcCount: Int = 4
    public var hcLowrank: Int = 320
    public var fullAttentionInterval: Int = 4
    public var layerTypes: [String] = []
    public var pleLayerIds: [Int] = [2]
    public var pleEmbedDim: Int = 2560
    public var pleConvKernelSize: Int = 4
    public var ngramSize: Int = 3
    public var headsPerNgram: Int = 8
    public var ngramVocabSizeBase: Int = 20_000_000
    public var makeNgramVocabSizeDivisibleBy: Int = 128
    public var splitNgramParts: Int = 128
    public var seed: Int = 1234
    public var indexerNHeads: Int = 4
    public var indexerKvHeads: Int = 1
    public var indexerHeadDim: Int = 128
    public var indexerBudget: Int = 2048
    public var indexerCompressRatio: Int = 4
    public var outputGateType: String = "sigmoid"
    public var hiddenAct: String = "silu"
    public var normTopkProb: Bool = true
    public var tieWordEmbeddings: Bool = false
    public var attentionBias: Bool = false
    public var eosTokenId: Int = 248_044
    public var ropeTheta: Float = 10_000_000
    public var partialRotaryFactor: Float = 0.25
    public var ropeScaling: [String: StringOrNumber]?
    public var mtpNumHiddenLayers: Int = 0

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case hcCount = "hc_count"
        case hcLowrank = "hc_lowrank"
        case fullAttentionInterval = "full_attention_interval"
        case layerTypes = "layer_types"
        case pleLayerIds = "ple_layer_ids"
        case pleEmbedDim = "ple_embed_dim"
        case pleConvKernelSize = "ple_conv_kernel_size"
        case ngramSize = "ngram_size"
        case headsPerNgram = "heads_per_ngram"
        case ngramVocabSizeBase = "ngram_vocab_size_base"
        case makeNgramVocabSizeDivisibleBy = "make_ngram_vocab_size_divisible_by"
        case splitNgramParts = "split_ngram_parts"
        case seed
        case indexerNHeads = "indexer_n_heads"
        case indexerKvHeads = "indexer_kv_heads"
        case indexerHeadDim = "indexer_head_dim"
        case indexerBudget = "indexer_budget"
        case indexerCompressRatio = "indexer_compress_ratio"
        case outputGateType = "output_gate_type"
        case hiddenAct = "hidden_act"
        case normTopkProb = "norm_topk_prob"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case eosTokenId = "eos_token_id"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case ropeScaling = "rope_scaling"
        case mtpNumHiddenLayers = "mtp_num_hidden_layers"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen4_exp_text"
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2560
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 48
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 24
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 2
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? max(1, hiddenSize / max(1, attentionHeads))
        linearNumValueHeads = try c.decodeIfPresent(Int.self, forKey: .linearNumValueHeads) ?? 48
        linearNumKeyHeads = try c.decodeIfPresent(Int.self, forKey: .linearNumKeyHeads) ?? 16
        linearKeyHeadDim = try c.decodeIfPresent(Int.self, forKey: .linearKeyHeadDim) ?? 128
        linearValueHeadDim = try c.decodeIfPresent(Int.self, forKey: .linearValueHeadDim) ?? 128
        linearConvKernelDim = try c.decodeIfPresent(Int.self, forKey: .linearConvKernelDim) ?? 4
        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 512
        numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 10
        sharedExpertIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 640
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 640
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 248_320
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 262_144
        hcCount = try c.decodeIfPresent(Int.self, forKey: .hcCount) ?? 4
        hcLowrank = try c.decodeIfPresent(Int.self, forKey: .hcLowrank) ?? 320
        fullAttentionInterval = try c.decodeIfPresent(Int.self, forKey: .fullAttentionInterval) ?? 4
        layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes) ?? []
        pleLayerIds = try c.decodeIfPresent([Int].self, forKey: .pleLayerIds) ?? []
        pleEmbedDim = try c.decodeIfPresent(Int.self, forKey: .pleEmbedDim) ?? hiddenSize
        pleConvKernelSize = try c.decodeIfPresent(Int.self, forKey: .pleConvKernelSize) ?? 4
        ngramSize = try c.decodeIfPresent(Int.self, forKey: .ngramSize) ?? 3
        headsPerNgram = try c.decodeIfPresent(Int.self, forKey: .headsPerNgram) ?? 8
        ngramVocabSizeBase = try c.decodeIfPresent(Int.self, forKey: .ngramVocabSizeBase) ?? 20_000_000
        makeNgramVocabSizeDivisibleBy =
            try c.decodeIfPresent(Int.self, forKey: .makeNgramVocabSizeDivisibleBy) ?? 128
        splitNgramParts = try c.decodeIfPresent(Int.self, forKey: .splitNgramParts) ?? 128
        seed = try c.decodeIfPresent(Int.self, forKey: .seed) ?? 1234
        indexerNHeads = try c.decodeIfPresent(Int.self, forKey: .indexerNHeads) ?? 4
        indexerKvHeads = try c.decodeIfPresent(Int.self, forKey: .indexerKvHeads) ?? 1
        indexerHeadDim = try c.decodeIfPresent(Int.self, forKey: .indexerHeadDim) ?? 128
        indexerBudget = try c.decodeIfPresent(Int.self, forKey: .indexerBudget) ?? 2048
        indexerCompressRatio = try c.decodeIfPresent(Int.self, forKey: .indexerCompressRatio) ?? 4
        outputGateType = try c.decodeIfPresent(String.self, forKey: .outputGateType) ?? "sigmoid"
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        if let eos = try? c.decode(Int.self, forKey: .eosTokenId) {
            eosTokenId = eos
        } else if let eosList = try? c.decode([Int].self, forKey: .eosTokenId), let first = eosList.first {
            eosTokenId = first
        } else {
            eosTokenId = 248_044
        }

        enum RopeKey: String, CodingKey { case ropeParameters = "rope_parameters" }
        if let ropeParameters = try decoder.container(keyedBy: RopeKey.self)
            .decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters)
        {
            var rp = ropeParameters
            if rp["type"] == nil, let ropeType = rp["rope_type"] {
                rp["type"] = ropeType
            }
            ropeTheta = rp["rope_theta"]?.asFloat() ?? 10_000_000
            partialRotaryFactor = rp["partial_rotary_factor"]?.asFloat() ?? 0.25
            ropeScaling = rp
        } else {
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000_000
            partialRotaryFactor =
                try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.25
            ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        }
        mtpNumHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .mtpNumHiddenLayers) ?? 0

        if layerTypes.isEmpty {
            layerTypes = (0 ..< hiddenLayers).map { i in
                ((i + 1) % fullAttentionInterval == 0) ? "qwen_sparse_attention" : "linear_attention"
            }
        } else {
            layerTypes = layerTypes.map { $0 == "full_attention" ? "qwen_sparse_attention" : $0 }
        }
        pleLayerIds = Array(Set(pleLayerIds)).sorted()
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(hiddenLayers, forKey: .hiddenLayers)
        try c.encode(attentionHeads, forKey: .attentionHeads)
        try c.encode(kvHeads, forKey: .kvHeads)
        try c.encode(headDim, forKey: .headDim)
        try c.encode(vocabularySize, forKey: .vocabularySize)
        try c.encode(hcCount, forKey: .hcCount)
        try c.encode(layerTypes, forKey: .layerTypes)
        try c.encode(pleLayerIds, forKey: .pleLayerIds)
        try c.encode(eosTokenId, forKey: .eosTokenId)
    }

    public func isLinear(layerIdx: Int) -> Bool {
        layerIdx < layerTypes.count
            ? layerTypes[layerIdx] == "linear_attention"
            : ((layerIdx + 1) % fullAttentionInterval != 0)
    }
}

public struct Qwen4ExpConfiguration: Codable, Sendable {
    public var modelType: String = "qwen4_exp"
    public var textConfig: Qwen4ExpTextConfiguration
    public var imageTokenId: Int = 248_056
    public var videoTokenId: Int = 248_057
    public var visionStartTokenId: Int = 248_053
    public var visionEndTokenId: Int = 248_054
    public var vocabSize: Int = 248_320

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
        case visionStartTokenId = "vision_start_token_id"
        case visionEndTokenId = "vision_end_token_id"
        case vocabSize = "vocab_size"
    }

    public init(textConfig: Qwen4ExpTextConfiguration = Qwen4ExpTextConfiguration()) {
        self.textConfig = textConfig
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen4_exp"
        if let nested = try c.decodeIfPresent(Qwen4ExpTextConfiguration.self, forKey: .textConfig) {
            textConfig = nested
        } else {
            textConfig = try Qwen4ExpTextConfiguration(from: decoder)
        }
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 248_056
        videoTokenId = try c.decodeIfPresent(Int.self, forKey: .videoTokenId) ?? 248_057
        visionStartTokenId = try c.decodeIfPresent(Int.self, forKey: .visionStartTokenId) ?? 248_053
        visionEndTokenId = try c.decodeIfPresent(Int.self, forKey: .visionEndTokenId) ?? 248_054
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? textConfig.vocabularySize
    }
}

// MARK: - Helpers

private func isPrime(_ value: Int) -> Bool {
    if value < 2 { return false }
    if value % 2 == 0 { return value == 2 }
    var divisor = 3
    while divisor * divisor <= value {
        if value % divisor == 0 { return false }
        divisor += 2
    }
    return true
}

private func nthPrimeAfter(start: Int, count: Int) -> Int {
    var prime = start
    for _ in 0 ..< count {
        prime += 1
        while !isPrime(prime) { prime += 1 }
    }
    return prime
}

private func splitmix64(_ value: UInt64) -> UInt64 {
    var z = value &+ 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
    return z ^ (z >> 31)
}

private func buildLayerMultipliers(
    unigramVocabSize: Int, ngramSize: Int, pleLayerIndex: Int, seed: Int
) -> MLXArray {
    let maxLong = Int64.max
    let multiplierMax = maxLong / Int64(max(unigramVocabSize, 1))
    let halfBound = max(1, multiplierMax / 2)
    let baseSeed = UInt64(bitPattern: Int64(seed)) &+ 10007 &* UInt64(pleLayerIndex)
    var multipliers = [Int64]()
    multipliers.reserveCapacity(ngramSize)
    for index in 0 ..< ngramSize {
        let value = baseSeed &+ 0x9E3779B97F4A7C15 &* UInt64(index + 1)
        let odd = 2 * Int64(splitmix64(value) % UInt64(halfBound)) + 1
        multipliers.append(odd)
    }
    return MLXArray(multipliers)
}

private func rotaryAtPositions(
    _ x: MLXArray, positions: MLXArray, theta: Float, rotaryDims: Int
) -> MLXArray {
    let dim = x.dim(-1)
    let rd = min(max(rotaryDims, 2), dim)
    let half = rd / 2
    guard half > 0 else { return x }

    var pos = positions.asType(.float32)
    if pos.ndim == 3 {
        pos = pos[0]
    }
    if pos.ndim == 1 {
        pos = pos.expandedDimensions(axis: 0)
    }
    // pos: [B, L]  x: [B, H, L, D]
    let freqIdx = MLXArray(Array(0 ..< half)).asType(.float32)
    let freqs = 1.0 / MLX.pow(MLXArray(theta), freqIdx * (2.0 / Float(rd)))
    let angles = pos[0..., .newAxis, 0..., .newAxis] * freqs[.newAxis, .newAxis, .newAxis, 0...]
    let cos = MLX.cos(angles).asType(x.dtype)
    let sin = MLX.sin(angles).asType(x.dtype)

    let rotated = x[0..., 0..., 0..., 0 ..< rd]
    let rest = rd < dim ? x[0..., 0..., 0..., rd...] : nil
    let x1 = rotated[0..., 0..., 0..., 0 ..< half]
    let x2 = rotated[0..., 0..., 0..., half...]
    let rotatedOut = concatenated([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis: -1)
    if let rest {
        return concatenated([rotatedOut, rest], axis: -1)
    }
    return rotatedOut
}

// MARK: - Cache

public final class Qwen4ExpLinearCache: ArraysCache {
    public init() {
        super.init(size: 4)
    }

    public override func copy() -> any KVCache {
        let new = Qwen4ExpLinearCache()
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.offset = self.offset
        return new
    }
}

public final class QSAKVCache: KVCacheSimple {
    public var indexKeys: MLXArray?
    public var indexPositionIds: MLXArray?

    public func updateIndexer(keys: MLXArray, positionIds: MLXArray) -> (MLXArray, MLXArray) {
        if let existing = indexKeys, let existingPos = indexPositionIds {
            indexKeys = concatenated([existing, keys], axis: 1)
            var newPos = positionIds
            var cached = existingPos
            if cached.ndim == 3 && newPos.ndim == 2 {
                newPos = broadcast(newPos.expandedDimensions(axis: 0), to: [cached.dim(0), newPos.dim(0), newPos.dim(1)])
            } else if cached.ndim == 2 && newPos.ndim == 3 {
                cached = broadcast(cached.expandedDimensions(axis: 0), to: [newPos.dim(0), cached.dim(0), cached.dim(1)])
            }
            indexPositionIds = concatenated([cached, newPos], axis: -1)
        } else {
            indexKeys = keys
            indexPositionIds = positionIds
        }
        return (indexKeys!, indexPositionIds!)
    }

    public override func copy() -> any KVCache {
        let new = QSAKVCache()
        new.step = self.step
        let s = super.state
        if s.count == 2 {
            new.state = s.map { $0[.ellipsis] }
        }
        new.offset = self.offset
        new.indexKeys = indexKeys.map { $0[.ellipsis] }
        new.indexPositionIds = indexPositionIds.map { $0[.ellipsis] }
        return new
    }

    public override func trim(_ n: Int) -> Int {
        let trimmed = super.trim(n)
        if trimmed > 0, let keys = indexKeys {
            let keep = keys.dim(1) - trimmed
            if keep > 0 {
                indexKeys = keys[0..., 0 ..< keep, 0...]
                if let pos = indexPositionIds {
                    indexPositionIds = pos[.ellipsis, 0 ..< keep]
                }
            }
        }
        return trimmed
    }
}

// MARK: - Norms

final class Qwen4ExpRMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float
    let groupSize: Int?

    init(dimensions: Int, eps: Float, groupSize: Int? = nil) {
        self.eps = eps
        self.groupSize = groupSize
        self._weight.wrappedValue = MLXArray.zeros([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let dtype = x.dtype
        var y = x.asType(.float32)
        let weight = self.weight.asType(.float32)
        if let groupSize {
            var shape = y.shape
            shape[shape.count - 1] = -1
            shape.append(groupSize)
            y = y.reshaped(shape)
            let w = weight.reshaped([-1, groupSize])
            y = y * rsqrt(y.square().mean(axis: -1, keepDims: true) + eps)
            y = y * (1.0 + w)
            y = y.reshaped(x.shape)
        } else {
            y = y * rsqrt(y.square().mean(axis: -1, keepDims: true) + eps)
            y = y * (1.0 + weight)
        }
        return y.asType(dtype)
    }
}

final class Qwen4ExpRMSNormGated: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float
    let activation: String

    init(dimensions: Int, eps: Float, activation: String) {
        self.eps = eps
        self.activation = activation
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, gate: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let y = MLXFast.rmsNorm(x, weight: weight, eps: eps).asType(.float32)
        var g = gate.asType(.float32)
        if activation == "sigmoid" {
            g = sigmoid(g)
        } else {
            g = silu(g)
        }
        return (y * g).asType(dtype)
    }
}

// MARK: - GDN / Attention / MoE

final class Qwen4ExpGatedDeltaNet: Module {
    let hiddenSize: Int
    let numVHeads: Int
    let numKHeads: Int
    let headKDim: Int
    let headVDim: Int
    let keyDim: Int
    let valueDim: Int
    let convKernelSize: Int
    let convDim: Int

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ModuleInfo(key: "norm") var norm: Qwen4ExpRMSNormGated
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ args: Qwen4ExpTextConfiguration) {
        self.hiddenSize = args.hiddenSize
        self.numVHeads = args.linearNumValueHeads
        self.numKHeads = args.linearNumKeyHeads
        self.headKDim = args.linearKeyHeadDim
        self.headVDim = args.linearValueHeadDim
        self.keyDim = headKDim * numKHeads
        self.valueDim = headVDim * numVHeads
        self.convKernelSize = args.linearConvKernelDim
        self.convDim = keyDim * 2 + valueDim

        _conv1d.wrappedValue = Conv1d(
            inputChannels: convDim,
            outputChannels: convDim,
            kernelSize: convKernelSize,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: convDim,
            bias: false
        )
        _inProjQKV.wrappedValue = Linear(hiddenSize, keyDim * 2 + valueDim, bias: false)
        _inProjZ.wrappedValue = Linear(hiddenSize, valueDim, bias: false)
        _inProjB.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)
        _inProjA.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)
        _dtBias.wrappedValue = MLXArray.ones([numVHeads])
        _aLog.wrappedValue = log(MLXRandom.uniform(low: 0, high: 16, [numVHeads]))
        _norm.wrappedValue = Qwen4ExpRMSNormGated(
            dimensions: headVDim, eps: args.rmsNormEps, activation: args.outputGateType)
        _outProj.wrappedValue = Linear(valueDim, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, mask: MLXArray? = nil, cache: ArraysCache? = nil)
        -> MLXArray
    {
        let B = inputs.dim(0)
        let S = inputs.dim(1)

        var qkv = inProjQKV(inputs)
        let z = inProjZ(inputs).reshaped(B, S, numVHeads, headVDim)
        let b = inProjB(inputs)
        let a = inProjA(inputs)

        let convState: MLXArray
        if let cacheState = cache?[0] {
            convState = cacheState
        } else {
            convState = MLXArray.zeros([B, convKernelSize - 1, convDim], dtype: inputs.dtype)
        }

        if let mask {
            qkv = MLX.where(mask[.ellipsis, .newAxis], qkv, 0)
        }

        let convInput = concatenated([convState, qkv], axis: 1)
        if let cache {
            cache[0] = convInput[0..., (-(convKernelSize - 1))...]
            cache.offset += S
        }

        let convOut = silu(conv1d(convInput))
        let convSplit = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        var q = convSplit[0].reshaped(B, S, numKHeads, headKDim)
        var k = convSplit[1].reshaped(B, S, numKHeads, headKDim)
        let v = convSplit[2].reshaped(B, S, numVHeads, headVDim)

        let scale = pow(Float(headKDim), -0.5)
        q = q * rsqrt(q.square().sum(axis: -1, keepDims: true) + 1e-6) * scale
        k = k * rsqrt(k.square().sum(axis: -1, keepDims: true) + 1e-6)

        let state = cache?[1]
        let (out, newState) = gatedDeltaUpdate(
            q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias, state: state, mask: mask)
        if let cache {
            cache[1] = newState
        }

        let normalized = norm(out, gate: z)
        return outProj(normalized.reshaped(B, S, -1))
    }
}

final class Qwen4ExpQSAIndexer: Module {
    let nHeads: Int
    let kvHeads: Int
    let headDim: Int
    let tokenBudget: Int
    let compressRatio: Int
    let blockTopk: Int
    let ropeTheta: Float
    let rotaryDims: Int

    @ModuleInfo(key: "index_qk_proj") var indexQkProj: Linear
    @ModuleInfo(key: "q_layernorm") var qLayernorm: Qwen4ExpRMSNorm
    @ModuleInfo(key: "k_layernorm") var kLayernorm: Qwen4ExpRMSNorm

    init(_ args: Qwen4ExpTextConfiguration) {
        self.nHeads = args.indexerNHeads
        self.kvHeads = args.indexerKvHeads
        self.headDim = args.indexerHeadDim
        self.tokenBudget = args.indexerBudget
        self.compressRatio = max(1, args.indexerCompressRatio)
        self.blockTopk = tokenBudget / compressRatio
        self.ropeTheta = args.ropeTheta
        self.rotaryDims = max(2, Int(Float(headDim) * args.partialRotaryFactor))

        _indexQkProj.wrappedValue = Linear(
            args.hiddenSize, (nHeads + kvHeads) * headDim, bias: false)
        _qLayernorm.wrappedValue = Qwen4ExpRMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kLayernorm.wrappedValue = Qwen4ExpRMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ hiddenStates: MLXArray, cache: QSAKVCache?, positionIds: MLXArray?
    ) -> MLXArray? {
        fromProjected(indexQkProj(hiddenStates), cache: cache, positionIds: positionIds)
    }

    func fromProjected(_ qk: MLXArray, cache: QSAKVCache?, positionIds: MLXArray?) -> MLXArray? {
        let B = qk.dim(0)
        let S = qk.dim(1)
        let pastLen = cache?.offset ?? 0
        var pos = positionIds
        if pos == nil {
            let positions = MLXArray(Array(pastLen ..< (pastLen + S)))
            pos = broadcast(positions.expandedDimensions(axis: 0), to: [B, S])
        }

        let split = qk.reshaped(B, S, nHeads + kvHeads, headDim)
        var query = split[0..., 0..., 0 ..< nHeads, 0...]
        var rawKeys = split[0..., 0..., nHeads..., 0...].squeezed(axis: 2)
        query = qLayernorm(query).transposed(0, 2, 1, 3)

        let fullPositionIds: MLXArray
        if let cache {
            (rawKeys, fullPositionIds) = cache.updateIndexer(keys: rawKeys, positionIds: pos!)
        } else {
            fullPositionIds = pos!
        }

        let keyLen = rawKeys.dim(1)
        let maxCompleteBlocks = keyLen / compressRatio
        if maxCompleteBlocks <= blockTopk {
            return nil
        }

        query = rotaryAtPositions(query, positions: pos!, theta: ropeTheta, rotaryDims: rotaryDims)

        let completeKeyLen = maxCompleteBlocks * compressRatio
        var pooled = rawKeys[0..., 0 ..< completeKeyLen, 0...]
            .reshaped(B, maxCompleteBlocks, compressRatio, headDim)
        pooled = kLayernorm(pooled.asType(.float32).mean(axis: 2).asType(rawKeys.dtype))
            .expandedDimensions(axis: 1)

        let blockStarts = MLXArray(Array(0 ..< maxCompleteBlocks).map { $0 * compressRatio })
        let blockPos: MLXArray
        if fullPositionIds.ndim == 3 {
            blockPos = takeAlong(fullPositionIds, blockStarts.reshaped([1, 1, maxCompleteBlocks]), axis: -1)
        } else {
            blockPos = takeAlong(
                fullPositionIds, broadcast(blockStarts.expandedDimensions(axis: 0), to: [B, maxCompleteBlocks]),
                axis: -1)
        }
        pooled = rotaryAtPositions(pooled, positions: blockPos, theta: ropeTheta, rotaryDims: rotaryDims)

        var scores = (query.asType(.float32).matmul(pooled.asType(.float32).transposed(0, 1, 3, 2)))
            .sum(axis: 1)
        scores = scores / sqrt(Float(headDim))

        let queryEnds = MLXArray(Array((pastLen + 1) ... (pastLen + S)))
        let completeCounts = queryEnds / compressRatio
        let blockIdx = MLXArray(Array(0 ..< maxCompleteBlocks))
        let validBlocks =
            blockIdx.reshaped([1, 1, maxCompleteBlocks])
            .< completeCounts.reshaped([1, S, 1])
        scores = MLX.where(validBlocks, scores, MLXArray(-Float.greatestFiniteMagnitude))

        let kth = max(0, scores.dim(-1) - blockTopk)
        let selected = MLX.argPartition(scores, kth: kth, axis: -1)[.ellipsis, kth...]
        let kthScores = takeAlong(
            scores, selected[0..., 0..., 0 ..< 1], axis: -1)
        var selectedTokens = scores .>= kthScores
        selectedTokens = MLX.repeated(selectedTokens, count: compressRatio, axis: -1)
        if completeKeyLen < keyLen {
            let pad = MLXArray.zeros([B, S, keyLen - completeKeyLen], dtype: selectedTokens.dtype)
            selectedTokens = concatenated([selectedTokens, pad], axis: -1)
        }

        let tokenIndices = MLXArray(Array(0 ..< keyLen))
        let tailStarts = completeCounts * compressRatio
        let tail =
            (tokenIndices.reshaped([1, 1, keyLen]) .>= tailStarts.reshaped([1, S, 1]))
            .&& (tokenIndices.reshaped([1, 1, keyLen]) .< queryEnds.reshaped([1, S, 1]))
        let causal = tokenIndices.reshaped([1, 1, keyLen]) .< queryEnds.reshaped([1, S, 1])
        let useSparse = completeCounts .> blockTopk
        selectedTokens = MLX.where(useSparse.reshaped([1, S, 1]), selectedTokens .|| tail, causal)
        return selectedTokens.expandedDimensions(axis: 1)
    }
}

final class Qwen4ExpAttention: Module {
    let attentionHeads: Int
    let kvHeads: Int
    let scale: Float
    let headDim: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Qwen4ExpRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Qwen4ExpRMSNorm
    @ModuleInfo(key: "indexer") var indexer: Qwen4ExpQSAIndexer

    let rope: RoPELayer

    init(_ args: Qwen4ExpTextConfiguration) {
        let headDim = args.headDim
        self.headDim = headDim
        self.attentionHeads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(
            args.hiddenSize, args.attentionHeads * headDim * 2, bias: args.attentionBias)
        _kProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _vProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _oProj.wrappedValue = Linear(
            args.attentionHeads * headDim, args.hiddenSize, bias: args.attentionBias)
        _qNorm.wrappedValue = Qwen4ExpRMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = Qwen4ExpRMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _indexer.wrappedValue = Qwen4ExpQSAIndexer(args)

        let ropeDims = Int(Float(headDim) * args.partialRotaryFactor)
        self.rope = initializeRope(
            dims: max(1, ropeDims),
            base: args.ropeTheta,
            traditional: false,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings
        )
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
        positionIds: MLXArray?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let qsaCache = cache as? QSAKVCache
        var attnMask = mask
        if let qsa = indexer(x, cache: qsaCache, positionIds: positionIds) {
            switch mask {
            case .none, .causal:
                attnMask = .array(qsa)
            case .array(let existing):
                if existing.dtype == .bool {
                    attnMask = .array(existing .&& qsa)
                } else {
                    let bias = MLX.where(
                        qsa, MLXArray(0, dtype: existing.dtype),
                        MLXArray(-Float.greatestFiniteMagnitude, dtype: existing.dtype))
                    attnMask = .array(existing + bias)
                }
            case .arrays(let arrays):
                if let existing = arrays.first {
                    attnMask = .array(existing.dtype == .bool ? existing .&& qsa : existing)
                } else {
                    attnMask = .array(qsa)
                }
            }
        }

        let qProjOutput = qProj(x)
        let qSplit = qProjOutput.reshaped(B, L, attentionHeads, -1).split(parts: 2, axis: -1)
        var queries = qSplit[0]
        let gate = qSplit[1].reshaped(B, L, -1)
        var keys = kProj(x)
        var values = vProj(x)

        queries = qNorm(queries).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: attnMask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(sigmoidMultiply(output, gate))
    }
}

final class Qwen4ExpSparseMoeBlock: Module, UnaryLayer {
    let normTopkProb: Bool
    let numExperts: Int
    let topK: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: Qwen3NextMLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    init(_ args: Qwen4ExpTextConfiguration) {
        self.normTopkProb = args.normTopkProb
        self.numExperts = max(args.numExperts, 1)
        self.topK = max(1, min(args.numExpertsPerTok, max(args.numExperts, 1)))
        _gate.wrappedValue = Linear(args.hiddenSize, self.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: self.numExperts
        )
        _sharedExpert.wrappedValue = Qwen3NextMLP(
            dimensions: args.hiddenSize, hiddenDimensions: args.sharedExpertIntermediateSize)
        _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gates = softmax(gate(x), axis: -1, precise: true)
        let kth = gates.dim(-1) - topK
        let inds = MLX.argPartition(gates, kth: kth, axis: -1)[.ellipsis, kth...]
        var scores = takeAlong(gates, inds, axis: -1)
        if normTopkProb {
            scores = scores / scores.sum(axis: -1, keepDims: true)
        }
        let y = switchMLP(x, inds)
        let combined = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)
        var sharedY = sharedExpert(x)
        sharedY = sigmoid(sharedExpertGate(x)) * sharedY
        return combined + sharedY
    }
}

final class Qwen4ExpGatedResidual: Module {
    let hcCount: Int
    let hiddenSize: Int
    let useCombine: Bool

    @ModuleInfo(key: "hc_norm") var hcNorm: Qwen4ExpRMSNorm
    @ModuleInfo(key: "input_mix_weight_down") var inputMixDown: Linear
    @ModuleInfo(key: "input_mix_weight_up") var inputMixUp: Linear
    @ModuleInfo(key: "block_inject_weight") var blockInject: Linear?

    init(_ args: Qwen4ExpTextConfiguration, useCombine: Bool = true) {
        self.hcCount = args.hcCount
        self.hiddenSize = args.hiddenSize
        self.useCombine = useCombine
        let hcHidden = hcCount * hiddenSize
        _hcNorm.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: hcHidden, eps: args.rmsNormEps, groupSize: hiddenSize)
        _inputMixDown.wrappedValue = Linear(hcHidden, args.hcLowrank, bias: false)
        _inputMixUp.wrappedValue = Linear(args.hcLowrank, hcHidden, bias: false)
        if useCombine {
            _blockInject.wrappedValue = Linear(hcHidden, hcCount, bias: false)
        }
        super.init()
    }

    func mixOnly(_ hyperInput: MLXArray) -> MLXArray {
        let normed = hcNorm(hyperInput)
        var mix = silu(inputMixDown(normed) / Float(hcCount))
        mix = sigmoid(inputMixUp(mix))
        mix = mix.reshaped(mix.shape.dropLast() + [hcCount, hiddenSize])
        let streams = normed.reshaped(normed.shape.dropLast() + [hcCount, hiddenSize])
        return (mix * streams).mean(axis: -2)
    }

    func callAsFunction(_ hyperInput: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let normed = hcNorm(hyperInput)
        var mix = silu(inputMixDown(normed) / Float(hcCount))
        mix = sigmoid(inputMixUp(mix))
        mix = mix.reshaped(mix.shape.dropLast() + [hcCount, hiddenSize])
        let streams = normed.reshaped(normed.shape.dropLast() + [hcCount, hiddenSize])
        let mixedInput = (mix * streams).mean(axis: -2)
        guard let blockInject else {
            return (mixedInput, hyperInput, MLXArray.ones(mixedInput.shape.dropLast() + [hcCount]))
        }
        let injectionWeights = 2 * sigmoid(blockInject(normed) / Float(hcCount))
        return (mixedInput, hyperInput, injectionWeights)
    }
}

// MARK: - PLE

final class Qwen4ExpShardedEmbedding: Module {
    let shardOffsets: [Int]
    let dims: Int
    @ModuleInfo(key: "shards") var shards: [Embedding]
    let weightScale: MLXArray

    init(numEmbeddings: Int, dims: Int, numShards: Int) {
        let shardsCount = max(1, min(numShards, numEmbeddings))
        let base = numEmbeddings / shardsCount
        let remainder = numEmbeddings % shardsCount
        var sizes = (0 ..< shardsCount).map { $0 < remainder ? base + 1 : base }
        if sizes.contains(0) {
            sizes = [numEmbeddings]
        }
        var offsets = [0]
        for size in sizes { offsets.append(offsets.last! + size) }
        self.shardOffsets = offsets
        self.dims = dims
        self._shards.wrappedValue = sizes.map { Embedding(embeddingCount: max($0, 1), dimensions: dims) }
        self.weightScale = MLXArray.ones([1])
        super.init()
    }

    func callAsFunction(_ indices: MLXArray) -> MLXArray {
        let shape = indices.shape
        let flat = indices.reshaped([-1]).asType(.int32)
        if flat.size == 0 {
            return MLXArray.zeros(shape + [dims], dtype: shards[0].weight.dtype)
        }
        let offs = MLXArray(shardOffsets.map { Int32($0) })
        let ge = expandedDimensions(flat, axis: 1) .>= expandedDimensions(offs, axis: 0)
        let shardIdx = clip(ge.asType(.int32).sum(axis: 1) - 1, min: 0, max: shards.count - 1)
        let starts = takeAlong(
            expandedDimensions(offs, axis: 0), expandedDimensions(shardIdx, axis: 1), axis: 1
        ).reshaped([-1])
        let local = flat - starts
        var result = MLXArray.zeros([flat.size, dims], dtype: shards[0].weight.dtype)
        for s in 0 ..< shards.count {
            let mask = shardIdx .== MLXArray(Int32(s))
            let vals = shards[s](local) * weightScale
            result = MLX.which(expandedDimensions(mask, axis: 1), vals, result)
        }
        return result.reshaped(shape + [dims])
    }
}

final class Qwen4ExpNGramEmbedding: Module {
    let ngramSize: Int
    let contextLen: Int
    let headsPerNgram: Int
    let ngramHeads: Int
    let eosTokenId: Int
    @ParameterInfo(key: "layer_multipliers") var layerMultipliers: MLXArray
    @ParameterInfo(key: "ngram_heads_vocab_sizes") var ngramHeadsVocabSizes: MLXArray
    @ParameterInfo(key: "ngram_heads_offsets") var ngramHeadsOffsets: MLXArray

    @ModuleInfo(key: "ngram_embedding") var ngramEmbedding: Qwen4ExpShardedEmbedding

    init(_ args: Qwen4ExpTextConfiguration, embeddingDim: Int, layerIdx: Int, pleLayerIndex: Int) {
        self.ngramSize = max(args.ngramSize, 2)
        self.contextLen = ngramSize - 1
        self.headsPerNgram = args.headsPerNgram
        self.ngramHeads = contextLen * headsPerNgram
        self.eosTokenId = args.eosTokenId
        self._layerMultipliers.wrappedValue = buildLayerMultipliers(
            unigramVocabSize: args.vocabularySize,
            ngramSize: ngramSize,
            pleLayerIndex: pleLayerIndex,
            seed: args.seed
        )

        var headVocabSizes = [Int]()
        var headOffsets = [Int]()
        var total = 0
        for headIdx in 0 ..< ngramHeads {
            let globalHead = pleLayerIndex * ngramHeads + headIdx
            let size = nthPrimeAfter(start: args.ngramVocabSizeBase - 1, count: globalHead + 1)
            headVocabSizes.append(size)
            headOffsets.append(total)
            total += size
        }
        self._ngramHeadsVocabSizes.wrappedValue = MLXArray(headVocabSizes.map { Int64($0) })
        self._ngramHeadsOffsets.wrappedValue = MLXArray(headOffsets.map { Int64($0) })

        let divisor = max(args.makeNgramVocabSizeDivisibleBy, 1)
        let padded = ((total + divisor - 1) / divisor) * divisor
        let perHeadDim = max(1, embeddingDim / max(ngramHeads, 1))
        _ngramEmbedding.wrappedValue = Qwen4ExpShardedEmbedding(
            numEmbeddings: max(padded, 1),
            dims: perHeadDim,
            numShards: max(1, args.splitNgramParts)
        )
        super.init()
        _ = layerIdx
    }

    private func shiftRightIgnoreEos(_ tokenIds: MLXArray, shift: Int) -> MLXArray {
        if shift == 0 { return tokenIds }
        let B = tokenIds.dim(0)
        let L = tokenIds.dim(1)
        let positions = MLXArray(Array(0 ..< L)).asType(.int64)
        let eosPositions = MLX.where(
            tokenIds .== MLXArray(Int64(eosTokenId)), positions, MLXArray(Int64(-1)))
        let previousInclusive = eosPositions.cummax(axis: 1)
        let prefix = MLXArray.full([B, 1], values: MLXArray(Int64(-1)), dtype: .int64)
        let previousEos = concatenated([prefix, previousInclusive[0..., 0 ..< (L - 1)]], axis: 1)
        let segmentStart = previousEos + 1
        let positionInSegment = positions.expandedDimensions(axis: 0) - segmentStart
        let sourcePositions = positions - shift
        let gather = MLX.maximum(sourcePositions, 0)
        let shifted = takeAlong(
            tokenIds, broadcast(gather.expandedDimensions(axis: 0), to: [B, L]), axis: 1)
        let valid =
            (positionInSegment .>= shift) .&& (sourcePositions.expandedDimensions(axis: 0) .>= 0)
        return MLX.where(valid, shifted, MLXArray(Int64(eosTokenId)))
    }

    func callAsFunction(_ inputIds: MLXArray, cache: ArraysCache?) -> MLXArray {
        let ids = inputIds.asType(.int64)
        let B = ids.dim(0)
        let S = ids.dim(1)
        let previous: MLXArray
        if let cached = cache?[3] {
            previous = cached
        } else {
            previous = MLXArray.full(
                [B, contextLen], values: MLXArray(Int64(eosTokenId)), dtype: .int64)
        }
        let tokenHistory = concatenated([previous, ids], axis: -1)
        if let cache {
            let histLen = tokenHistory.dim(1)
            cache[3] = tokenHistory[0..., max(0, histLen - contextLen)...].asType(.int64)
        }

        var shifted = [MLXArray]()
        for shift in 0 ..< ngramSize {
            shifted.append(shiftRightIgnoreEos(tokenHistory, shift: shift))
        }

        var blocks = [MLXArray]()
        for ngram in 2 ... ngramSize {
            let start = (ngram - 2) * headsPerNgram
            let end = start + headsPerNgram
            var mixed = shifted[0] * layerMultipliers[0]
            for position in 1 ..< ngram {
                mixed = mixed ^ (shifted[position] * layerMultipliers[position])
            }
            let sizes = ngramHeadsVocabSizes[start ..< end]
            let offsets = ngramHeadsOffsets[start ..< end]
            let ngramIds = mixed.expandedDimensions(axis: -1) % sizes.reshaped([1, 1, -1])
            blocks.append(ngramIds + offsets.reshaped([1, 1, -1]))
        }
        let ngramIds = concatenated(blocks, axis: -1)[0..., (-S)...]
        let embeddings = ngramEmbedding(ngramIds)
        return embeddings.reshaped(embeddings.shape.dropLast(2) + [-1])
    }
}

final class Qwen4ExpPLELayer: Module {
    let hiddenSize: Int
    let hcCount: Int
    let convDilation: Int
    let shortConvStateLen: Int

    @ModuleInfo(key: "ple_embedding") var pleEmbedding: Qwen4ExpNGramEmbedding
    @ModuleInfo(key: "key_proj") var keyProj: Linear
    @ModuleInfo(key: "value_proj") var valueProj: Linear
    @ModuleInfo(key: "norm_key") var normKey: Qwen4ExpRMSNorm
    @ModuleInfo(key: "norm_query") var normQuery: Qwen4ExpRMSNorm
    @ModuleInfo(key: "norm_conv") var normConv: Qwen4ExpRMSNorm
    @ModuleInfo(key: "conv1d") var conv1d: Conv1d

    init(_ args: Qwen4ExpTextConfiguration, layerIdx: Int, pleLayerIndex: Int) {
        self.hiddenSize = args.hiddenSize
        self.hcCount = args.hcCount
        let hcHidden = hiddenSize * hcCount
        self.convDilation = args.ngramSize
        self.shortConvStateLen = (args.pleConvKernelSize - 1) * convDilation
        _pleEmbedding.wrappedValue = Qwen4ExpNGramEmbedding(
            args, embeddingDim: args.pleEmbedDim, layerIdx: layerIdx, pleLayerIndex: pleLayerIndex)
        _keyProj.wrappedValue = Linear(args.pleEmbedDim, hcHidden, bias: false)
        _valueProj.wrappedValue = Linear(args.pleEmbedDim, hiddenSize, bias: false)
        _normKey.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: hcHidden, eps: args.rmsNormEps, groupSize: hiddenSize)
        _normQuery.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: hcHidden, eps: args.rmsNormEps, groupSize: hiddenSize)
        _normConv.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: hcHidden, eps: args.rmsNormEps, groupSize: hiddenSize)
        _conv1d.wrappedValue = Conv1d(
            inputChannels: hcHidden,
            outputChannels: hcHidden,
            kernelSize: args.pleConvKernelSize,
            stride: 1,
            padding: 0,
            dilation: convDilation,
            groups: hcHidden,
            bias: false
        )
        super.init()
    }

    private func shortConv(_ x: MLXArray, cache: ArraysCache?) -> MLXArray {
        let B = x.dim(0)
        let state: MLXArray
        if let cached = cache?[2] {
            state = cached
        } else {
            state = MLXArray.zeros([B, shortConvStateLen, x.dim(-1)], dtype: x.dtype)
        }
        let convInput = concatenated([state, x], axis: 1)
        if let cache {
            cache[2] = convInput[0..., (-shortConvStateLen)...]
        }
        return silu(conv1d(convInput))
    }

    func callAsFunction(
        _ hiddenStates: MLXArray, inputIds: MLXArray, cache: ArraysCache?, mask: MLXArray?
    ) -> MLXArray {
        let embeddings = pleEmbedding(inputIds, cache: cache)
        let keys = normKey(keyProj(embeddings)).reshaped(
            hiddenStates.shape.dropLast() + [hcCount, hiddenSize])
        let values = valueProj(embeddings)
        let queries = normQuery(hiddenStates).reshaped(
            hiddenStates.shape.dropLast() + [hcCount, hiddenSize])
        var gate = (keys * queries).sum(axis: -1, keepDims: true) / sqrt(Float(hiddenSize))
        gate = MLX.sign(gate) * sqrt(MLX.maximum(MLX.abs(gate), MLXArray(Float(1e-6))))
        var gatedValues = sigmoid(gate) * values.expandedDimensions(axis: -2)
        gatedValues = gatedValues.reshaped(hiddenStates.shape)
        var normed = normConv(gatedValues)
        if let mask, mask.ndim == 2 {
            gatedValues = MLX.where(mask[.ellipsis, .newAxis], gatedValues, 0)
            normed = MLX.where(mask[.ellipsis, .newAxis], normed, 0)
        }
        return gatedValues + shortConv(normed, cache: cache)
    }
}

// MARK: - Decoder / Model

public final class Qwen4ExpDecoderLayer: Module {
    public let isLinear: Bool
    public let hasPLE: Bool

    @ModuleInfo(key: "linear_attn") var linearAttn: Qwen4ExpGatedDeltaNet?
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen4ExpAttention?
    @ModuleInfo(key: "mlp") var mlp: Qwen4ExpSparseMoeBlock
    @ModuleInfo(key: "ple") var ple: Qwen4ExpPLELayer?
    @ModuleInfo(key: "attn_hyper_connection") var attnHyper: Qwen4ExpGatedResidual
    @ModuleInfo(key: "mlp_hyper_connection") var mlpHyper: Qwen4ExpGatedResidual

    init(_ args: Qwen4ExpTextConfiguration, layerIdx: Int) {
        self.isLinear = args.isLinear(layerIdx: layerIdx)
        if isLinear {
            _linearAttn.wrappedValue = Qwen4ExpGatedDeltaNet(args)
        } else {
            _selfAttn.wrappedValue = Qwen4ExpAttention(args)
        }
        _mlp.wrappedValue = Qwen4ExpSparseMoeBlock(args)
        let pleIndex = args.pleLayerIds.firstIndex(of: layerIdx + 1)
        self.hasPLE = pleIndex != nil
        if let pleIndex {
            _ple.wrappedValue = Qwen4ExpPLELayer(args, layerIdx: layerIdx, pleLayerIndex: pleIndex)
        }
        _attnHyper.wrappedValue = Qwen4ExpGatedResidual(args)
        _mlpHyper.wrappedValue = Qwen4ExpGatedResidual(args)
        super.init()
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        inputIds: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?,
        positionIds: MLXArray?
    ) -> MLXArray {
        var h = hiddenStates
        if let ple {
            h = h + ple(h, inputIds: inputIds, cache: cache as? ArraysCache, mask: ssmMask)
        }
        let (mixedAttn, hyperAttn, injectAttn) = attnHyper(h)
        let branchAttn: MLXArray
        if isLinear {
            branchAttn = linearAttn!(mixedAttn, mask: ssmMask, cache: cache as? ArraysCache)
        } else {
            branchAttn = selfAttn!(
                mixedAttn, mask: attentionMask, cache: cache, positionIds: positionIds)
        }
        h = hyperAttn + (branchAttn.expandedDimensions(axis: -2) * injectAttn.expandedDimensions(axis: -1))
            .reshaped(hyperAttn.shape)

        let (mixedMlp, hyperMlp, injectMlp) = mlpHyper(h)
        let branchMlp = mlp(mixedMlp)
        return hyperMlp + (branchMlp.expandedDimensions(axis: -2) * injectMlp.expandedDimensions(axis: -1))
            .reshaped(hyperMlp.shape)
    }
}

public final class Qwen4ExpTextModelInner: Module {
    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    public let layers: [Qwen4ExpDecoderLayer]
    @ModuleInfo(key: "hyper_connection_mixer") var mixer: Qwen4ExpGatedResidual
    let args: Qwen4ExpTextConfiguration
    public let ssmIdx: Int
    public let faIdx: Int

    init(_ args: Qwen4ExpTextConfiguration) {
        self.args = args
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { Qwen4ExpDecoderLayer(args, layerIdx: $0) }
        _mixer.wrappedValue = Qwen4ExpGatedResidual(args, useCombine: false)
        self.ssmIdx = layers.firstIndex(where: \.isLinear) ?? 0
        self.faIdx = layers.firstIndex(where: { !$0.isLinear }) ?? 0
        super.init()
    }

    public func callAsFunction(
        _ inputs: MLXArray,
        inputsEmbeds: MLXArray? = nil,
        cache: [KVCache?]? = nil,
        positionIds: MLXArray? = nil
    ) -> MLXArray {
        mixer.mixOnly(forwardPreMixer(inputs, inputsEmbeds: inputsEmbeds, cache: cache, positionIds: positionIds))
    }

    /// Un-mixed hyper-connection streams `[B, S, hc_count * hidden]`. Lightning MTP fuses these.
    public func forwardPreMixer(
        _ inputs: MLXArray,
        inputsEmbeds: MLXArray? = nil,
        cache: [KVCache?]? = nil,
        positionIds: MLXArray? = nil
    ) -> MLXArray {
        var tokens = inputs
        if tokens.ndim == 1 {
            tokens = tokens.expandedDimensions(axis: 0)
        }
        var hidden = inputsEmbeds ?? embedTokens(tokens)
        hidden = tiled(hidden, repetitions: [1, 1, args.hcCount])

        var cacheArray = cache
        if cacheArray == nil {
            cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
        }
        let faMask = createAttentionMask(h: hidden, cache: cacheArray?[faIdx])
        let ssmMask = (cacheArray?[ssmIdx] as? ArraysCache)?.makeMask(N: hidden.dim(1))

        for (i, layer) in layers.enumerated() {
            let attnMask: MLXFast.ScaledDotProductAttentionMaskMode =
                layer.isLinear ? .none : faMask
            hidden = layer(
                hidden,
                inputIds: tokens,
                attentionMask: attnMask,
                ssmMask: layer.isLinear ? ssmMask : nil,
                cache: cacheArray?[i],
                positionIds: positionIds
            )
        }
        return hidden
    }
}

/// Native Flash-Next Lightning MTP head (mlx-vlm `qwen4_exp_mtp`).
/// Fuses token embeddings with un-mixed hyper-connection streams `[B, S, hc*H]`.
final class Qwen4ExpMTPPredictor: Module {
    let args: Qwen4ExpTextConfiguration

    @ModuleInfo(key: "pre_fc_norm_embedding") var preFcNormEmbedding: Qwen4ExpRMSNorm
    @ModuleInfo(key: "pre_fc_norm_hidden") var preFcNormHidden: Qwen4ExpRMSNorm
    @ModuleInfo(key: "fc_embedding") var fcEmbedding: Linear
    @ModuleInfo(key: "fc_hidden") var fcHidden: Linear
    @ModuleInfo(key: "layers") var layers: [Qwen4ExpDecoderLayer]
    @ModuleInfo(key: "hyper_connection_mixer") var mixer: Qwen4ExpGatedResidual

    init(_ args: Qwen4ExpTextConfiguration) {
        self.args = args
        let hidden = args.hiddenSize
        let hcHidden = args.hcCount * hidden
        _preFcNormEmbedding.wrappedValue = Qwen4ExpRMSNorm(dimensions: hidden, eps: args.rmsNormEps)
        _preFcNormHidden.wrappedValue = Qwen4ExpRMSNorm(dimensions: hcHidden, eps: args.rmsNormEps)
        _fcEmbedding.wrappedValue = Linear(hidden, hidden, bias: false)
        _fcHidden.wrappedValue = Linear(hidden, hidden, bias: false)
        var layerArgs = args
        let depth = max(args.mtpNumHiddenLayers, 1)
        layerArgs.hiddenLayers = depth
        layerArgs.layerTypes = Array(repeating: "qwen_sparse_attention", count: depth)
        layerArgs.fullAttentionInterval = 1
        layerArgs.pleLayerIds = []
        _layers.wrappedValue = (0 ..< depth).map { Qwen4ExpDecoderLayer(layerArgs, layerIdx: $0) }
        _mixer.wrappedValue = Qwen4ExpGatedResidual(layerArgs, useCombine: false)
        super.init()
    }

    func fuse(tokenEmbed: MLXArray, hidden: MLXArray) -> MLXArray {
        let hc = args.hcCount
        let d = args.hiddenSize
        var h = hidden
        if h.ndim == 2 {
            h = h.reshaped(1, h.dim(0), h.dim(1))
        }
        let expected = hc * d
        if h.dim(-1) == d {
            h = tiled(h, repetitions: [1, 1, hc])
        }
        precondition(
            h.ndim == 3 && h.dim(-1) == expected,
            "Qwen4 Lightning MTP expects hidden shape [batch, tokens, hc_count * hidden_size]"
        )
        let projectedEmbedding = fcEmbedding(preFcNormEmbedding(tokenEmbed))
        var streams = preFcNormHidden(h)
        streams = streams.reshaped(streams.shape.dropLast() + [hc, d])
        let projectedHidden = fcHidden(streams)
        return (projectedEmbedding.expandedDimensions(axis: -2) + projectedHidden).reshaped(h.shape)
    }

    func callAsFunction(
        tokenEmbed: MLXArray,
        hidden: MLXArray,
        tokens: MLXArray,
        cache: [KVCache]?
    ) -> MLXArray {
        var h = fuse(tokenEmbed: tokenEmbed, hidden: hidden)
        let caches: [KVCache?] = cache.map { $0.map { Optional($0) } } ?? Array(repeating: nil, count: layers.count)
        let faMask = createAttentionMask(h: h, cache: caches.first ?? nil)
        for (i, layer) in layers.enumerated() {
            h = layer(
                h,
                inputIds: tokens,
                attentionMask: faMask,
                ssmMask: nil,
                cache: i < caches.count ? caches[i] : nil,
                positionIds: nil
            )
        }
        return h
    }
}

public class Qwen4ExpTextModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let model: Qwen4ExpTextModelInner
    public let configuration: Qwen4ExpTextConfiguration
    public var nativeMtpAvailable = false

    @ModuleInfo(key: "lm_head") public var lmHead: Linear?
    @ModuleInfo(key: "mtp") var mtp: Qwen4ExpMTPPredictor?

    public init(_ args: Qwen4ExpTextConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Qwen4ExpTextModelInner(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func attachNativeMtp(from weights: [String: MLXArray]) {
        let hasMtp = weights.keys.contains { key in
            key.contains(".mtp.") || key.hasPrefix("mtp.")
        }
        guard hasMtp else { return }
        if mtp == nil {
            _mtp.wrappedValue = Qwen4ExpMTPPredictor(configuration)
        }
        nativeMtpAvailable = true
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let typed: [KVCache?]? = cache.map { $0.map { Optional($0) } }
        var out = model(inputs, inputsEmbeds: nil, cache: typed, positionIds: nil)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        model.layers.map { layer in
            if layer.isLinear {
                return ArraysCache(size: 4)
            }
            if let maxKV = parameters?.maxKVSize {
                return RotatingKVCache(maxSize: maxKV, keep: 4)
            }
            return QSAKVCache()
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = Qwen4ExpModel.remapWeights(weights, layerCount: configuration.hiddenLayers)
        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }
        weights = Qwen4ExpModel.canonicalizeRMSNorm(weights)
        attachNativeMtp(from: weights)
        return weights
    }

    private static func mtpTokenBatch(_ tokens: MLXArray) -> MLXArray {
        tokens.ndim == 1 ? tokens.reshaped(1, -1) : tokens
    }
}

extension Qwen4ExpTextModel: MtpTarget, MtpDrafter {
    public var mtpBlockSize: Int {
        guard nativeMtpAvailable, mtp != nil else { return 0 }
        return max(configuration.mtpNumHiddenLayers, 1) + 1
    }

    public func mtpEmbed(_ tokens: MLXArray) -> MLXArray {
        model.embedTokens(Self.mtpTokenBatch(tokens))
    }

    public func mtpLmHead(_ hidden: MLXArray) -> MLXArray {
        var h = hidden
        if let mtp, h.dim(-1) == configuration.hiddenSize * configuration.hcCount {
            h = mtp.mixer.mixOnly(h)
        }
        if let lmHead {
            return lmHead(h)
        }
        return model.embedTokens.asLinear(h)
    }

    public func mtpHiddenAndLogits(_ tokens: MLXArray, cache: [KVCache]?) -> (
        hidden: MLXArray, logits: MLXArray
    ) {
        let ids = Self.mtpTokenBatch(tokens)
        let typed: [KVCache?]? = cache.map { $0.map { Optional($0) } }
        let preMixer = model.forwardPreMixer(ids, inputsEmbeds: nil, cache: typed, positionIds: nil)
        let mixed = model.mixer.mixOnly(preMixer)
        let logits: MLXArray
        if let lmHead {
            logits = lmHead(mixed)
        } else {
            logits = model.embedTokens.asLinear(mixed)
        }
        return (preMixer, logits)
    }

    public func bindMtp(
        embed: @escaping (MLXArray) -> MLXArray, lmHead: @escaping (MLXArray) -> MLXArray
    ) {}

    public func mtpForward(tokenEmbed: MLXArray, hidden: MLXArray, cache: [KVCache]?) -> MLXArray {
        guard let mtp else {
            return hidden
        }
        var embed = tokenEmbed
        if embed.ndim == 1 {
            embed = embed.reshaped(1, -1)
        }
        if embed.ndim == 2 && embed.dim(-1) != configuration.hiddenSize {
            embed = model.embedTokens(embed)
        }
        let dummyIds = MLXArray.zeros([embed.dim(0), embed.dim(1)], dtype: .int32)
        return mtp(tokenEmbed: embed, hidden: hidden, tokens: dummyIds, cache: cache)
    }

    public func mtpNewCache(parameters: GenerateParameters?) -> [KVCache] {
        (mtp?.layers ?? []).map { _ in QSAKVCache() }
    }
}

extension Qwen4ExpTextModel: LoRAModel {
    public var loraLayers: [Module] { model.layers }
}

public class Qwen4ExpModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "language_model") var languageModel: Qwen4ExpTextModel

    public init(_ args: Qwen4ExpConfiguration) {
        let text = Qwen4ExpTextModel(args.textConfig)
        self.vocabularySize = text.vocabularySize
        self.kvHeads = text.kvHeads
        _languageModel.wrappedValue = text
    }

    public init(_ args: Qwen4ExpTextConfiguration) {
        let text = Qwen4ExpTextModel(args)
        self.vocabularySize = text.vocabularySize
        self.kvHeads = text.kvHeads
        _languageModel.wrappedValue = text
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try languageModel.newCache(parameters: parameters)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let mapped = Self.remapWeights(weights, layerCount: languageModel.configuration.hiddenLayers)
        var nested = [String: MLXArray]()
        for (key, value) in mapped {
            if key.hasPrefix("language_model.") {
                nested[key] = value
            } else if key.hasPrefix("lm_head") {
                nested["language_model." + key] = value
            } else {
                nested["language_model." + key] = value
            }
        }
        if languageModel.configuration.tieWordEmbeddings {
            nested["language_model.lm_head.weight"] = nil
        }
        nested = Self.canonicalizeRMSNorm(nested)
        languageModel.attachNativeMtp(from: nested)
        return nested
    }

    public static func remapWeights(_ weights: [String: MLXArray], layerCount: Int) -> [String: MLXArray] {
        var out = [String: MLXArray]()
        let shard = try! NSRegularExpression(pattern: #"\.ngram_embedding\.shard_(\d+)(?=\.)"#)
        for (rawKey, rawValue) in weights {
            if rawKey.contains("vision_tower") || rawKey.contains("model.visual")
                || rawKey.contains("patch_embed")
            {
                continue
            }
            if rawKey.contains("ngram_embedding")
                && (rawKey.hasSuffix("weight_scale") || rawKey.hasSuffix("weightScale"))
            {
                continue
            }
            var key = rawKey
            if key.hasPrefix("model.language_model") {
                key = key.replacingOccurrences(of: "model.language_model", with: "language_model.model")
            }
            // pipenetwork mixed quant stores a 1-vector scale; ModuleInfo loads it.
            let nsKey = key as NSString
            let range = NSRange(location: 0, length: nsKey.length)
            key = shard.stringByReplacingMatches(
                in: key, options: [], range: range, withTemplate: ".ngram_embedding.shards.$1")
            var value = rawValue
            if key.contains("conv1d.weight") && value.dim(-1) != 1 {
                value = value.movedAxis(source: 2, destination: 1)
            }
            out[key] = value
        }

        for layer in 0 ..< layerCount {
            let prefixes = [
                "language_model.model.layers.\(layer).mlp",
                "model.language_model.layers.\(layer).mlp",
                "layers.\(layer).mlp",
            ]
            for prefix in prefixes {
                stackExperts(prefix: prefix, weights: &out)
            }
        }
        var mtpLayer = 0
        while true {
            let prefixes = [
                "language_model.mtp.layers.\(mtpLayer).mlp",
                "mtp.layers.\(mtpLayer).mlp",
            ]
            let found = prefixes.contains { prefix in
                out["\(prefix).experts.gate_up_proj"] != nil
                    || out["\(prefix).experts.gate_up_proj.weight"] != nil
                    || out["\(prefix).experts.0.gate_proj.weight"] != nil
            }
            guard found else { break }
            for prefix in prefixes {
                stackExperts(prefix: prefix, weights: &out)
            }
            mtpLayer += 1
        }
        return out
    }

    static func stackExperts(prefix: String, weights: inout [String: MLXArray]) {
        if weights["\(prefix).switch_mlp.gate_proj.weight"] != nil { return }
        if let gateUp = weights["\(prefix).experts.gate_up_proj"]
            ?? weights["\(prefix).experts.gate_up_proj.weight"]
        {
            weights["\(prefix).experts.gate_up_proj"] = nil
            weights["\(prefix).experts.gate_up_proj.weight"] = nil
            let mid = gateUp.dim(-2) / 2
            weights["\(prefix).switch_mlp.gate_proj.weight"] = gateUp[.ellipsis, ..<mid, 0...]
            weights["\(prefix).switch_mlp.up_proj.weight"] = gateUp[.ellipsis, mid..., 0...]
            if let down = weights["\(prefix).experts.down_proj"]
                ?? weights["\(prefix).experts.down_proj.weight"]
            {
                weights["\(prefix).experts.down_proj"] = nil
                weights["\(prefix).experts.down_proj.weight"] = nil
                weights["\(prefix).switch_mlp.down_proj.weight"] = down
            }
            return
        }
        guard weights["\(prefix).experts.0.gate_proj.weight"] != nil else { return }
        var expertCount = 0
        while weights["\(prefix).experts.\(expertCount).gate_proj.weight"] != nil {
            expertCount += 1
        }
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            for suffix in ["weight", "scales", "biases"] {
                let first = "\(prefix).experts.0.\(projection).\(suffix)"
                guard weights[first] != nil else { continue }
                var stacked: [MLXArray] = []
                for expert in 0 ..< expertCount {
                    let k = "\(prefix).experts.\(expert).\(projection).\(suffix)"
                    if let value = weights[k] {
                        stacked.append(value)
                        weights[k] = nil
                    }
                }
                if !stacked.isEmpty {
                    weights["\(prefix).switch_mlp.\(projection).\(suffix)"] = MLX.stacked(stacked)
                }
            }
        }
    }

    public static func canonicalizeRMSNorm(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        let anchors = weights.compactMap { key, value -> MLXArray? in
            guard key.contains(".attn_hyper_connection.hc_norm.weight") else { return nil }
            return value
        }
        guard anchors.count >= 4 else { return weights }
        let means = stacked(anchors.map { $0.asType(.float32).mean() })
        eval(means)
        let meanValues = means.asArray(Float.self)
        guard !meanValues.isEmpty else { return weights }
        let sorted = meanValues.sorted()
        let median = sorted[sorted.count / 2]
        let onesVote = Float(meanValues.filter { $0 > 0.5 }.count) / Float(meanValues.count)
        let onesCentered = onesVote >= 0.9 && median >= 0.75 && median <= 1.5
        guard onesCentered else { return weights }

        var out = weights
        for (key, value) in weights where key.hasSuffix(".weight") && value.ndim == 1 {
            if key.contains("hc_norm") || key.contains("q_norm") || key.contains("k_norm")
                || key.contains("q_layernorm") || key.contains("k_layernorm")
                || key.contains("norm_key") || key.contains("norm_query")
                || key.contains("norm_conv") || key.contains("pre_fc_norm")
            {
                out[key] = value.asType(.float32) - 1
            }
        }
        return out
    }
}

extension Qwen4ExpModel: LoRAModel {
    public var loraLayers: [Module] { languageModel.model.layers }
}

extension Qwen4ExpModel: MtpTarget, MtpDrafter {
    public var mtpBlockSize: Int { languageModel.mtpBlockSize }

    public func mtpEmbed(_ tokens: MLXArray) -> MLXArray {
        languageModel.mtpEmbed(tokens)
    }

    public func mtpLmHead(_ hidden: MLXArray) -> MLXArray {
        languageModel.mtpLmHead(hidden)
    }

    public func mtpHiddenAndLogits(_ tokens: MLXArray, cache: [KVCache]?) -> (
        hidden: MLXArray, logits: MLXArray
    ) {
        languageModel.mtpHiddenAndLogits(tokens, cache: cache)
    }

    public func bindMtp(
        embed: @escaping (MLXArray) -> MLXArray, lmHead: @escaping (MLXArray) -> MLXArray
    ) {
        languageModel.bindMtp(embed: embed, lmHead: lmHead)
    }

    public func mtpForward(tokenEmbed: MLXArray, hidden: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel.mtpForward(tokenEmbed: tokenEmbed, hidden: hidden, cache: cache)
    }

    public func mtpNewCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.mtpNewCache(parameters: parameters)
    }
}
