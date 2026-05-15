// Distributed layer invocation protocols for pipeline-parallel sharding.
// Defined in MLXLMCommon so both vendor model code (MLXLLM) and
// distributed orchestration (NovaMLXDistributed) can reference them.

import MLX

// MARK: - 4-Param Layer Protocol

/// Protocol for transformer layers that accept (x, attentionMask, ssmMask, cache).
/// Used by Qwen35 hybrid (attention + GatedDeltaNet) and similar architectures.
public protocol DistributedLayerProtocol4 {
    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray
}

// MARK: - 2-Param Layer Protocol

/// Protocol for simpler transformer layers that accept (x, cache).
/// Used by older/simpler architectures.
public protocol TransformerBlockProtocol {
    func callAsFunction(_ x: MLXArray, cache: KVCache?) -> MLXArray
}

// MARK: - 3-Param Layer Protocol

/// Protocol for transformer layers that accept (x, mask, cache).
/// Used by standard attention-only architectures (Qwen3, Qwen3MoE, etc).
public protocol DistributedLayerProtocol3 {
    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray
}

// MARK: - Unary Layer Protocol

/// Protocol for modules callable with a single input (Embedding, Linear, RMSNorm).
public protocol UnaryLayerProtocol {
    func callAsFunction(_ x: MLXArray) -> MLXArray
}

// MARK: - Embedding with Linear Projection

/// Protocol for embedding modules that support tied-weight linear projection.
public protocol EmbeddingAsLinearProtocol {
    func asLinear(_ x: MLXArray) -> MLXArray
}
