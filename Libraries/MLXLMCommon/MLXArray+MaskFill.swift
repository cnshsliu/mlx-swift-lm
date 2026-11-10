import MLX

/// Compatibility shims for mlx-swift APIs added in 0.31.6 that the vendored
/// mlx-swift tree does not yet expose.
extension MLXArray {
    public static func maskFill(for dtype: DType) -> MLXArray {
        MLXArray(-Float.greatestFiniteMagnitude, dtype: dtype)
    }
}

extension DType {
    var greatestFiniteMagnitudeArray: MLXArray {
        MLXArray(Float.greatestFiniteMagnitude, dtype: self)
    }
}
