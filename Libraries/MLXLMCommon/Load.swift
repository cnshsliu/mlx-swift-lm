// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Load model weights.
///
/// This is typically called via ``GenericModelFactory/load(from:using:configuration:useLatest:progressHandler:)``.
/// This function loads all `safetensor` files in the given `modelDirectory`,
/// calls ``BaseLanguageModel/sanitize(weights:metadata:)`` to allow per-model preprocessing,
/// applies optional quantization, and
/// updates the model with the weights.
public func loadWeights(
    modelDirectory: URL, model: BaseLanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
) throws {
    // load the weights and collect metadata from the first safetensor file
    var weights = [String: MLXArray]()
    var metadata = [String: String]()
    let enumerator = FileManager.default.enumerator(
        at: modelDirectory, includingPropertiesForKeys: nil)!
    var safetensorCount = 0
    for case let url as URL in enumerator {
        if url.pathExtension == "safetensors" {
            safetensorCount += 1
            let (w, m) = try loadArraysAndMetadata(url: url)
            for (key, value) in w {
                weights[key] = value
            }
            if metadata.isEmpty {
                metadata = m
            }
        }
    }

    // per-model cleanup (models can inspect metadata to customize behavior)
    weights = model.sanitize(weights: weights, metadata: metadata)

    // quantize if needed
    if quantization != nil || perLayerQuantization != nil {
        // Pre-quantized weights are already packed (dtype=uint32) with scales/biases.
        // The default `quantize()` path calls `MLX.quantized()` on the layer's current
        // (randomly-initialized) weight, then `model.update()` overwrites the result with
        // the pre-quantized arrays.  For some model/dtype combinations this re-quantization
        // step crashes inside the Metal backend (SIGTRAP in mlx_quantize).  We avoid that
        // by building QuantizedLinear directly from the pre-quantized arrays.
        let preQuantizedPaths = Set(
            weights.compactMap { key, value -> String? in
                guard key.hasSuffix(".scales") else { return nil }
                let path = String(key.dropLast(".scales".count))
                guard let w = weights["\(path).weight"], w.dtype == .uint32 else { return nil }
                return path
            }
        )

        if !preQuantizedPaths.isEmpty {
            let leafModules = model.leafModules().flattened()
            var updates: [(String, Module)] = []
            for (path, module) in leafModules {
                guard preQuantizedPaths.contains(path) else { continue }

                let qConfig = perLayerQuantization?.quantization(layer: path) ?? quantization
                guard let (groupSize, bits, mode) = qConfig?.asTuple else { continue }
                guard let w = weights["\(path).weight"],
                      let scales = weights["\(path).scales"]
                else { continue }
                let biases = weights["\(path).biases"]

                if let linear = module as? Linear {
                    let quantized = QuantizedLinear(
                        weight: w,
                        bias: linear.bias,
                        scales: scales,
                        biases: biases,
                        groupSize: groupSize,
                        bits: bits,
                        mode: mode
                    )
                    updates.append((path, quantized))
                } else if module is Embedding {
                    let quantized = QuantizedEmbedding(
                        weight: w,
                        scales: scales,
                        biases: biases,
                        groupSize: groupSize,
                        bits: bits,
                        mode: mode
                    )
                    updates.append((path, quantized))
                } else if let switchLinear = module as? SwitchLinear {
                    let quantized = QuantizedSwitchLinear(
                        weight: w,
                        bias: switchLinear.bias,
                        scales: scales,
                        biases: biases,
                        inputDims: switchLinear.inputDims,
                        outputDims: switchLinear.outputDims,
                        numExperts: switchLinear.numExperts,
                        groupSize: groupSize,
                        bits: bits,
                        mode: mode
                    )
                    updates.append((path, quantized))
                }
            }
            if !updates.isEmpty {
                let unflat = ModuleChildren.unflattened(updates)

                do {
                    try model.update(modules: unflat, verify: .none)
                } catch {
                    // Bulk update can fail with mismatchedContainers for models using
                    // plain var arrays (e.g. var layers: [Block]). Fallback: navigate
                    // the module tree and replace each leaf via updateModule.
                    var failures = 0
                    for (path, newModule) in updates {
                        let pathParts = path.split(separator: ".").map { String($0) }

                        func findAndReplace(parent: Module, parts: [String], replacement: Module) -> Bool {
                            guard let first = parts.first else { return false }
                            let rest = Array(parts.dropFirst())

                            if rest.isEmpty {
                                do {
                                    try parent.updateModule(key: first, replacement)
                                    return true
                                } catch {
                                    return false
                                }
                            }

                            let childItems = parent.items()
                            guard let childItem = childItems[first] else { return false }

                            switch childItem {
                            case .value(.module(let childModule)):
                                return findAndReplace(parent: childModule, parts: rest, replacement: replacement)
                            case .array(let arr):
                                guard let idx = Int(rest.first ?? "") else { return false }
                                let deeperRest = Array(rest.dropFirst())
                                guard idx < arr.count else { return false }
                                switch arr[idx] {
                                case .value(.module(let childModule)):
                                    return findAndReplace(parent: childModule, parts: deeperRest, replacement: replacement)
                                default:
                                    return false
                                }
                            default:
                                return false
                            }
                        }

                        if !findAndReplace(parent: model, parts: pathParts, replacement: newModule) {
                            failures += 1
                        }
                    }

                    if failures > 0 {
                        print("[Load] Pre-quantized module replacement: \(failures)/\(updates.count) failures")
                    }

                    // updateModule changes the @ModuleInfo wrapper's value but does NOT
                    // rebuild the _items cache. Without rebuilding, update(parameters:)
                    // would traverse stale caches pointing to the old (non-quantized) modules
                    // and crash on shape mismatches. Rebuild caches on every module in the tree.
                    model.visit(modules: { _, m in
                        _ = try? m.update(modules: ModuleChildren(), verify: .none)
                    })
                }
            }
        }

        // Quantize remaining (non-pre-quantized) layers.
        // quantize() uses model.update(modules:) with try!, which crashes on
        // mismatchedContainers for nested models. Only call if there are remaining
        // unquantized modules, and wrap safely.
        let remainingToQuantize = model.leafModules().flattened().filter { path, _ in
            !preQuantizedPaths.contains(path) && weights["\(path).scales"] != nil
        }
        if !remainingToQuantize.isEmpty {
            do {
                // Build quantized replacements for remaining modules
                var quantUpdates = [(String, Module)]()
                for (path, module) in remainingToQuantize {
                    let qConfig = perLayerQuantization?.quantization(layer: path) ?? quantization
                    guard let (groupSize, bits, mode) = qConfig?.asTuple else { continue }
                    if let quantized = quantizeSingle(layer: module, groupSize: groupSize, bits: bits, mode: mode) {
                        quantUpdates.append((path, quantized))
                    }
                }
                if !quantUpdates.isEmpty {
                    let qUnflat = ModuleChildren.unflattened(quantUpdates)
                    try model.update(modules: qUnflat, verify: .none)
                }
            } catch {
                // quantize fallback: apply individually
                for (path, module) in remainingToQuantize {
                    let qConfig = perLayerQuantization?.quantization(layer: path) ?? quantization
                    guard let (groupSize, bits, mode) = qConfig?.asTuple else { continue }
                    if let quantized = quantizeSingle(layer: module, groupSize: groupSize, bits: bits, mode: mode) {
                        let pathParts = path.split(separator: ".").map { String($0) }
                        // Navigate to parent and replace
                        func findAndReplace(parent: Module, parts: [String], replacement: Module) -> Bool {
                            guard let first = parts.first else { return false }
                            let rest = Array(parts.dropFirst())
                            if rest.isEmpty {
                                return (try? parent.updateModule(key: first, replacement)) != nil
                            }
                            let childItems = parent.items()
                            guard let childItem = childItems[first] else { return false }
                            switch childItem {
                            case .value(.module(let childModule)):
                                return findAndReplace(parent: childModule, parts: rest, replacement: replacement)
                            case .array(let arr):
                                guard let idx = Int(rest.first ?? "") else { return false }
                                let deeperRest = Array(rest.dropFirst())
                                guard idx < arr.count else { return false }
                                switch arr[idx] {
                                case .value(.module(let childModule)):
                                    return findAndReplace(parent: childModule, parts: deeperRest, replacement: replacement)
                                default: return false
                                }
                            default: return false
                            }
                        }
                        _ = findAndReplace(parent: model, parts: pathParts, replacement: quantized)
                    }
                }
                model.visit(modules: { _, m in
                    _ = try? m.update(modules: ModuleChildren(), verify: .none)
                })
            }
        }
    }

    // apply the loaded weights
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [])

    // Staged eval: materialize arrays in per-layer batches to keep peak GPU
    // memory bounded.  For models with many weight arrays (>500), evaluating
    // everything at once can exceed Metal's recommended working set, causing
    // a command buffer failure (SIGABRT via mlx check_error).
    // Note: Memory.activeMemory may be 0 for memory-mapped safetensor data,
    // so we use weight count as a proxy for model size.
    if weights.count > 500 {
        stagedEval(model: model)
    } else {
        eval(model)
    }
}

/// Evaluate a model's arrays in small batches to keep peak GPU memory bounded.
/// For large models, evaluating everything at once can exceed Metal's working set.
/// Batching avoids the peak where source arrays and Metal buffers coexist.
private func stagedEval(model: BaseLanguageModel) {
    let allArrays = model.innerState()
    let total = allArrays.count
    // Use small batches (8 arrays) to limit peak Metal memory during materialization.
    // Each batch evals a few arrays, then clearCache releases intermediate Metal buffers.
    let batchSize = 8

    for i in stride(from: 0, to: total, by: batchSize) {
        let end = min(i + batchSize, total)
        let batch = Array(allArrays[i..<end])
        eval(batch)
        if i % (batchSize * 20) == 0 && i > 0 {
            Memory.clearCache()
        }
    }
    Memory.clearCache()
}
