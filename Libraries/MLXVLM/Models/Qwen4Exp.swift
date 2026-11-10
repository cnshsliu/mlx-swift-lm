//
//  Qwen4Exp.swift
//  mlx-swift-lm
//
//  VLM wrapper for Qwen3.8-Flash-Next (qwen4_exp). Language path lives in
//  MLXLLM.Qwen4ExpModel; vision reuses the published Qwen3-VL ViT.
//

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

private enum Qwen4ExpVLError: Error {
    case featureTokenMismatch(expected: Int, actual: Int)
}

public struct Qwen4ExpVLMConfiguration: Codable, Sendable {
    public var modelType: String = "qwen4_exp"
    public var textConfig: Qwen4ExpTextConfiguration
    public var visionConfig: Qwen3VLConfiguration.VisionConfiguration
    public var imageTokenId: Int = 248_056
    public var videoTokenId: Int = 248_057
    public var visionStartTokenId: Int = 248_053
    public var visionEndTokenId: Int = 248_054
    public var vocabSize: Int = 248_320

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
        case visionStartTokenId = "vision_start_token_id"
        case visionEndTokenId = "vision_end_token_id"
        case vocabSize = "vocab_size"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen4_exp"
        textConfig = try c.decode(Qwen4ExpTextConfiguration.self, forKey: .textConfig)
        visionConfig = try c.decode(Qwen3VLConfiguration.VisionConfiguration.self, forKey: .visionConfig)
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 248_056
        videoTokenId = try c.decodeIfPresent(Int.self, forKey: .videoTokenId) ?? 248_057
        visionStartTokenId = try c.decodeIfPresent(Int.self, forKey: .visionStartTokenId) ?? 248_053
        visionEndTokenId = try c.decodeIfPresent(Int.self, forKey: .visionEndTokenId) ?? 248_054
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? textConfig.vocabularySize
    }
}

public class Qwen4Exp: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") var visionModel: Qwen3VLVision.VisionModel
    @ModuleInfo(key: "language_model") var languageModel: Qwen4ExpTextModel

    public let config: Qwen4ExpVLMConfiguration
    public var vocabularySize: Int { config.vocabSize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public init(_ config: Qwen4ExpVLMConfiguration) {
        self.config = config
        _visionModel.wrappedValue = Qwen3VLVision.VisionModel(config.visionConfig)
        _languageModel.wrappedValue = Qwen4ExpTextModel(config.textConfig)
        super.init()
    }

    public var loraLayers: [Module] { languageModel.model.layers }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try languageModel.newCache(parameters: parameters)
    }

    public func prepare(
        _ input: LMInput,
        cache: [any KVCache],
        state: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        var inputIds = input.text.tokens
        if inputIds.ndim == 1 {
            inputIds = inputIds.expandedDimensions(axis: 0)
        }

        var pixelValues: MLXArray?
        var frames: [THW] = []
        let visionDType = visionModel.patchEmbed.proj.weight.dtype
        var pixelParts: [MLXArray] = []
        if let image = input.image {
            pixelParts.append(image.pixels.asType(visionDType))
            if let imageFrames = image.frames { frames.append(contentsOf: imageFrames) }
        }
        if let video = input.video {
            pixelParts.append(video.pixels.asType(visionDType))
            if let videoFrames = video.frames { frames.append(contentsOf: videoFrames) }
        }
        if !pixelParts.isEmpty {
            pixelValues = concatenated(pixelParts)
        }

        var inputEmbeddings: MLXArray?
        if let pixelValues, !frames.isEmpty {
            let textEmbeds = languageModel.model.embedTokens(inputIds)
            let (visionHidden, _) = visionModel(pixelValues, gridTHW: frames)
            let (merged, _) = try mergeInputIdsWithImageFeatures(
                imageFeatures: visionHidden.asType(textEmbeds.dtype),
                inputEmbeds: textEmbeds,
                inputIds: inputIds
            )
            inputEmbeddings = merged
        }

        let typed: [KVCache?] = cache.map { Optional($0) }
        var hidden = languageModel.model(
            inputIds, inputsEmbeds: inputEmbeddings, cache: typed, positionIds: nil)
        if let lmHead = languageModel.lmHead {
            hidden = lmHead(hidden)
        } else {
            hidden = languageModel.model.embedTokens.asLinear(hidden)
        }
        let total = inputIds.dim(-1)
        prefill.progress?(total, total)
        return .logits(LMOutput(logits: hidden))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let mapped = Qwen4ExpModel.remapWeights(
            weights, layerCount: languageModel.configuration.hiddenLayers)
        var nested = [String: MLXArray]()
        for (key, value) in mapped {
            var key = key
            if key.contains("model.visual") {
                key = key.replacingOccurrences(of: "model.visual", with: "vision_tower")
            }
            if key.hasPrefix("lm_head") {
                key = "language_model." + key
            } else if !key.hasPrefix("language_model.") && !key.hasPrefix("vision_tower") {
                key = "language_model." + key
            }
            nested[key] = value
        }
        nested = Qwen4ExpModel.canonicalizeRMSNorm(nested)
        // Text load: pipenetwork mixed-4_8bit vision keys are Qwen3-VL Conv3d
        // (`patch_embed.proj.weight`) and do not map onto this PatchEmbed yet.
        nested = nested.filter {
            if $0.key.contains("vision_tower") || $0.key.contains("patch_embed") { return false }
            if $0.key.contains("ngram_embedding")
                && ($0.key.contains("weight_scale") || $0.key.contains("weightScale"))
            {
                return false
            }
            return true
        }
        languageModel.attachNativeMtp(from: nested)
        return nested
    }

    private func mergeInputIdsWithImageFeatures(
        imageFeatures: MLXArray,
        inputEmbeds: MLXArray,
        inputIds: MLXArray
    ) throws -> (MLXArray, MLXArray) {
        let imageMask = inputIds .== MLXArray(config.imageTokenId)
        let videoMask = inputIds .== MLXArray(config.videoTokenId)
        var specialMask = imageMask .|| videoMask
        let nImageTokens = specialMask.sum().item(Int.self)
        specialMask = expandedDimensions(specialMask, axis: -1)
        let maskExpanded = broadcast(specialMask, to: inputEmbeds.shape)
        let nImageFeatures = imageFeatures.dim(0)
        guard maskExpanded.sum().item(Int.self) == imageFeatures.size else {
            throw Qwen4ExpVLError.featureTokenMismatch(
                expected: nImageTokens, actual: nImageFeatures)
        }
        let flattenedEmbeds = inputEmbeds.flattened()
        let flattenedFeatures = imageFeatures.flattened()
        let flattenedMask = maskExpanded.flattened().asArray(Bool.self)
        var result = flattenedEmbeds.asArray(Float.self)
        let feat = flattenedFeatures.asArray(Float.self)
        var fi = 0
        for (idx, flag) in flattenedMask.enumerated() where flag {
            if fi < feat.count {
                result[idx] = feat[fi]
                fi += 1
            }
        }
        let merged = MLXArray(result).reshaped(inputEmbeds.shape).asType(inputEmbeds.dtype)
        return (merged, specialMask.squeezed(axis: -1))
    }
}

extension Qwen4Exp: MtpTarget, MtpDrafter {
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
