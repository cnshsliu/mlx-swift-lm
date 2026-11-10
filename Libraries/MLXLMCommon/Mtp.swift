import Foundation
import MLX
import MLXNN

/// Target model that can drive a native MTP drafter (hidden states + shared embed/lm_head).
public protocol MtpTarget: LanguageModel {
    func mtpEmbed(_ tokens: MLXArray) -> MLXArray
    func mtpLmHead(_ hidden: MLXArray) -> MLXArray
    /// Hidden states `[B, S, D]` and logits `[B, S, V]` for `tokens`.
    func mtpHiddenAndLogits(_ tokens: MLXArray, cache: [KVCache]?) -> (hidden: MLXArray, logits: MLXArray)
}

/// Split `qwen3_5_mtp` (or in-graph `mtp.*`) drafter. Not a chat model.
public protocol MtpDrafter: LanguageModel {
    var mtpBlockSize: Int { get }
    func bindMtp(embed: @escaping (MLXArray) -> MLXArray, lmHead: @escaping (MLXArray) -> MLXArray)
    func mtpForward(tokenEmbed: MLXArray, hidden: MLXArray, cache: [KVCache]?) -> MLXArray
    /// KV cache for the MTP head (may be smaller than the backbone `newCache()`).
    func mtpNewCache(parameters: GenerateParameters?) -> [KVCache]
}

extension MtpDrafter {
    public func mtpNewCache(parameters: GenerateParameters?) -> [KVCache] {
        (try? newCache(parameters: parameters)) ?? []
    }
}

extension LanguageModel {
    public var isMtpDrafter: Bool { self is any MtpDrafter }
}

/// True when config.json is an MTP draft head, not a backbone.
public func isMtpDraftConfig(at modelDir: URL) -> Bool {
    let cfg = modelDir.appendingPathComponent("config.json")
    guard let data = try? Data(contentsOf: cfg),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return false }
    let mt = (json["model_type"] as? String ?? "").lowercased()
    if mt == "qwen3_5_mtp" || mt.hasSuffix("_mtp") || mt.contains("dspark") { return true }
    return false
}

/// True when the checkpoint actually contains MTP tensors (not just a config flag).
public func checkpointHasMtpWeights(at modelDir: URL) -> Bool {
    if isMtpDraftConfig(at: modelDir) { return true }
    let index = modelDir.appendingPathComponent("model.safetensors.index.json")
    if let data = try? Data(contentsOf: index),
       let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
       let map = json["weight_map"] as? [String: Any]
    {
        return map.keys.contains {
            $0.contains("mtp.") || $0.hasPrefix("mtp") || $0.contains("mtp_layers")
        }
    }
    // Single-file safetensors: treat qwen3_5_mtp config as sufficient.
    return false
}

/// Sibling draft id for a backbone, e.g. `…/Qwen3.8-27B-8bit` → `…/Qwen3.8-27B-MTP-4bit`.
public func mtpDraftCandidates(forMainId mainId: String) -> [String] {
    let parts = mainId.split(separator: "/", maxSplits: 1).map(String.init)
    let org = parts.count == 2 ? parts[0] : ""
    let name = parts.count == 2 ? parts[1] : mainId
    var stems = [name]
    for suffix in ["-8bit", "-4bit", "-6bit", "-3bit", "-2bit"] where name.hasSuffix(suffix) {
        stems.append(String(name.dropLast(suffix.count)))
    }
    var out: [String] = []
    var seen = Set<String>()
    func add(_ id: String) {
        if seen.insert(id).inserted { out.append(id) }
    }
    for stem in stems {
        if !stem.contains("-MTP") {
            add(org.isEmpty ? "\(stem)-MTP-4bit" : "\(org)/\(stem)-MTP-4bit")
            add(org.isEmpty ? "\(stem)-MTP-8bit" : "\(org)/\(stem)-MTP-8bit")
        }
    }
    return out
}
