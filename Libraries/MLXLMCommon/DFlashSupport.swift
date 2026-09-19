import Foundation
import MLX
import MLXNN

/// True when config.json is a DFlash / DFlash2 drafter, not a chat backbone.
public func isDFlashDraftConfig(at modelDir: URL) -> Bool {
    let cfg = modelDir.appendingPathComponent("config.json")
    guard let data = try? Data(contentsOf: cfg),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return false }
    let arches = (json["architectures"] as? [String]) ?? []
    if arches.contains(where: { $0.lowercased().contains("dflash") }) { return true }
    if json["dflash_config"] is [String: Any] { return true }
    return false
}

/// Companion DFlash2 repos for a backbone id.
public func dflashDraftCandidates(forMainId mainId: String) -> [String] {
    let name = (mainId.split(separator: "/").last.map(String.init) ?? mainId).lowercased()
    if name.contains("-mtp") || name.contains("dflash") || name.contains("dspark") { return [] }
    if name.contains("qwen3.8-27b") || name.contains("qwen3_8-27b") {
        return ["incoai/Qwen3.8-27B-DFlash2"]
    }
    return []
}

/// True when config.json is a DeepSeek DSpark draft head, not a chat backbone.
public func isDSparkDraftConfig(at modelDir: URL) -> Bool {
    let cfg = modelDir.appendingPathComponent("config.json")
    guard let data = try? Data(contentsOf: cfg),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return false }
    let mt = (json["model_type"] as? String ?? "").lowercased()
    if mt.contains("dspark") { return true }
    let arches = (json["architectures"] as? [String]) ?? []
    return arches.contains { $0.lowercased().contains("dspark") }
}

/// Companion DSpark drafter for DeepSeek V4.1 Flash (Qwen DFlash is a different recipe).
public func dsparkDraftCandidates(forMainId mainId: String) -> [String] {
    let name = (mainId.split(separator: "/").last.map(String.init) ?? mainId).lowercased()
    if name.contains("dspark") || name.contains("dflash") || name.contains("-mtp") { return [] }
    if name.contains("deepseek-v4.1-flash") || name.contains("deepseek_v4.1_flash") {
        return ["mlx-community/DeepSeek-V4.1-Flash-DSpark-drafter"]
    }
    return []
}

/// Greedy speculative accept: longest matching prefix of `drafted` vs target argmax,
/// then the first mismatched (bonus) token from the target.
public enum DFlashAccept: Sendable {
    public struct Result: Equatable, Sendable {
        public let accepted: Int
        public let committed: [Int]
    }

    public static func greedy(drafted: [Int], targetArgmax: [Int]) -> Result {
        var accepted = 0
        while accepted < drafted.count && accepted < targetArgmax.count
            && drafted[accepted] == targetArgmax[accepted]
        {
            accepted += 1
        }
        var committed = Array(drafted.prefix(accepted))
        if accepted < targetArgmax.count {
            committed.append(targetArgmax[accepted])
        }
        return Result(accepted: accepted, committed: committed)
    }
}

/// DFlash2 / GDN interaction limits. The old 8192 prompt cap was a workaround
/// for recording a full-prefill spec capture, then verifying a short block
/// against that stale conv window (`mlx_reshape` SIGTRAP).
public enum DFlashLimits: Sendable {
    /// Only the short `[anchor]+drafts` verify window should snapshot GDN state.
    public static let maxSpecRecordTokens = 32
    /// Prefill the target in chunks so GDN conv never sees a 10k+ sequence in one
    /// Convolution, and so the lazy graph is eval'd before the first verify.
    public static let prefillChunkTokens = 4096

    public static func shouldRecordSpec(sequenceLength: Int) -> Bool {
        sequenceLength > 1 && sequenceLength <= maxSpecRecordTokens
    }
}

/// Target that can feed a DFlash drafter (hidden-state taps + hybrid rollback).
public protocol DFlashTarget: LanguageModel {
    var dflashEmbedding: Embedding { get }
    func dflashEmbed(_ tokens: MLXArray) -> MLXArray
    func dflashLmHead(_ hidden: MLXArray) -> MLXArray
    /// Logits `[B,S,V]` and fused tap hidden `[B,S, nTap*H]` for `[anchor]+drafts`.
    func dflashVerify(
        _ tokens: MLXArray, cache: [KVCache]?, tapLayerIds: [Int]
    ) -> (logits: MLXArray, fused: MLXArray)
    func dflashRollback(cache: [KVCache], nRejected: Int, specWidth: Int)
}
