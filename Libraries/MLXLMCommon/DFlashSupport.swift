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

/// Whether one realized DFlash block may commit tokens.
/// `specWidth` is the token width submitted to verify (anchor plus drafts).
public enum DFlashBlockCheck: Sendable {
    public struct Decision: Equatable, Sendable {
        public let acceptNormally: Bool
        public let specWidth: Int
        public let nRejected: Int
    }

    public static func decide(
        draftCount: Int, logitLength: Int, fusedLength: Int, accepted: Int
    ) -> Decision {
        let width = draftCount + 1
        let lengthsAgree = logitLength == width && fusedLength == width
        let acceptedInRange = accepted >= 0 && accepted <= draftCount
        if lengthsAgree && acceptedInRange {
            return Decision(
                acceptNormally: true,
                specWidth: width,
                nRejected: draftCount - accepted)
        }
        return Decision(
            acceptNormally: false,
            specWidth: width,
            nRejected: draftCount)
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

/// Per-request draft length. `cap == 0` is one target token and no drafter call.
/// Starts at `maxCap`. After `minRoundsBeforePark` speculative rounds, a rolling
/// mean of accepted draft tokens below `parkBelow` parks the drafter.
/// A parked request stays at cap 0. A full-width probe costs a wide verify, which
/// on this model is the slow part, so it does not buy its way back on. A later
/// speculative round that does run can still leave the park when it accepts
/// at least `parkBelow` tokens.
public struct DFlashCapPolicy: Sendable {
    public let maxCap: Int
    public private(set) var cap: Int
    public private(set) var rounds: Int
    public let windowSize: Int
    public let parkBelow: Double
    public let minRoundsBeforePark: Int
    public let probeEvery: Int
    private var acceptedWindow: [Int]
    private var parked: Bool

    public init(
        maxCap: Int,
        windowSize: Int = 4,
        parkBelow: Double = 3,
        minRoundsBeforePark: Int = 3,
        probeEvery: Int = 32
    ) {
        let maxCap = Swift.max(1, maxCap)
        self.maxCap = maxCap
        self.cap = maxCap
        self.rounds = 0
        self.windowSize = Swift.max(1, windowSize)
        self.parkBelow = parkBelow
        self.minRoundsBeforePark = Swift.max(1, minRoundsBeforePark)
        self.probeEvery = Swift.max(2, probeEvery)
        self.acceptedWindow = []
        self.parked = false
    }

    public mutating func observe(accepted: Int, capUsed: Int) {
        rounds += 1
        if capUsed <= 0 {
            cap = 0
            return
        }
        let accepted = Swift.max(0, accepted)
        if parked {
            if Double(accepted) >= parkBelow {
                parked = false
                acceptedWindow = [accepted]
                cap = Swift.min(maxCap, accepted + 1)
            } else {
                cap = 0
            }
            return
        }
        acceptedWindow.append(accepted)
        if acceptedWindow.count > windowSize {
            acceptedWindow.removeFirst()
        }
        if acceptedWindow.count < minRoundsBeforePark {
            cap = maxCap
            return
        }
        let mean = Double(acceptedWindow.reduce(0, +)) / Double(acceptedWindow.count)
        if mean < parkBelow {
            parked = true
            cap = 0
            return
        }
        if accepted >= capUsed {
            let opened = capUsed < maxCap ? capUsed + 1 : maxCap
            cap = Swift.min(maxCap, Swift.max(cap, opened))
            return
        }
        let withHeadroom = Int(mean.rounded()) + 1
        cap = Swift.min(maxCap, Swift.max(1, withHeadroom))
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
