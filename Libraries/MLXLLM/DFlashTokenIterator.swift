import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// One DFlash round. `cap == 0` is a dense target step (no drafted tokens).
public struct DFlashRoundStats: Sendable {
    public let proposed: Int
    public let accepted: Int
    public let cap: Int
    public let draftMs: Double
    public let verifyMs: Double
    public let acceptMs: Double
    public let rollbackMs: Double
}

/// Speculative decoding with a DFlash / DFlash2 block-diffusion drafter.
public struct DFlashTokenIterator: TokenIteratorProtocol {
    let target: any DFlashTarget
    let drafter: DFlashDraftModel
    var mainCache: [KVCache]
    var draftCache: [DFlashCtxCache]
    var processor: LogitProcessor?
    let sampler: LogitSampler
    public var tokenCount = 0
    public let maxTokens: Int?
    public var promptPrefillTime: TimeInterval = 0.0
    public var totalProposed = 0
    public var totalAccepted = 0
    public var onSpeculationRound: (@Sendable (DFlashRoundStats) -> Void)?

    private var pendingTokens = [Int]()
    private var pendingIndex = 0
    private var pending: Int = 0
    private var pendingCtx: MLXArray
    /// `pendingCtx` is not in the draft cache yet. A block forward writes it.
    /// A parked step writes it once, then stops: later parked steps have no new
    /// target hidden, and writing the same row again would shift the cache.
    private var contextNeedsFlush: Bool
    private var policy: DFlashCapPolicy
    private let blockSize: Int
    private let maskId: Int
    private let tap: [Int]
    private let greedy: Bool
    /// `NOVAMLX_DFLASH_PROFILE=1` evals each stage before stopping its clock.
    private let profileSync: Bool

    public init(
        input: LMInput,
        target: any DFlashTarget,
        drafter: DFlashDraftModel,
        mainCache: [KVCache]? = nil,
        parameters: GenerateParameters,
        processor: LogitProcessor? = nil
    ) throws {
        self.target = target
        self.drafter = drafter
        self.mainCache = try mainCache ?? target.newCache(parameters: parameters)
        self.draftCache = drafter.makeCache()
        self.sampler = parameters.sampler()
        self.processor = processor ?? parameters.processor()
        self.maxTokens = parameters.maxTokens
        self.blockSize = drafter.config.blockSize
        // One losing round is enough to park. Three forced full blocks were a
        // tax on chat: each one cost more per token than a plain step.
        self.policy = DFlashCapPolicy(
            maxCap: Swift.max(1, blockSize - 1), minRoundsBeforePark: 1)
        self.maskId = drafter.config.maskTokenId
        self.tap = drafter.config.targetLayerIds
        self.greedy = parameters.temperature <= 0
        self.profileSync = ProcessInfo.processInfo.environment["NOVAMLX_DFLASH_PROFILE"] == "1"
        self.pendingCtx = MLXArray.zeros([1, 1, 1])
        self.contextNeedsFlush = true

        drafter.bind(
            embed: target.dflashEmbedding,
            lmHead: { [target] h in target.dflashLmHead(h) }
        )

        let start = Date()
        try prepare(input: input)
        self.promptPrefillTime = Date().timeIntervalSince(start)
    }

    mutating func prepare(input: LMInput) throws {
        processor?.prompt(input.text.tokens)
        var prompt = input.text.tokens
        if prompt.ndim == 1 { prompt = prompt.reshaped(1, -1) }
        let total = prompt.dim(1)
        guard total > 0 else { return }
        let chunk = DFlashLimits.prefillChunkTokens
        var logits = MLXArray.zeros([1, 1, 1])
        var fused = MLXArray.zeros([1, 1, 1])
        var start = 0
        while start < total {
            let end = Swift.min(start + chunk, total)
            let slice = prompt[0..., start ..< end]
            let step = target.dflashVerify(slice, cache: mainCache, tapLayerIds: tap)
            eval(step.logits, step.fused)
            logits = step.logits
            fused = step.fused
            start = end
        }
        pendingCtx = fused.dim(1) > 1 ? fused[0..., (-1)..., 0...] : fused
        var stepLogits = logits[0..., -1, 0...]
        stepLogits = processor?.process(logits: stepLogits) ?? stepLogits
        let token: Int
        if greedy {
            token = argMax(stepLogits, axis: -1).item(Int.self)
        } else {
            token = sampler.sample(logits: stepLogits).item(Int.self)
        }
        processor?.didSample(token: MLXArray([Int32(token)]))
        pending = token
        pendingTokens = [token]
    }

    mutating func speculateRound() {
        let budget = maxTokens.map { $0 - tokenCount } ?? (policy.maxCap + 1)
        guard budget > 0 else { return }
        // A round commits the accepted drafts plus one bonus token. Leave that
        // bonus inside the remaining budget so the last token matches dense decode.
        let thisCap = Swift.min(policy.cap, budget - 1)
        if thisCap <= 0 {
            denseStep()
            return
        }

        // The block stays full width. Non-sliding draft layers attend both ways, so a
        // shorter block would change the draft. `thisCap` only shortens the walk and the verify.
        // selectBlock writes `pendingCtx` into the draft cache.
        contextNeedsFlush = false
        let tDraft = Date()
        let blockIds = [Int32(pending)] + Array(repeating: Int32(maskId), count: blockSize - 1)
        let block = MLXArray(blockIds, [1, blockSize])
        let draftArr = drafter.selectBlock(
            block: block, targetHidden: pendingCtx, cache: draftCache,
            cap: thisCap, anchorId: pending)
        eval(draftArr)
        let drafted = draftArr.asArray(Int.self)
        let draftMs = Date().timeIntervalSince(tDraft) * 1000

        let tVerify = Date()
        var verifyList = [Int32(pending)]
        verifyList.append(contentsOf: drafted.map { Int32($0) })
        let verifyIds = MLXArray(verifyList, [1, verifyList.count])
        let (vLogits, vFused) = target.dflashVerify(
            verifyIds, cache: mainCache, tapLayerIds: tap)
        let argmax = argmaxRows(vLogits)
        if profileSync {
            eval(vLogits, vFused)
            eval(mainCache.flatMap { $0.state })
        }
        let verifyMs = Date().timeIntervalSince(tVerify) * 1000
        finishRound(
            drafted: drafted, argmax: argmax, fused: vFused,
            draftMs: draftMs, verifyMs: verifyMs)
        contextNeedsFlush = true
    }

    /// One target token, on the same single-token forward a draft-off step uses.
    /// The wide `dflashVerify` path stays for real draft blocks only.
    private mutating func denseStep() {
        let tDraft = Date()
        if contextNeedsFlush {
            drafter.appendContext(pendingCtx, cache: draftCache)
            contextNeedsFlush = false
        }
        let draftMs = Date().timeIntervalSince(tDraft) * 1000

        let tVerify = Date()
        let verifyIds = MLXArray([Int32(pending)], [1, 1])
        var row = target(verifyIds, cache: mainCache)
        row = row[0..., -1, 0...]
        row = processor?.process(logits: row) ?? row
        let token = argMax(row, axis: -1).item(Int.self)
        asyncEval(mainCache.flatMap { $0.state })
        let verifyMs = Date().timeIntervalSince(tVerify) * 1000

        totalProposed += 0
        pendingTokens.append(token)
        pending = token
        processor?.didSample(token: MLXArray([Int32(token)]))
        policy.observe(accepted: 0, capUsed: 0)
        onSpeculationRound?(
            DFlashRoundStats(
                proposed: 0, accepted: 0, cap: 0,
                draftMs: draftMs, verifyMs: verifyMs,
                acceptMs: 0, rollbackMs: 0))
    }

    private func argmaxRows(_ logits: MLXArray) -> [Int] {
        var argmax = [Int]()
        let seq = logits.dim(1)
        for i in 0 ..< seq {
            var row = logits[0..., i, 0...]
            row = processor?.process(logits: row) ?? row
            argmax.append(argMax(row, axis: -1).item(Int.self))
        }
        return argmax
    }

    private mutating func finishRound(
        drafted: [Int], argmax: [Int], fused: MLXArray,
        draftMs: Double, verifyMs: Double
    ) {
        let tAccept = Date()
        let result = DFlashAccept.greedy(drafted: drafted, targetArgmax: argmax)
        let acceptMs = Date().timeIntervalSince(tAccept) * 1000
        let n = result.accepted
        totalProposed += drafted.count
        totalAccepted += n
        let tRoll = Date()
        target.dflashRollback(
            cache: mainCache, nRejected: drafted.count - n, specWidth: 1 + drafted.count)
        if profileSync {
            eval(mainCache.flatMap { $0.state })
        }
        let rollbackMs = Date().timeIntervalSince(tRoll) * 1000
        let keep = Swift.min(n + 1, fused.dim(1))
        pendingCtx = fused[0..., ..<keep, 0...]
        pendingTokens.append(contentsOf: result.committed)
        if let last = result.committed.last {
            pending = last
            for tid in result.committed {
                processor?.didSample(token: MLXArray([Int32(tid)]))
            }
        }
        policy.observe(accepted: n, capUsed: drafted.count)
        onSpeculationRound?(
            DFlashRoundStats(
                proposed: drafted.count, accepted: n, cap: drafted.count,
                draftMs: draftMs, verifyMs: verifyMs,
                acceptMs: acceptMs, rollbackMs: rollbackMs))
    }

    public mutating func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens { return nil }
        if pendingIndex >= pendingTokens.count {
            pendingTokens.removeAll(keepingCapacity: true)
            pendingIndex = 0
            speculateRound()
            if pendingTokens.isEmpty { return nil }
        }
        let t = pendingTokens[pendingIndex]
        pendingIndex += 1
        tokenCount += 1
        return t
    }
}


