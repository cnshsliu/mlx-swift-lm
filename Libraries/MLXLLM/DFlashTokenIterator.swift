import Foundation
import MLX
import MLXLMCommon
import MLXNN

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
    public var onSpeculationRound: (@Sendable (Int, Int) -> Void)?

    private var pendingTokens = [Int]()
    private var pendingIndex = 0
    private var pending: Int = 0
    private var pendingCtx: MLXArray
    private let cap: Int
    private let blockSize: Int
    private let maskId: Int
    private let tap: [Int]
    private let greedy: Bool

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
        self.cap = Swift.max(1, blockSize - 1)
        self.maskId = drafter.config.maskTokenId
        self.tap = drafter.config.targetLayerIds
        self.greedy = parameters.temperature <= 0
        self.pendingCtx = MLXArray.zeros([1, 1, 1])

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
        let remaining = maxTokens.map { $0 - tokenCount } ?? cap
        let thisCap = Swift.min(cap, Swift.max(1, remaining))
        guard thisCap > 0 else { return }

        let blockIds = [Int32(pending)] + Array(repeating: Int32(maskId), count: blockSize - 1)
        let block = MLXArray(blockIds, [1, blockSize])
        let draftArr = drafter.selectBlock(
            block: block, targetHidden: pendingCtx, cache: draftCache,
            cap: thisCap, anchorId: pending)
        eval(draftArr)
        let drafted = draftArr.asArray(Int.self)

        var verifyList = [Int32(pending)]
        verifyList.append(contentsOf: drafted.map { Int32($0) })
        let verifyIds = MLXArray(verifyList, [1, verifyList.count])
        let (vLogits, vFused) = target.dflashVerify(
            verifyIds, cache: mainCache, tapLayerIds: tap)
        var argmax = [Int]()
        let seq = vLogits.dim(1)
        for i in 0 ..< seq {
            var row = vLogits[0..., i, 0...]
            row = processor?.process(logits: row) ?? row
            argmax.append(argMax(row, axis: -1).item(Int.self))
        }
        let result = DFlashAccept.greedy(drafted: drafted, targetArgmax: argmax)
        let n = result.accepted
        totalProposed += drafted.count
        totalAccepted += n
        onSpeculationRound?(drafted.count, n)
        target.dflashRollback(cache: mainCache, nRejected: drafted.count - n, specWidth: 1 + drafted.count)
        pendingCtx = vFused[0..., ..<(n + 1), 0...]
        pendingTokens.append(contentsOf: result.committed)
        if let last = result.committed.last {
            pending = last
            for tid in result.committed {
                processor?.didSample(token: MLXArray([Int32(tid)]))
            }
        }
    }

    public mutating func next() -> Int? {
        if pendingIndex < pendingTokens.count {
            let t = pendingTokens[pendingIndex]
            pendingIndex += 1
            tokenCount += 1
            if let maxTokens, tokenCount >= maxTokens { return nil }
            return t
        }
        pendingTokens.removeAll(keepingCapacity: true)
        pendingIndex = 0
        if let maxTokens, tokenCount >= maxTokens { return nil }
        speculateRound()
        if pendingTokens.isEmpty { return nil }
        let t = pendingTokens[pendingIndex]
        pendingIndex += 1
        tokenCount += 1
        if let maxTokens, tokenCount >= maxTokens { return nil }
        return t
    }
}


