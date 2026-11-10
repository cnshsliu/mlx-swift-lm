import Foundation
import MLX
import MLXNN

/// Speculative decoding using a native MTP drafter (hidden + token embed), not a second LM.
public struct MtpTokenIterator: TokenIteratorProtocol {
    let target: any MtpTarget
    let drafter: any MtpDrafter
    var y: LMInput.Text
    var lastHidden: MLXArray
    var mainCache: [KVCache]
    var draftCache: [KVCache]
    var processor: LogitProcessor?
    let sampler: LogitSampler
    public var tokenCount = 0
    public let maxTokens: Int?
    let numDraftTokens: Int
    private var pendingTokens = [Int]()
    private var pendingIndex = 0
    public var promptPrefillTime: TimeInterval = 0.0
    public var totalProposed = 0
    public var totalAccepted = 0
    public var onSpeculationRound: (@Sendable (Int, Int) -> Void)?

    public init(
        input: LMInput,
        target: any MtpTarget,
        drafter: any MtpDrafter,
        mainCache: [KVCache]? = nil,
        parameters: GenerateParameters,
        numDraftTokens: Int,
        processor: LogitProcessor? = nil
    ) throws {
        self.target = target
        self.drafter = drafter
        self.y = input.text
        self.mainCache = try mainCache ?? target.newCache(parameters: parameters)
        self.draftCache = drafter.mtpNewCache(parameters: parameters)
        self.sampler = parameters.sampler()
        self.processor = processor ?? parameters.processor()
        self.maxTokens = parameters.maxTokens
        self.numDraftTokens = Swift.max(1, numDraftTokens)
        self.lastHidden = MLXArray.zeros([1, 1, 1])

        drafter.bindMtp(embed: { tokens in target.mtpEmbed(tokens) }, lmHead: { h in target.mtpLmHead(h) })

        let start = Date()
        try prepare(input: input, windowSize: parameters.prefill.stepSize)
        self.promptPrefillTime = Date().timeIntervalSince(start)
    }

    mutating func prepare(input: LMInput, windowSize: Int?) throws {
        processor?.prompt(input.text.tokens)
        switch try target.prepare(
            input, cache: mainCache, state: nil, prefill: .init(stepSize: windowSize)
        ) {
        case .tokens(let tokens):
            var prompt = tokens.tokens
            if prompt.ndim == 1 { prompt = prompt.reshaped(1, -1) }
            let (hidden, logits) = target.mtpHiddenAndLogits(prompt, cache: mainCache)
            lastHidden = Self.lastTokenHidden(hidden)
            var stepLogits = logits[0..., -1, 0...]
            stepLogits = processor?.process(logits: stepLogits) ?? stepLogits
            let token = sampler.sample(logits: stepLogits)
            processor?.didSample(token: token)
            y = .init(tokens: token)
        case .logits(let result):
            var logits = result.logits[0..., -1, 0...]
            logits = processor?.process(logits: logits) ?? logits
            let token = sampler.sample(logits: logits)
            processor?.didSample(token: token)
            y = .init(tokens: token)
            let (hidden, _) = target.mtpHiddenAndLogits(token, cache: mainCache)
            lastHidden = Self.lastTokenHidden(hidden)
        }
    }

    mutating func speculateRound() {
        let remaining = maxTokens.map { $0 - tokenCount } ?? numDraftTokens
        let numDraft = Swift.min(remaining, numDraftTokens)
        guard numDraft > 0 else { return }

        var hidden = lastHidden
        var tok = y.tokens.ndim == 1 ? y.tokens.reshaped(1, 1) : y.tokens
        if tok.ndim == 1 { tok = tok.reshaped(1, -1) }
        var draftTokens = [MLXArray]()
        for _ in 0 ..< numDraft {
            let embed = target.mtpEmbed(tok)
            hidden = drafter.mtpForward(tokenEmbed: embed, hidden: hidden, cache: draftCache)
            var logits = target.mtpLmHead(hidden)
            logits = logits[0..., -1, 0...]
            // Qwen MTP heads are trained for greedy draft/verify. Sampling
            // independently at temp>0 makes draft and target disagree even
            // when logits match, collapsing accept rate and decode to ~1 tok/s.
            let next = Self.greedyToken(logits)
            asyncEval(next)
            draftTokens.append(next)
            tok = next.ndim == 1 ? next.reshaped(1, 1) : next
        }

        let verifyTokens = [y.tokens.ndim == 1 ? y.tokens.reshaped(1) : y.tokens.flattened()] + draftTokens.map { $0.flattened() }
        let verifyInput = concatenated(verifyTokens)
        let (verifyHidden, verifyLogits) = target.mtpHiddenAndLogits(verifyInput, cache: mainCache)
        let seq = verifyLogits.dim(1)
        let verifyStart = Swift.max(0, seq - (numDraft + 1))

        var sampled = [Int]()
        for i in 0 ..< (numDraft + 1) {
            var logits = verifyLogits[0..., verifyStart + i, 0...]
            logits = processor?.process(logits: logits) ?? logits
            sampled.append(Self.greedyToken(logits).item(Int.self))
        }

        eval(draftTokens)
        let draftIds = draftTokens.map { $0.item(Int.self) }
        var accepted = 0
        for i in 0 ..< numDraft {
            guard sampled[i] == draftIds[i] else { break }
            pendingTokens.append(sampled[i])
            accepted += 1
        }
        totalProposed += numDraft
        totalAccepted += accepted
        onSpeculationRound?(numDraft, accepted)
        pendingTokens.append(sampled[accepted])
        for tid in pendingTokens {
            processor?.didSample(token: MLXArray([Int32(tid)]))
        }

        let extraMain = numDraft - accepted
        if extraMain > 0, canTrimPromptCache(mainCache) {
            trimPromptCache(mainCache, numTokens: extraMain)
        }
        let extraDraft = Swift.max(numDraft - accepted - 1, 0)
        if extraDraft > 0, canTrimPromptCache(draftCache) {
            trimPromptCache(draftCache, numTokens: extraDraft)
        }

        let acceptedHiddenIndex = verifyStart + accepted
        lastHidden = verifyHidden[0..., acceptedHiddenIndex ... acceptedHiddenIndex, 0...]
        y = .init(tokens: MLXArray([sampled[accepted]]))
    }

    private static func greedyToken(_ logits: MLXArray) -> MLXArray {
        argMax(logits, axis: -1)
    }

    private static func lastTokenHidden(_ hidden: MLXArray) -> MLXArray {
        let seq = hidden.dim(1)
        let idx = Swift.max(seq - 1, 0)
        return hidden[0..., idx ... idx, 0...]
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
