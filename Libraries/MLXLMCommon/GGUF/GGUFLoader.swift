import Foundation
import MLX

// MARK: - Public load API

/// Load weights from GGUF files in a model directory.
///
/// Parses `.gguf` files, dequantizes tensors to float16, and maps llama.cpp
/// tensor names onto Hugging Face / MLX module paths.
public func loadGGUFWeights(modelDirectory: URL) throws -> [String: MLXArray] {
    let ggufFiles = findGGUFFiles(in: modelDirectory)
    guard !ggufFiles.isEmpty else {
        throw GGUFError.readError("No .gguf files found in \(modelDirectory.path)")
    }

    var allWeights: [String: MLXArray] = [:]
    var architecture: String?

    for fileURL in ggufFiles {
        let gguf = try GGUFFile.parse(url: fileURL)
        if architecture == nil {
            architecture = gguf.architecture
        }
        if gguf.expertCount > 0 {
            throw GGUFError.readError(
                "MoE GGUF (expert_count=\(gguf.expertCount)) is not supported. Convert to MLX safetensors."
            )
        }
        let weights = try loadTensors(from: gguf)
        for (key, value) in weights {
            allWeights[key] = value
        }
    }

    let mapper = GGUFWeightMapper(architecture: architecture ?? "llama")
    return mapper.mapWeights(allWeights)
}

/// True when `directory` contains at least one `.gguf` file (non-recursive).
public func hasGGUFWeights(in directory: URL) -> Bool {
    !findGGUFFiles(in: directory).isEmpty
}

/// Write `config.json` / tokenizer sidecars from GGUF metadata when they are missing.
///
/// `LLMModelFactory` constructs the Swift model from `config.json` before
/// `loadWeights` runs, so GGUF-only folders must be prepared first.
public func prepareGGUFModelDirectory(_ directory: URL) throws {
    guard hasGGUFWeights(in: directory) else { return }
    let files = findGGUFFiles(in: directory)
    guard let first = files.first else { return }
    let gguf = try GGUFFile.parse(url: first)
    try GGUFHuggingFaceConfig.writeIfNeeded(from: gguf, into: directory)
    try GGUFTokenizerSidecar.writeIfNeeded(from: gguf, into: directory)
}

func findGGUFFiles(in directory: URL) -> [URL] {
    let fm = FileManager.default
    guard
        let contents = try? fm.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
    else { return [] }
    return contents.filter { $0.pathExtension.lowercased() == "gguf" }.sorted {
        $0.lastPathComponent < $1.lastPathComponent
    }
}

func loadTensors(from gguf: GGUFFile) throws -> [String: MLXArray] {
    var weights: [String: MLXArray] = [:]
    weights.reserveCapacity(gguf.tensors.count)
    var skipped: [String] = []

    for (name, info) in gguf.tensors {
        if name == "rope_freqs" || name.hasPrefix("rope_freqs") { continue }
        guard info.type.isSupported else {
            skipped.append("\(name) (\(info.type))")
            continue
        }
        let data = try gguf.readTensorData(name)
        weights[name] = try GGUFDequant.toMLXArray(data: data, info: info)
    }

    if !skipped.isEmpty {
        throw GGUFError.readError(
            "Unsupported GGUF quantizations: \(skipped.prefix(8).joined(separator: ", ")). Convert to Q4_K / Q5_K / Q6_K / Q8_0 or MLX safetensors."
        )
    }
    return weights
}

// MARK: - Hugging Face config.json

public enum GGUFHuggingFaceConfig {
    static func writeIfNeeded(from gguf: GGUFFile, into directory: URL) throws {
        let url = directory.appendingPathComponent("config.json")
        if FileManager.default.fileExists(atPath: url.path) { return }
        let json = try makeJSON(from: gguf)
        try json.write(to: url, atomically: true, encoding: .utf8)
    }

    public static func hfModelType(ggufArch: String) -> String {
        switch ggufArch.lowercased() {
        case "llama": return "llama"
        case "mistral": return "mistral"
        case "qwen2": return "qwen2"
        case "qwen3": return "qwen3"
        case "qwen3moe": return "qwen3_moe"
        case "gemma": return "gemma"
        case "gemma2": return "gemma2"
        case "gemma3": return "gemma3_text"
        case "phi3": return "phi3"
        case "phi2", "phi": return "phi"
        case "starcoder2": return "starcoder2"
        case "internlm2": return "internlm2"
        case "minicpm3", "minicpm": return "minicpm"
        case "deepseek2": return "deepseek_v2"
        case "granite": return "granite"
        case "command-r", "command_r": return "cohere"
        default:
            return ggufArch.lowercased().replacingOccurrences(of: "-", with: "_")
        }
    }

    public static func architectureName(hfType: String) -> String {
        switch hfType {
        case "llama": return "LlamaForCausalLM"
        case "mistral": return "MistralForCausalLM"
        case "qwen2": return "Qwen2ForCausalLM"
        case "qwen3": return "Qwen3ForCausalLM"
        case "qwen3_moe": return "Qwen3MoeForCausalLM"
        case "gemma": return "GemmaForCausalLM"
        case "gemma2": return "Gemma2ForCausalLM"
        case "gemma3_text": return "Gemma3ForCausalLM"
        case "phi3": return "Phi3ForCausalLM"
        case "phi": return "PhiForCausalLM"
        case "starcoder2": return "Starcoder2ForCausalLM"
        case "internlm2": return "InternLM2ForCausalLM"
        case "minicpm": return "MiniCPMForCausalLM"
        case "deepseek_v2": return "DeepseekV2ForCausalLM"
        case "granite": return "GraniteForCausalLM"
        case "cohere": return "CohereForCausalLM"
        default: return "LlamaForCausalLM"
        }
    }

    static func makeJSON(from gguf: GGUFFile) throws -> String {
        guard let arch = gguf.architecture, !arch.isEmpty else {
            throw GGUFError.readError("GGUF is missing general.architecture")
        }
        if gguf.expertCount > 0 && hfModelType(ggufArch: arch) != "qwen3_moe" {
            throw GGUFError.readError(
                "MoE GGUF architecture '\(arch)' is not supported. Convert to MLX safetensors."
            )
        }
        let hfType = hfModelType(ggufArch: arch)
        let hidden = gguf.archInt("embedding_length") ?? 0
        let layers = gguf.archInt("block_count") ?? 0
        let heads = gguf.archInt("attention.head_count") ?? 0
        let kvHeads = gguf.archInt("attention.head_count_kv") ?? heads
        let intermediate = gguf.archInt("feed_forward_length") ?? 0
        let ctx = gguf.archInt("context_length") ?? 4096
        let vocab =
            gguf.archInt("vocab_size")
            ?? gguf.meta("tokenizer.ggml.tokens")?.stringArray?.count
            ?? 0
        let rms =
            gguf.archFloat("attention.layer_norm_rms_epsilon")
            ?? gguf.archFloat("attention.layer_norm_epsilon")
            ?? 1e-6
        let rope = gguf.archFloat("rope.freq_base") ?? 10_000
        let headDim = gguf.archInt("attention.key_length")
        let tie = gguf.tensors["output.weight"] == nil
        let bos = gguf.meta("tokenizer.ggml.bos_token_id")?.intValue
        let eos = gguf.meta("tokenizer.ggml.eos_token_id")?.intValue

        guard hidden > 0, layers > 0, heads > 0, vocab > 0 else {
            throw GGUFError.readError(
                "GGUF metadata incomplete for '\(arch)' (hidden=\(hidden) layers=\(layers) heads=\(heads) vocab=\(vocab))"
            )
        }

        var obj: [String: Any] = [
            "model_type": hfType,
            "architectures": [architectureName(hfType: hfType)],
            "hidden_size": hidden,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": kvHeads,
            "intermediate_size": intermediate > 0 ? intermediate : hidden * 4,
            "rms_norm_eps": rms,
            "vocab_size": vocab,
            "max_position_embeddings": ctx,
            "rope_theta": rope,
            "tie_word_embeddings": tie,
            "torch_dtype": "float16",
        ]
        if let headDim { obj["head_dim"] = headDim }
        if let bos { obj["bos_token_id"] = bos }
        if let eos { obj["eos_token_id"] = eos }
        if let ropeDim = gguf.archInt("rope.dimension_count") {
            obj["partial_rotary_factor"] = Float(ropeDim) / Float(headDim ?? (hidden / max(heads, 1)))
        }

        let data = try JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys])
        return String(data: data, encoding: .utf8) ?? "{}"
    }
}

// MARK: - Tokenizer sidecar

enum GGUFTokenizerSidecar {
    static func writeIfNeeded(from gguf: GGUFFile, into directory: URL) throws {
        let tokJSON = directory.appendingPathComponent("tokenizer.json")
        if FileManager.default.fileExists(atPath: tokJSON.path) {
            writeTokenizerConfigIfNeeded(from: gguf, into: directory, overwrite: false)
            return
        }
        guard let tokens = gguf.meta("tokenizer.ggml.tokens")?.stringArray, !tokens.isEmpty else {
            return
        }
        let merges = gguf.meta("tokenizer.ggml.merges")?.stringArray ?? []
        let scores = gguf.meta("tokenizer.ggml.scores")?.floatArray ?? []
        let modelType = gguf.meta("tokenizer.ggml.model")?.stringValue ?? ""

        var vocab: [String: Int] = [:]
        vocab.reserveCapacity(tokens.count)
        for (i, tok) in tokens.enumerated() {
            vocab[tok] = i
        }

        var model: [String: Any]
        if !merges.isEmpty || modelType.lowercased() == "gpt2" {
            model = [
                "type": "BPE",
                "dropout": NSNull(),
                "unk_token": NSNull(),
                "continuing_subword_prefix": "",
                "end_of_word_suffix": "",
                "fuse_unk": false,
                "byte_fallback": true,
                "vocab": vocab,
                "merges": merges,
            ]
        } else {
            let unigram: [[Any]] = tokens.enumerated().map { idx, tok in
                let score = idx < scores.count ? scores[idx] : 0.0
                return [tok, score]
            }
            model = [
                "type": "Unigram",
                "unk_id": gguf.meta("tokenizer.ggml.unknown_token_id")?.intValue ?? 0,
                "vocab": unigram,
            ]
        }

        let bosId = gguf.meta("tokenizer.ggml.bos_token_id")?.intValue
        let eosId = gguf.meta("tokenizer.ggml.eos_token_id")?.intValue
        let unkId = gguf.meta("tokenizer.ggml.unknown_token_id")?.intValue
        var added: [[String: Any]] = []
        func addSpecial(_ id: Int?) {
            guard let id, id >= 0, id < tokens.count else { return }
            added.append([
                "id": id,
                "content": tokens[id],
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true,
            ])
        }
        addSpecial(unkId)
        addSpecial(bosId)
        addSpecial(eosId)

        let tokenizer: [String: Any] = [
            "version": "1.0",
            "truncation": NSNull(),
            "padding": NSNull(),
            "added_tokens": added,
            "normalizer": NSNull(),
            "pre_tokenizer": [
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": true,
            ],
            "post_processor": [
                "type": "ByteLevel",
                "add_prefix_space": true,
                "trim_offsets": false,
                "use_regex": true,
            ],
            "decoder": [
                "type": "ByteLevel",
                "add_prefix_space": true,
                "trim_offsets": true,
                "use_regex": true,
            ],
            "model": model,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: tokenizer, options: [.prettyPrinted])
        try data.write(to: tokJSON, options: .atomic)
        writeTokenizerConfigIfNeeded(from: gguf, into: directory, overwrite: true)
    }

    private static func writeTokenizerConfigIfNeeded(
        from gguf: GGUFFile, into directory: URL, overwrite: Bool
    ) {
        let url = directory.appendingPathComponent("tokenizer_config.json")
        if !overwrite, FileManager.default.fileExists(atPath: url.path) { return }
        let tokens = gguf.meta("tokenizer.ggml.tokens")?.stringArray ?? []
        func token(_ key: String) -> String? {
            guard let id = gguf.meta(key)?.intValue, id >= 0, id < tokens.count else { return nil }
            return tokens[id]
        }
        var obj: [String: Any] = [:]
        if let bos = token("tokenizer.ggml.bos_token_id") { obj["bos_token"] = bos }
        if let eos = token("tokenizer.ggml.eos_token_id") { obj["eos_token"] = eos }
        if let unk = token("tokenizer.ggml.unknown_token_id") { obj["unk_token"] = unk }
        if let pad = token("tokenizer.ggml.padding_token_id") { obj["pad_token"] = pad }
        if let tmpl = gguf.meta("tokenizer.chat_template")?.stringValue {
            obj["chat_template"] = tmpl
        }
        guard !obj.isEmpty,
            let data = try? JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted])
        else { return }
        try? data.write(to: url, options: .atomic)
    }
}

// MARK: - GGUF → MLX name map

struct GGUFWeightMapper {
    let architecture: String

    func mapWeights(_ ggufWeights: [String: MLXArray]) -> [String: MLXArray] {
        var mapped: [String: MLXArray] = [:]
        mapped.reserveCapacity(ggufWeights.count)
        for (name, array) in ggufWeights {
            if let mappedName = mapName(name) {
                mapped[mappedName] = array
            }
        }
        return mapped
    }

    private func mapName(_ ggufName: String) -> String? {
        let hasWeight = ggufName.hasSuffix(".weight")
        let hasBias = ggufName.hasSuffix(".bias")
        let suffix: String
        let baseName: String
        if hasWeight {
            suffix = ".weight"
            baseName = String(ggufName.dropLast(7))
        } else if hasBias {
            suffix = ".bias"
            baseName = String(ggufName.dropLast(5))
        } else {
            suffix = ""
            baseName = ggufName
        }
        guard let mappedBase = mapBaseName(baseName) else { return nil }
        return mappedBase + suffix
    }

    private func mapBaseName(_ name: String) -> String? {
        switch name {
        case "token_embd": return "model.embed_tokens"
        case "output_norm": return "model.norm"
        case "output": return "lm_head"
        case "rope_freqs": return nil
        default: break
        }
        if name.hasPrefix("blk.") {
            return mapBlockName(name)
        }
        return name
    }

    private func mapBlockName(_ name: String) -> String? {
        let parts = name.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count >= 3, parts[0] == "blk" else { return nil }
        let layerIdx = parts[1]
        let component = parts.dropFirst(2).joined(separator: ".")
        guard let comp = mapComponent(component) else { return nil }
        return "model.layers.\(layerIdx).\(comp)"
    }

    private func mapComponent(_ component: String) -> String? {
        switch component {
        case "attn_norm": return "input_layernorm"
        case "attn_q": return "self_attn.q_proj"
        case "attn_k": return "self_attn.k_proj"
        case "attn_v": return "self_attn.v_proj"
        case "attn_output": return "self_attn.o_proj"
        case "attn_q_norm": return "self_attn.q_norm"
        case "attn_k_norm": return "self_attn.k_norm"
        case "attn_post_norm": return "post_attention_layernorm"
        case "ffn_norm": return "post_attention_layernorm"
        case "ffn_gate": return "mlp.gate_proj"
        case "ffn_up": return "mlp.up_proj"
        case "ffn_down": return "mlp.down_proj"
        case "attn_norm_2": return "pre_feed_forward_layernorm"
        case "ffn_norm_2": return "post_feed_forward_layernorm"
        case "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_inp":
            return nil
        default: return nil
        }
    }
}
