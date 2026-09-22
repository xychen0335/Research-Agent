# models/

工作目录里的模型快照。训练和 HuggingFace 推理只 load 这里的目录，不从 Hub 拉权重。

```text
models/
  Qwen3.5-4B/     # 含 config.json、tokenizer、权重
```

把 `Qwen/Qwen3.5-4B` 的 HuggingFace 快照拷到上述目录即可。Ollama GGUF 仍走本机 `ollama pull`，不放在这里。
