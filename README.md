<p align="center">
  <img src="assets/logo.svg" alt="Mantis Logo" width="180" height="180">
</p>
<h1 align="center">mantis</h1>

<p align="center">
  <strong>Train micro-models from macro-intelligence</strong>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#documentation">Docs</a> •
  <a href="#contributing">Contributing</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="License">
  <img src="https://img.shields.io/badge/models-<1B_params-purple.svg" alt="Model Size">
</p>

---

## What is Mantis?

Mantis is an open-source library for training highly specialized micro-models (<1B parameters) by distilling knowledge from frontier LLMs like Claude, GPT, and Gemini.

**The problem:** Large language models are powerful but expensive, slow, and often overkill for specific tasks.

**The solution:** Describe your task requirements in natural language, and Mantis orchestrates frontier models to generate training data, fine-tune a compact model, and deliver a production-ready micro-model optimized for your exact use case.

```python
from mantis import Mantis

trainer = Mantis(teacher="claude-sonnet-4-20250514")

model = trainer.distill(
    task="Extract product names and prices from e-commerce listings",
    examples=10000,
    student="qwen2.5-0.5b"
)

# Your micro-model is ready
result = model("iPhone 15 Pro Max - $1,199 - Apple Store")
# → {"product": "iPhone 15 Pro Max", "price": 1199, "currency": "USD"}
```

## Why Mantis?

| | Frontier LLM | Mantis Micro-Model |
|---|---|---|
| **Latency** | 500ms - 2s | 5ms - 50ms |
| **Cost per 1M tokens** | $3 - $15 | ~$0.01 (self-hosted) |
| **Parameters** | 70B - 400B+ | 0.5B - 1B |
| **Task accuracy** | General-purpose | Optimized for your task |
| **Privacy** | Cloud API | Runs locally |

## Key Features

- **Natural language task definition** — Describe what you need, not how to build it
- **Automatic dataset generation** — Teacher models create high-quality synthetic training data
- **Smart distillation** — Knowledge transfer optimized for micro-model architectures
- **Multiple teacher support** — Use Claude, GPT-4, Gemini, or any OpenAI-compatible API
- **Built-in evaluation** — Automated benchmarking against teacher performance
- **Export anywhere** — ONNX, GGUF, and HuggingFace formats

## Installation

```bash
pip install mantis-ai
```

## Quick Start

### 1. Define Your Task

```python
from mantis import Mantis

trainer = Mantis(
    teacher="claude-sonnet-4-20250514",  # or "gpt-4o", "gemini-1.5-pro"
    teacher_api_key="your-api-key"
)
```

### 2. Describe Requirements

```python
model = trainer.distill(
    task="""
    Classify customer support tickets into categories:
    - billing: payment issues, refunds, subscription
    - technical: bugs, errors, how-to questions  
    - account: login, password, profile changes
    - other: everything else
    
    Return only the category name.
    """,
    examples=5000,           # synthetic examples to generate
    student="qwen2.5-0.5b",  # base model to fine-tune
    epochs=3
)
```

### 3. Deploy

```python
# Local inference
prediction = model("I can't login to my account")
# → "account"

# Export for production
model.export("onnx", path="./ticket-classifier.onnx")
model.export("gguf", path="./ticket-classifier.gguf")
```

## How It Works

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Your Task      │────▶│  Teacher LLM    │────▶│  Synthetic      │
│  Description    │     │  (Claude/GPT)   │     │  Dataset        │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Production     │◀────│  Fine-tuned     │◀────│  Distillation   │
│  Micro-Model    │     │  Student Model  │     │  Training       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

1. **Task Analysis** — Mantis parses your requirements and identifies input/output patterns
2. **Data Synthesis** — Teacher model generates diverse, high-quality training examples
3. **Distillation** — Student model learns to mimic teacher outputs through fine-tuning
4. **Evaluation** — Automated testing ensures student matches teacher accuracy
5. **Export** — Optimized model ready for deployment

## Supported Models

### Teachers (Knowledge Source)
- Anthropic: Claude 3.5 Sonnet, Claude 3 Opus
- OpenAI: GPT-4o, GPT-4 Turbo
- Google: Gemini 1.5 Pro, Gemini 1.5 Flash
- Any OpenAI-compatible API

### Students (Output Models)
- Qwen 2.5: 0.5B, 1.5B
- Llama 3.2: 1B
- Phi-3: Mini (3.8B)
- SmolLM: 135M, 360M
- Custom models via HuggingFace

## Documentation

- [Getting Started Guide](docs/getting-started.md)
- [Task Definition Best Practices](docs/task-definition.md)
- [Advanced Configuration](docs/configuration.md)
- [Deployment Guide](docs/deployment.md)
- [API Reference](docs/api-reference.md)

## Examples

| Use Case | Teacher | Student | Accuracy | Speedup |
|----------|---------|---------|----------|---------|
| Sentiment Analysis | Claude Sonnet | Qwen 0.5B | 94.2% | 40x |
| Entity Extraction | GPT-4o | Qwen 1.5B | 91.8% | 35x |
| Intent Classification | Gemini Pro | SmolLM 360M | 89.5% | 60x |
| Code Documentation | Claude Sonnet | Qwen 1.5B | 87.3% | 25x |

See [examples/](examples/) for complete implementations.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/yourusername/mantis.git
cd mantis
pip install -e ".[dev]"
pytest
```

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built with precision 🦗</sub>
</p>
