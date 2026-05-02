# MiniMind

## What MiniMind Provides

- Complete MiniMind-LLM structure code, including Dense and MoE versions.
- A main-branch architecture aligned with the Qwen3 and Qwen3-MoE ecosystem.
- Tokenizer and tokenizer training code, with template tokens such as `<tool_call>`, `<tool_response>`, and `<think>`.
- Full training pipelines: Pretrain, SFT, LoRA, RLHF-DPO, RLAIF with PPO / GRPO / CISPO, Tool Use, Agentic RL, Adaptive Thinking, and Model Distillation.
- Open-source data for multiple stages, including collected, distilled, cleaned, and deduplicated datasets.
- Core training algorithms and core modules implemented from scratch with native PyTorch.
- Compatibility with common frameworks and tools such as transformers, trl, peft, llama.cpp, vllm, ollama, and Llama-Factory.
- Single-machine single-GPU and single-machine multi-GPU training through DDP or DeepSpeed.
- Training visualization with wandb or SwanLab.
- Evaluation on C-Eval, C-MMLU, OpenBookQA, and other benchmarks.
- Long-context RoPE extrapolation through YaRN.
- A minimal OpenAI API-compatible server, with support for `reasoning_content`, `tool_calls`, and `open_thinking`.
- A minimal Streamlit-based chat WebUI.
- Experimental extensions, including a discrete diffusion language model, a linear-attention model, MiniMind-V for vision, and MiniMind-O for multimodal use.

## Released Models

| Model | Parameters | Release Date |
|---|---:|---|
| minimind-3 | 64M | 2026-04-01 |
| minimind-3-moe | 198M-A64M | 2026-04-01 |
| minimind2-small | 26M | 2025-04-26 |
| minimind2-moe | 145M | 2025-04-26 |
| minimind2 | 104M | 2025-04-26 |
| minimind-v1-small | 26M | 2024-08-28 |
| minimind-v1-moe | 4×26M | 2024-09-17 |
| minimind-v1 | 108M | 2024-09-01 |

## Important Changelog

### 2026-04-01

- Released `minimind-3` and `minimind-3-moe`.
- Fully updated the structure, tokenizer, training pipeline, inference interface, and default configuration.
- Aligned the main branch with the Qwen3 / Qwen3-MoE ecosystem.
- Switched the default training data to `pretrain_t2t(_mini).jsonl`, `sft_t2t(_mini).jsonl`, `rlaif.jsonl`, `agent_rl.jsonl`, and `agent_rl_math.jsonl`.
- Removed the standalone `train_reason.py`; thinking is now controlled by `chat_template + <think>` and `open_thinking`.
- Integrated tool-call capability into the main SFT data.
- Added `train_agent.py` for Agentic RL, supporting GRPO / CISPO in multi-turn tool-use scenarios.
- Decoupled the rollout engine in the RLAIF / Agentic RL pipeline.
- Updated the tokenizer to BPE + ByteLevel and added tool-call / thinking tokens.
- Added a LoRA weight merge and export pipeline.

### 2025-10-24

- Added RLAIF algorithms: PPO, GRPO, and SPO, all implemented from scratch.
- Added checkpoint resume support.
- Added RLAIF data and simplified the DPO dataset.
- Added YaRN for long-context RoPE extrapolation.
- Chat templates now fully support Tool Calling and Reasoning tags.
- SwanLab replaces WandB for easier access in China.
- Standardized code and fixed known issues.

### 2025-04-26

- Major update with renamed model parameters for better compatibility with Transformers.
- Refactored generation to inherit from GenerationMixin.
- Added support for llama.cpp, vllm, ollama, and other inference ecosystems.
- Standardized the code and directory structure.
- Changed vocabulary tokens from `<s></s>` to `<|im_start|><|im_end|>`.
- After this update, old models before 2025-04-26 are no longer directly loaded in the old way.

### 2025-02-09 and earlier

- Released the minimind2 series.
- Almost completely refactored the codebase.
- Switched data to JSONL format.
- Improved pretraining data quality, making fast reproduction possible on one 3090 GPU.
- Decoupled LoRA from the peft wrapper and implemented LoRA from scratch.
- Implemented DPO and model distillation with native PyTorch.
- Added the MiniMind-V vision direction.
- The project was first open-sourced on 2024-08-27.

## Quick Start

### Step 0: Installation

```bash
# Clone the repository from the official source, then install dependencies
cd minimind
pip install -r requirements.txt
```

### Model Inference

```bash
# Download the model with ModelScope
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3

# Inference with the Transformers-format model
python eval_llm.py --load_from ./minimind-3

# Inference with a PyTorch model; weights must be in ./out
python eval_llm.py --load_from ./model --weight full_sft
```

### Optional WebUI

```bash
# Requires Python >= 3.10 and Streamlit
pip install streamlit
cp -r minimind-3 ./scripts/minimind-3
cd scripts
streamlit run web_demo.py
```

### Third-party Inference Frameworks

```bash
# Ollama
ollama run QuocBry/minimind-3

# vLLM
vllm serve /path/to/model --served-model-name "minimind"
```

## Model Training

Before CUDA training, check whether PyTorch can detect the GPU:

```python
import torch
print(torch.cuda.is_available())
```

If CUDA is unavailable, CPU or MPS can still be used, but speed and compatibility will differ significantly.

### Required Data

For quick reproduction of the MiniMind Zero dialogue model, the default required files are:

- `pretrain_t2t_mini.jsonl`
- `sft_t2t_mini.jsonl`

Place dataset files in `./dataset`.

### Checkpoint Resume

All training scripts support checkpoint saving. Add `--from_resume 1` to resume automatically:

```bash
python train_pretrain.py --from_resume 1
python train_full_sft.py --from_resume 1
```

Checkpoints are saved in `./checkpoints/`, including the model, optimizer, and training progress. File names follow `<weight_name>_<dimension>_resume.pth`, for example `full_sft_512_resume.pth`.

### Required Pretraining

```bash
cd trainer
python train_pretrain.py
```

Default output: `out/pretrain_*.pth`.

### Required Instruction Fine-tuning / SFT

```bash
cd trainer
python train_full_sft.py
```

Default output: `out/full_sft_*.pth`.

### Testing a Trained Model

```bash
python eval_llm.py --weight full_sft
```

For multiple GPUs, use DDP:

```bash
torchrun --nproc_per_node N train_xxx.py
```

## Data Introduction

### Tokenizer

A tokenizer can be understood as the LLM's dictionary. It maps natural language to token IDs and decodes token IDs back to text. The project includes `train_tokenizer.py` as an example for vocabulary training.

Retraining the tokenizer is not recommended unless necessary, because changing the tokenizer affects model weights, data format, inference interfaces, and ecosystem compatibility. For a small model such as MiniMind, vocabulary size also directly affects the parameter share of the embedding and output layers.

| Tokenizer | Vocabulary Size | Source |
|---|---:|---|
| Yi | 64,000 | 01.AI, China |
| Qwen2 | 151,643 | Alibaba Cloud, China |
| ChatGLM | 151,329 | Zhipu AI, China |
| Mistral | 32,000 | Mistral AI, France |
| Llama 3 | 128,000 | Meta, USA |
| MiniMind | 6,400 | Custom |

The current main branch uniformly uses `minimind_tokenizer` and no longer maintains a `mistral_tokenizer` version.

### Pretrain Data

The main pretraining data for `MiniMind-3` consists of:

- `pretrain_t2t.jsonl`
- `pretrain_t2t_mini.jsonl`

The data is normalized as `text -> next token prediction`, balancing text quality, sample length, Chinese-English ability, and alignment with later SFT / Tool Calling / RLAIF stages.

Example format with Chinese examples translated into English:

```jsonl
{"text": "How can I overcome procrastination? Curing procrastination is not easy, but the following suggestions may help."}
{"text": "Morning sunlight passed through the curtains and entered the room, while the book pages on the desk were gently turned by the wind."}
{"text": "Transformers model contextual relationships through self-attention and are an important foundation of modern large language models."}
```

### SFT Data

The main SFT data consists of:

- `sft_t2t.jsonl`
- `sft_t2t_mini.jsonl`

The current version emphasizes unified templates, multi-turn dialogue, thinking tags, and Tool Calling. SFT data includes instruction-following data, public dialogue data, synthetic model data, and distilled data.

Example format with Chinese examples translated into English:

```jsonl
{
  "conversations": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hello!"},
    {"role": "user", "content": "Goodbye"},
    {"role": "assistant", "content": "Goodbye!"}
  ]
}
```

Tool-use example:

```jsonl
{
  "conversations": [
    {"role": "system", "content": "# Tools ...", "tools": "[...]"},
    {"role": "user", "content": "Translate 'Hello world' into English"},
    {"role": "assistant", "content": "", "tool_calls": "[{\"name\":\"translate_text\",\"arguments\":{\"text\":\"Hello world\",\"target_language\":\"english\"}}]"},
    {"role": "tool", "content": "{\"translated_text\":\"Hello World\"}"},
    {"role": "assistant", "content": "Hello World"}
  ]
}
```

### RL Data

The main RL data is `dpo.jsonl`, used for preference training in RLHF or preference-optimization stages. `chosen` represents the preferred answer, while `rejected` represents the weaker answer.

Example:

```json
{
  "chosen": [
    {"content": "Q", "role": "user"},
    {"content": "good answer", "role": "assistant"}
  ],
  "rejected": [
    {"content": "Q", "role": "user"},
    {"content": "bad answer", "role": "assistant"}
  ]
}
```

### MiniMind Training Dataset

Main dataset files:

```bash
./dataset/
├── agent_rl.jsonl (86MB)
├── agent_rl_math.jsonl (18MB)
├── dpo.jsonl (53MB)
├── pretrain_t2t_mini.jsonl (1.2GB, recommended)
├── pretrain_t2t.jsonl (10GB)
├── rlaif.jsonl (24MB, recommended)
├── sft_t2t_mini.jsonl (1.6GB, recommended)
└── sft_t2t.jsonl (14GB)
```

Recommended use:

- For fastest reproduction: `pretrain_t2t_mini.jsonl` + `sft_t2t_mini.jsonl`.
- For full training: `pretrain_t2t` + `sft_t2t` + `rlaif/agent_rl`.
- Current SFT data already includes Tool Call samples, so a separate Tool Calling SFT pass is usually unnecessary.

`max_seq_len` refers to token length, not character count. The tokenizer averages about 1.5-1.7 Chinese characters per token and 4-5 English characters per token.

## Model

### Structure

`minimind-3` Dense uses a Transformer Decoder-Only architecture aligned with the Qwen3 ecosystem and convenient for conversion to transformers / llama.cpp / ollama / vllm.

Key features:

- Pre-Normalization + RMSNorm.
- SwiGLU activation.
- RoPE rotary positional encoding with YaRN extrapolation support.
- `q_heads=8`, `kv_heads=4`, `max_position_embeddings=32768`, `rope_theta=1e6`.

`minimind-3-moe` extends MoE feed-forward layers on the same structure, compatible with Qwen3-MoE-style configuration and without shared experts. The default configuration is 4 experts / top-1 routing.

| Model Name | Params | Vocab | Max pos | RoPE theta | Layers | d_model | KV heads | Q heads | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| minimind-3 | 64M | 6400 | 32768 | 1e6 | 8 | 768 | 4 | 8 | Dense |
| minimind-3-moe | 198M-A64M | 6400 | 32768 | 1e6 | 8 | 768 | 4 | 8 | 4 experts / top-1 |
| minimind2-small | 26M | 6400 | 32768 | 1e6 | 8 | 512 | 2 | 8 | Historical |
| minimind2-moe | 145M | 6400 | 32768 | 1e6 | 8 | 640 | 2 | 8 | Historical |
| minimind2 | 104M | 6400 | 32768 | 1e6 | 16 | 768 | 2 | 8 | Historical |

### Model Configuration

For models around 100M parameters, the choice between `d_model` and `n_layers` strongly affects training stability and final performance. The main branch of `minimind-3` uses `dim=768, n_layers=8` as an engineering trade-off: the network is shallow enough to train quickly, while the hidden dimension is not so small that representation quality collapses.

## Estimated Training Cost

Time is measured in hours, and cost is measured in CNY. The estimate assumes one NVIDIA 3090 with a rental price of about 1.3 CNY per hour.

| Model | Params | pretrain_t2t_mini | sft_t2t_mini | toolcall | RLAIF |
|---|---:|---:|---:|---:|---:|
| minimind-3 | 64M | ≈1.21h / ≈1.57 CNY | ≈1.10h / ≈1.43 CNY | ≈0.9h / ≈1.17 CNY | ≈1.1h / ≈1.43 CNY |
| minimind-3-moe | 198M-A64M | ≈1.69h / ≈2.20 CNY | ≈1.54h / ≈2.00 CNY | ≈1.26h / ≈1.64 CNY | ≈1.54h / ≈2.00 CNY |

With `pretrain_t2t_mini` + `sft_t2t_mini`, `minimind-3` can train a Zero dialogue model from scratch in about 2.31 hours, costing about 3 CNY on one 3090.

## Main Training Stages

### Pretraining

Pretraining helps the model learn foundational knowledge, language patterns, and statistical relationships in text. Its core objective is high-quality next-token prediction.

```bash
cd trainer
torchrun --nproc_per_node 1 train_pretrain.py
# or
python train_pretrain.py
```

Example test prompts translated into English:

```text
User: Why is the sky blue?
Assistant: The sky appears blue mainly because, after sunlight enters the atmosphere, short-wavelength blue light is more easily scattered by air molecules, so human eyes receive more blue light from all directions.

User: Explain what machine learning is.
Assistant: Machine learning is an important branch of artificial intelligence. It trains models with data so that systems can automatically learn patterns and continuously improve classification, prediction, recommendation, natural language processing, and other tasks.
```

### Supervised Fine-Tuning / SFT

SFT adapts the model to multi-turn dialogue, the `user / assistant / system / tool` role structure, instruction following, tool calling, and thinking tags.

```bash
cd trainer
torchrun --nproc_per_node 1 train_full_sft.py
# or
python train_full_sft.py
```

Example test prompts translated into English:

```text
User: Explain what machine learning is.
Assistant: Machine learning is one of the core technologies of artificial intelligence. It allows computers to learn patterns from data through algorithms and continuously improve predictions or decisions. Common applications include recommendation systems, image recognition, speech recognition, and natural language processing.

User: Recommend some Chinese foods.
Assistant: Examples include Peking duck, Lanzhou beef noodles, Sichuan hot pot, Cantonese dim sum, soup dumplings, and mapo tofu. These foods represent regional flavor styles and are good entry points for understanding Chinese food culture.
```

### Knowledge Distillation

MiniMind supports:

- Black-box distillation: learning from teacher-model answers.
- White-box distillation: also learning the teacher model's token probability distribution.

Reference script:

```bash
cd trainer
python train_distillation.py
```

### LoRA

LoRA is a parameter-efficient fine-tuning method. It updates only a small number of newly added parameters while keeping the main model weights unchanged. It is useful for adapting a model to private or vertical domains such as medicine, law, education, or enterprise data.

```bash
cd trainer
python train_lora.py
```

Example domain data translated into English:

```jsonl
{"conversations": [{"role": "user", "content": "What pillow height is best for someone with cervical spondylosis?"}, {"role": "assistant", "content": "Patients with cervical spondylosis should choose pillow height based on..."}]}
{"conversations": [{"role": "user", "content": "What is your name?"}, {"role": "assistant", "content": "Hello, my name is MiniMind, an AI assistant developed by QuocBry. I can help with translation, recommendations, and other tasks."}]}
```

Run inference with base weights plus LoRA weights:

```bash
python eval_llm.py --weight full_sft --lora_weight lora_medical
```

### Tool Calling and Adaptive Thinking

MiniMind supports templates with `tool_call`, `tool_response`, and `think`. Main-branch SFT data already contains Tool Call data, and `open_thinking` allows thinking-style output to be enabled or disabled when appropriate.

### RLHF / RLAIF / Agentic RL

- RLHF-DPO: uses preference pairs between better and worse answers.
- RLAIF: uses AI feedback to optimize policies, supporting PPO, GRPO, and CISPO.
- Agentic RL: trains the model in multi-turn tool-use environments, especially useful for tasks that require verification or environment interaction.

Main scripts:

```bash
python train_dpo.py
python train_ppo.py
python train_grpo.py
python train_agent.py
```

## Open-Sourced Training Results

The project provides two groups of weights:

- Native PyTorch weights.
- Transformers-format weights.

PyTorch naming conventions:

- Dense:
  - `pretrain_{hidden_size}.pth`
  - `full_sft_{hidden_size}.pth`
  - `dpo_{hidden_size}.pth`
  - `ppo_actor_{hidden_size}.pth`
  - `grpo_{hidden_size}.pth`
  - `agent_{hidden_size}.pth`
  - `lora_xxx_{hidden_size}.pth`
- MoE:
  - Add `_moe`, for example `pretrain_{hidden_size}_moe.pth`.

## Evaluation

The project includes subjective and objective evaluations:

- Comparisons among RL models on subjective Q&A, light agent tasks, and answer quality.
- Comparisons between MiniMind and other small models.
- Scores for accuracy, completeness, logic, and code quality.
- Objective evaluation on C-Eval, C-MMLU, ARC Easy, PIQA, OpenBookQA, HellaSwag, and Social IQA.

MiniMind can reach basic dialogue quality at low cost, but very small models still have clear limitations in factual knowledge, stability, generalization, and programming ability.

## RoPE and YaRN for Long Context

MiniMind supports RoPE length extrapolation through YaRN. With the native torch model, enable it with:

```bash
python eval_llm.py --weight full_sft --inference_rope_scaling
```

For Transformers-format models, add `rope_scaling` configuration to `config.json`.

## Conversion and Deployment

### Model Conversion

`scripts/convert_model.py` converts between torch and transformers formats. If using native torch weights, convert them to transformers format before using common inference tools.

### OpenAI-compatible API

`serve_openai_api.py` provides a lightweight OpenAI API-compatible chat service and supports `reasoning_content`, `tool_calls`, and `open_thinking`.

```bash
cd scripts
python serve_openai_api.py
```

Test the API:

```bash
cd scripts
python chat_api.py
```

API payload example without URL:

```json
{
  "model": "model-identifier",
  "messages": [
    {"role": "user", "content": "What is the highest mountain in the world?"}
  ],
  "temperature": 0.7,
  "max_tokens": 1024,
  "stream": true,
  "open_thinking": true
}
```

### SGLang

SGLang is a high-performance inference engine supporting RadixAttention and continuous batching. It requires a CUDA environment.

```bash
python -m sglang.launch_server --model-path /path/to/model --attention-backend triton --host 0.0.0.0 --port 8998
```

### vLLM

vLLM is suitable for quick deployment with good VRAM utilization and throughput.

```bash
vllm serve /path/to/model --model-impl transformers --served-model-name "minimind" --port 8998
```

### llama.cpp

llama.cpp is a lightweight C++ inference framework that can run from the command line, supports multithreaded inference, and offers some GPU acceleration options.

General workflow:

1. Install llama.cpp according to its official documentation.
2. Add MiniMind tokenizer support in `convert_hf_to_gguf.py` if needed.
3. Convert the HuggingFace-format MiniMind model to GGUF.
4. Optionally quantize the model.
5. Run inference with `llama-cli`.

### Ollama

Ollama makes it easy to run local models. Create a `minimind.modelfile`, point it to a GGUF file, define a system prompt, template, and stop parameters. Then run:

```bash
ollama create -f minimind.modelfile minimind-local
ollama run minimind-local
```

You can also quickly run the provided model:

```bash
ollama run QuocBry/minimind-3
```

Example system prompt translated into English:

```text
Your name is MiniMind. You are a helpful and knowledgeable AI assistant. Please answer users in a complete and friendly way. When asked your name, answer MiniMind.
```

### MNN

MNN is an inference engine for edge devices, supporting lightweight deployment and high-performance inference.

```bash
cd MNN/transformers/llm/export
python llmexport.py --path /path/to/model/ --export mnn --hqq --dst_path model-mnn
./llm_demo /path/to/model-mnn/config.json prompt.txt
```

## Acknowledgments and Contributions

The project thanks contributors who helped with training records, data processing, tutorial organization, and code breakdowns. Related projects and research such as Llama, llama2.c, TinyLlama, DeepSeek, ChatLM-mini-Chinese, Mistral-MoE, and many other small-LLM projects inspired MiniMind.

## MiniMind Related Achievements

MiniMind has been used or referenced in research and materials related to medical benchmarks, expert load balancing in MoE, evaluation of LLM-generated legal text, generalization in next-token prediction, federated learning under device heterogeneity, vessel trajectory prediction, AI for cryptography, and resilience to hardware transient faults.

The Chinese book title in the original file is translated as:

- Writing Large Models from Scratch: From Neural Networks to Transformers, by Wang Shuang, Mou Chen, and Wang Haoyi, Tsinghua University Press.

## Citation

```bibtex
@misc{minimind,
  title = {MiniMind: Train a Tiny LLM from Scratch},
  author = {QuocBry},
  year = {2026},
  note = {GitHub repository, accessed 2026}
}
```

## License

This project is released under the Apache License 2.0.
