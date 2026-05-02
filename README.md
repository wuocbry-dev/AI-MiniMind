# MiniMind- Bản dịch tiếng việt của QuocBry
# Tác giả chính:Jingyao Gong 2024
# Tác giả Việt hóa và bổ sung:QuocBry 2026

## MiniMind cung cấp những gì?

- Mã cấu trúc MiniMind-LLM đầy đủ, bao gồm Dense và MoE.
- Cấu trúc nhánh chính tương thích với hệ sinh thái Qwen3 và Qwen3-MoE.
- Mã huấn luyện tokenizer và bộ tách từ, hỗ trợ các token mẫu như `<tool_call>`, `<tool_response>`, `<think>`.
- Quy trình huấn luyện đầy đủ: Pretrain, SFT, LoRA, RLHF-DPO, RLAIF với PPO / GRPO / CISPO, Tool Use, Agentic RL, Adaptive Thinking và Model Distillation.
- Dữ liệu mã nguồn mở cho nhiều giai đoạn, gồm dữ liệu thu thập, chưng cất, làm sạch và khử trùng lặp.
- Các thuật toán huấn luyện và mô-đun lõi được triển khai từ đầu bằng PyTorch gốc.
- Tương thích với các framework và công cụ phổ biến như transformers, trl, peft, llama.cpp, vllm, ollama và Llama-Factory.
- Hỗ trợ huấn luyện một máy một GPU và một máy nhiều GPU với DDP hoặc DeepSpeed.
- Hỗ trợ ghi nhận/hiển thị quá trình huấn luyện bằng wandb hoặc SwanLab.
- Hỗ trợ đánh giá trên C-Eval, C-MMLU, OpenBookQA và các bộ benchmark khác.
- Hỗ trợ mở rộng ngữ cảnh dài bằng YaRN cho RoPE.
- Cung cấp máy chủ API tối giản tương thích giao thức OpenAI API, hỗ trợ `reasoning_content`, `tool_calls`, `open_thinking`.
- Cung cấp WebUI chat tối giản dựa trên Streamlit.
- Có các hướng mở rộng thử nghiệm: mô hình ngôn ngữ khuếch tán rời rạc, mô hình attention tuyến tính, mô hình thị giác MiniMind-V và mô hình đa phương thức MiniMind-O.

## Danh sách mô hình đã phát hành

| Mô hình | Tham số | Ngày phát hành |
|---|---:|---|
| minimind-3 | 64M | 2026-04-01 |
| minimind-3-moe | 198M-A64M | 2026-04-01 |
| minimind2-small | 26M | 2025-04-26 |
| minimind2-moe | 145M | 2025-04-26 |
| minimind2 | 104M | 2025-04-26 |
| minimind-v1-small | 26M | 2024-08-28 |
| minimind-v1-moe | 4×26M | 2024-09-17 |
| minimind-v1 | 108M | 2024-09-01 |

## Nhật ký cập nhật quan trọng

### 2026-04-01

- Phát hành `minimind-3` và `minimind-3-moe`.
- Cập nhật toàn diện cấu trúc, tokenizer, pipeline huấn luyện, giao diện suy luận và cấu hình mặc định.
- Cấu trúc nhánh chính đồng bộ với hệ sinh thái Qwen3 / Qwen3-MoE.
- Dữ liệu mặc định chuyển sang `pretrain_t2t(_mini).jsonl`, `sft_t2t(_mini).jsonl`, `rlaif.jsonl`, `agent_rl.jsonl`, `agent_rl_math.jsonl`.
- Loại bỏ script `train_reason.py`; năng lực suy nghĩ được điều khiển qua `chat_template + <think>` và công tắc `open_thinking`.
- Tích hợp năng lực tool call vào dữ liệu SFT nhánh chính.
- Thêm script `train_agent.py` cho Agentic RL, hỗ trợ GRPO / CISPO trong kịch bản dùng công cụ nhiều lượt.
- Tách rollout engine trong pipeline RLAIF / Agentic RL.
- Cập nhật tokenizer theo BPE + ByteLevel và bổ sung token dành cho tool call / thinking.
- Thêm quy trình gộp và xuất trọng số LoRA.

### 2025-10-24

- Thêm thuật toán RLAIF: PPO, GRPO, SPO, đều được triển khai từ đầu.
- Thêm chức năng tiếp tục huấn luyện từ checkpoint.
- Thêm dữ liệu RLAIF và đơn giản hóa dữ liệu DPO.
- Thêm YaRN để mở rộng ngữ cảnh dài cho RoPE.
- Chat template hỗ trợ đầy đủ tag Tool Calling và Reasoning.
- SwanLab được dùng thay cho WandB trong môi trường mạng Trung Quốc.
- Chuẩn hóa mã và sửa lỗi.

### 2025-04-26

- Cập nhật lớn, đổi tên tham số mô hình để tương thích tốt hơn với Transformers.
- Tái cấu trúc phương thức generate, kế thừa GenerationMixin.
- Hỗ trợ llama.cpp, vllm, ollama và nhiều hệ sinh thái suy luận khác.
- Chuẩn hóa mã và cấu trúc thư mục.
- Thay token `<s></s>` bằng `<|im_start|><|im_end|>`.
- Sau cập nhật này, các mô hình cũ trước 2025-04-26 không còn được tải trực tiếp theo cách cũ.

### 2025-02-09 và các mốc cũ hơn

- Ra mắt dòng minimind2.
- Tái cấu trúc gần như toàn bộ mã.
- Chuyển dữ liệu sang định dạng JSONL.
- Cải thiện chất lượng dữ liệu pretrain, giúp có thể tái tạo nhanh trên một GPU 3090.
- Tách LoRA khỏi wrapper peft, triển khai LoRA từ đầu.
- Triển khai DPO và distillation bằng PyTorch gốc.
- Bổ sung hướng thị giác MiniMind-V.
- Dự án được mở nguồn lần đầu vào 2024-08-27.

## Bắt đầu nhanh


### Bước 0: cài đặt

```bash
# Sao chép kho mã từ nguồn chính thức rồi cài đặt phụ thuộc
cd minimind
pip install -r requirements.txt
```

### Suy luận mô hình

```bash
# Tải mô hình bằng ModelScope
modelscope download --model QuocBry/minimind-3 --local_dir ./minimind-3

# Suy luận với mô hình định dạng Transformers
python eval_llm.py --load_from ./minimind-3

# Suy luận với mô hình PyTorch, cần có trọng số trong ./out
python eval_llm.py --load_from ./model --weight full_sft
```

### WebUI tùy chọn

```bash
# Cần Python >= 3.10 và Streamlit
pip install streamlit
cp -r minimind-3 ./scripts/minimind-3
cd scripts
streamlit run web_demo.py
```

### Framework suy luận bên thứ ba

```bash
# Ollama
ollama run QuocBry/minimind-3

# vLLM
vllm serve /path/to/model --served-model-name "minimind"
```

## Huấn luyện mô hình

Trước khi huấn luyện bằng CUDA, nên kiểm tra PyTorch có nhận GPU hay không:

```python
import torch
print(torch.cuda.is_available())
```

Nếu CUDA không khả dụng, vẫn có thể chạy bằng CPU hoặc MPS, nhưng tốc độ và độ tương thích sẽ khác rất nhiều.

### Dữ liệu cần tải

Để tái tạo nhanh mô hình đối thoại MiniMind Zero, mặc định chỉ cần:

- `pretrain_t2t_mini.jsonl`
- `sft_t2t_mini.jsonl`

Các file dữ liệu nên đặt trong thư mục `./dataset`.

### Tiếp tục huấn luyện từ checkpoint

Tất cả script huấn luyện đều hỗ trợ lưu checkpoint. Thêm tham số `--from_resume 1` để tự động khôi phục tiến độ:

```bash
python train_pretrain.py --from_resume 1
python train_full_sft.py --from_resume 1
```

Checkpoint được lưu trong `./checkpoints/`, gồm mô hình, optimizer và tiến độ huấn luyện. Tên file có dạng `<tên_trọng_số>_<dimension>_resume.pth`, ví dụ `full_sft_512_resume.pth`.

### Pretraining bắt buộc

```bash
cd trainer
python train_pretrain.py
```

Kết quả mặc định: `out/pretrain_*.pth`.

### Instruction fine-tuning / SFT bắt buộc

```bash
cd trainer
python train_full_sft.py
```

Kết quả mặc định: `out/full_sft_*.pth`.

### Kiểm tra mô hình đã huấn luyện

```bash
python eval_llm.py --weight full_sft
```

Nếu có nhiều GPU, có thể dùng DDP:

```bash
torchrun --nproc_per_node N train_xxx.py
```

## Giới thiệu dữ liệu

### Tokenizer

Tokenizer có thể hiểu như “từ điển” của LLM, chuyển văn bản tự nhiên thành token id và giải mã token id trở lại văn bản. Dự án có `train_tokenizer.py` làm ví dụ huấn luyện từ vựng.

Không nên huấn luyện lại tokenizer nếu không thật sự cần, vì thay đổi tokenizer sẽ ảnh hưởng đến trọng số mô hình, định dạng dữ liệu, giao diện suy luận và khả năng tương thích với hệ sinh thái. Với mô hình nhỏ như MiniMind, kích thước từ vựng cũng ảnh hưởng trực tiếp đến tỷ lệ tham số của embedding và output layer.

| Tokenizer | Kích thước từ vựng | Nguồn |
|---|---:|---|
| Yi | 64.000 | 01.AI, Trung Quốc |
| Qwen2 | 151.643 | Alibaba Cloud, Trung Quốc |
| ChatGLM | 151.329 | Zhipu AI, Trung Quốc |
| Mistral | 32.000 | Mistral AI, Pháp |
| Llama 3 | 128.000 | Meta, Hoa Kỳ |
| MiniMind | 6.400 | Tự định nghĩa |

Nhánh chính hiện dùng thống nhất `minimind_tokenizer` và không còn duy trì phiên bản `mistral_tokenizer`.

### Dữ liệu Pretrain

Dữ liệu pretrain chính của `MiniMind-3` gồm:

- `pretrain_t2t.jsonl`
- `pretrain_t2t_mini.jsonl`

Dữ liệu được chuẩn hóa theo dạng `text -> next token prediction`, nhằm cân bằng chất lượng văn bản, độ dài mẫu, khả năng song ngữ Trung-Anh và khả năng nối tiếp với SFT / Tool Calling / RLAIF.

Ví dụ định dạng:

```jsonl
{"text": "Làm thế nào để vượt qua sự trì hoãn? Việc chữa trị thói quen trì hoãn không dễ, nhưng một số gợi ý sau có thể hữu ích."}
{"text": "Ánh nắng buổi sáng xuyên qua rèm cửa và chiếu vào căn phòng, những trang sách trên bàn khẽ lay động theo gió."}
{"text": "Transformer sử dụng cơ chế tự chú ý để mô hình hóa quan hệ ngữ cảnh, là một cấu trúc nền tảng của các mô hình ngôn ngữ lớn hiện đại."}
```

### Dữ liệu SFT

Dữ liệu SFT chính gồm:

- `sft_t2t.jsonl`
- `sft_t2t_mini.jsonl`

Dữ liệu nhấn mạnh template thống nhất, đối thoại nhiều lượt, tag suy nghĩ và Tool Calling. Dữ liệu SFT bao gồm dữ liệu làm theo chỉ dẫn, hội thoại công khai, dữ liệu tổng hợp bằng mô hình và dữ liệu chưng cất.

Ví dụ định dạng:

```jsonl
{
  "conversations": [
    {"role": "user", "content": "Xin chào"},
    {"role": "assistant", "content": "Xin chào!"},
    {"role": "user", "content": "Tạm biệt"},
    {"role": "assistant", "content": "Tạm biệt!"}
  ]
}
```

### Dữ liệu RL

Dữ liệu RL chính gồm `dpo.jsonl`, dùng cho huấn luyện ưu tiên trong RLHF hoặc các giai đoạn tối ưu hóa dựa trên phản hồi. Trong đó `chosen` là câu trả lời được ưu tiên, `rejected` là câu trả lời kém hơn.

Ví dụ:

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

### Bộ dữ liệu huấn luyện MiniMind

Các file dữ liệu chính:

```bash
./dataset/
├── agent_rl.jsonl (86MB)
├── agent_rl_math.jsonl (18MB)
├── dpo.jsonl (53MB)
├── pretrain_t2t_mini.jsonl (1.2GB, khuyến nghị)
├── pretrain_t2t.jsonl (10GB)
├── rlaif.jsonl (24MB, khuyến nghị)
├── sft_t2t_mini.jsonl (1.6GB, khuyến nghị)
└── sft_t2t.jsonl (14GB)
```

Gợi ý sử dụng:

- Muốn tái tạo nhanh: dùng `pretrain_t2t_mini.jsonl` + `sft_t2t_mini.jsonl`.
- Muốn huấn luyện đầy đủ hơn: dùng `pretrain_t2t` + `sft_t2t` + `rlaif/agent_rl`.
- Dữ liệu SFT hiện đã trộn sẵn mẫu Tool Call nên thường không cần một vòng fine-tuning Tool Calling riêng.

`max_seq_len` tính theo token, không phải số ký tự. Tokenizer của dự án có tỷ lệ khoảng 1,5-1,7 ký tự/token với văn bản tiếng Trung và 4-5 ký tự/token với tiếng Anh.

## Mô hình

### Cấu trúc

`minimind-3` Dense dùng kiến trúc Transformer Decoder-Only, tương thích với hệ sinh thái Qwen3 và thuận tiện chuyển sang transformers / llama.cpp / ollama / vllm.

Đặc điểm chính:

- Pre-Normalization + RMSNorm.
- Hàm kích hoạt SwiGLU.
- RoPE rotary positional encoding, hỗ trợ ngoại suy bằng YaRN.
- `q_heads=8`, `kv_heads=4`, `max_position_embeddings=32768`, `rope_theta=1e6`.

`minimind-3-moe` mở rộng lớp feed-forward MoE trên cùng cấu trúc, tương thích phong cách cấu hình Qwen3-MoE và loại bỏ shared expert. Cấu hình mặc định: 4 experts / top-1 routing.

| Tên mô hình | Tham số | Vocab | Max pos | RoPE theta | Layers | d_model | KV heads | Q heads | Ghi chú |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| minimind-3 | 64M | 6400 | 32768 | 1e6 | 8 | 768 | 4 | 8 | Dense |
| minimind-3-moe | 198M-A64M | 6400 | 32768 | 1e6 | 8 | 768 | 4 | 8 | 4 experts / top-1 |
| minimind2-small | 26M | 6400 | 32768 | 1e6 | 8 | 512 | 2 | 8 | Phiên bản cũ |
| minimind2-moe | 145M | 6400 | 32768 | 1e6 | 8 | 640 | 2 | 8 | Phiên bản cũ |
| minimind2 | 104M | 6400 | 32768 | 1e6 | 16 | 768 | 2 | 8 | Phiên bản cũ |

### Cấu hình mô hình

Với mô hình khoảng 100M tham số, lựa chọn giữa `d_model` và `n_layers` ảnh hưởng lớn đến ổn định huấn luyện và hiệu năng. Nhánh chính `minimind-3` chọn `dim=768, n_layers=8` như một thỏa hiệp kỹ thuật: mạng không quá sâu để huấn luyện nhanh, nhưng chiều ẩn không quá nhỏ để tránh suy giảm biểu diễn.

## Chi phí huấn luyện ước tính

Đơn vị thời gian là giờ, chi phí là CNY. Ước tính dựa trên một GPU NVIDIA 3090 với giá thuê khoảng 1,3 CNY/giờ.

| Mô hình | Tham số | pretrain_t2t_mini | sft_t2t_mini | toolcall | RLAIF |
|---|---:|---:|---:|---:|---:|
| minimind-3 | 64M | ≈1,21h / ≈1,57 CNY | ≈1,10h / ≈1,43 CNY | ≈0,9h / ≈1,17 CNY | ≈1,1h / ≈1,43 CNY |
| minimind-3-moe | 198M-A64M | ≈1,69h / ≈2,20 CNY | ≈1,54h / ≈2,00 CNY | ≈1,26h / ≈1,64 CNY | ≈1,54h / ≈2,00 CNY |

Với `pretrain_t2t_mini` + `sft_t2t_mini`, `minimind-3` có thể huấn luyện một mô hình đối thoại Zero từ đầu trong khoảng 2,31 giờ, chi phí khoảng 3 CNY trên một GPU 3090.

## Các giai đoạn huấn luyện chính

### Pretraining

Pretraining giúp mô hình học kiến thức nền tảng, mẫu ngôn ngữ và quan hệ thống kê trong văn bản. Mục tiêu cốt lõi là học cách hoàn thành chuỗi từ chất lượng cao.

```bash
cd trainer
torchrun --nproc_per_node 1 train_pretrain.py
# hoặc
python train_pretrain.py
```

### Supervised Fine-Tuning / SFT

SFT giúp mô hình thích nghi với hội thoại nhiều lượt, cấu trúc vai trò `user / assistant / system / tool`, làm theo chỉ dẫn, gọi công cụ và thẻ suy nghĩ.

```bash
cd trainer
torchrun --nproc_per_node 1 train_full_sft.py
# hoặc
python train_full_sft.py
```

### Knowledge Distillation

Dự án hỗ trợ cả:

- Black-box distillation: học từ câu trả lời của mô hình giáo viên.
- White-box distillation: học thêm phân phối xác suất token của mô hình giáo viên.

Script tham khảo:

```bash
cd trainer
python train_distillation.py
```

### LoRA

LoRA là phương pháp fine-tuning tiết kiệm tham số, chỉ cập nhật một lượng nhỏ tham số mới trong khi giữ nguyên phần lớn trọng số mô hình gốc. Phù hợp với thích nghi miền riêng như y tế, pháp lý, giáo dục hoặc dữ liệu doanh nghiệp.

```bash
cd trainer
python train_lora.py
```

Khi suy luận có thể kết hợp base model và trọng số LoRA:

```bash
python eval_llm.py --weight full_sft --lora_weight lora_medical
```

### Tool Calling và Adaptive Thinking

MiniMind hỗ trợ mẫu hội thoại có `tool_call`, `tool_response` và `think`. SFT nhánh chính đã trộn dữ liệu Tool Call, còn `open_thinking` cho phép bật/tắt cách mô hình sinh nội dung suy nghĩ trong ngữ cảnh phù hợp.

### RLHF / RLAIF / Agentic RL

- RLHF-DPO: dùng dữ liệu ưu tiên giữa câu trả lời tốt và câu trả lời kém.
- RLAIF: dùng phản hồi từ AI để tối ưu chính sách, hỗ trợ PPO, GRPO, CISPO.
- Agentic RL: huấn luyện mô hình trong kịch bản nhiều lượt có gọi công cụ, đặc biệt phù hợp với bài toán cần kiểm tra kết quả hoặc tương tác với môi trường.

Các script chính:

```bash
python train_dpo.py
python train_ppo.py
python train_grpo.py
python train_agent.py
```

## Kết quả huấn luyện mở nguồn

Dự án cung cấp hai nhóm trọng số:

- Trọng số PyTorch gốc.
- Trọng số định dạng Transformers.

Quy ước đặt tên trọng số PyTorch:

- Dense:
  - `pretrain_{hidden_size}.pth`
  - `full_sft_{hidden_size}.pth`
  - `dpo_{hidden_size}.pth`
  - `ppo_actor_{hidden_size}.pth`
  - `grpo_{hidden_size}.pth`
  - `agent_{hidden_size}.pth`
  - `lora_xxx_{hidden_size}.pth`
- MoE:
  - Thêm hậu tố `_moe`, ví dụ `pretrain_{hidden_size}_moe.pth`.

## Đánh giá

Dự án có đánh giá chủ quan và khách quan:

- So sánh các mô hình sau RL theo câu hỏi chủ quan, task agent nhẹ và khả năng trả lời.
- So sánh MiniMind với một số mô hình nhỏ khác.
- Đánh giá độ chính xác, độ đầy đủ, logic và chất lượng code.
- Hỗ trợ đánh giá khách quan bằng các bộ như C-Eval, C-MMLU, ARC Easy, PIQA, OpenBookQA, HellaSwag, Social IQA.

MiniMind có thể đạt chất lượng hội thoại cơ bản với chi phí thấp, nhưng các mô hình rất nhỏ vẫn có hạn chế rõ rệt về tri thức thực tế, tính ổn định, khả năng tổng quát và lập trình.

## RoPE và YaRN cho ngữ cảnh dài

MiniMind hỗ trợ mở rộng độ dài ngữ cảnh của RoPE bằng YaRN. Với mô hình torch gốc, thêm tham số:

```bash
python eval_llm.py --weight full_sft --inference_rope_scaling
```

Với định dạng Transformers, có thể thêm cấu hình `rope_scaling` vào `config.json`.

## Chuyển đổi và triển khai

### Chuyển đổi mô hình

Script `scripts/convert_model.py` dùng để chuyển đổi giữa định dạng torch và transformers. Nếu dùng trọng số torch gốc, nên chuyển sang transformers trước khi dùng các công cụ suy luận phổ biến.

### API tương thích OpenAI

`serve_openai_api.py` cung cấp dịch vụ chat nhẹ tương thích OpenAI API, hỗ trợ các trường `reasoning_content`, `tool_calls`, `open_thinking`.

```bash
cd scripts
python serve_openai_api.py
```

Kiểm tra API:

```bash
cd scripts
python chat_api.py
```

Ví dụ payload API không kèm URL:

```json
{
  "model": "model-identifier",
  "messages": [
    {"role": "user", "content": "Ngọn núi cao nhất thế giới là gì?"}
  ],
  "temperature": 0.7,
  "max_tokens": 1024,
  "stream": true,
  "open_thinking": true
}
```

### SGLang

SGLang là engine suy luận hiệu năng cao, hỗ trợ RadixAttention và continuous batching. Cần môi trường CUDA.

```bash
python -m sglang.launch_server --model-path /path/to/model --attention-backend triton --host 0.0.0.0 --port 8998
```

### vLLM

vLLM phù hợp để triển khai nhanh mô hình lớn với khả năng sử dụng VRAM và throughput tốt.

```bash
vllm serve /path/to/model --model-impl transformers --served-model-name "minimind" --port 8998
```

### llama.cpp

llama.cpp là framework C++ nhẹ, có thể dùng dòng lệnh, hỗ trợ suy luận đa luồng và một số tùy chọn tăng tốc GPU.

Quy trình tổng quát:

1. Cài llama.cpp theo tài liệu chính thức.
2. Bổ sung hỗ trợ tokenizer MiniMind trong `convert_hf_to_gguf.py` nếu cần.
3. Chuyển mô hình HuggingFace sang GGUF.
4. Có thể lượng tử hóa sang Q8 hoặc định dạng khác.
5. Chạy suy luận bằng `llama-cli`.

### Ollama

Ollama giúp chạy mô hình cục bộ đơn giản. Có thể tạo `minimind.modelfile`, trỏ đến file GGUF, khai báo system prompt, template và tham số dừng. Sau đó chạy:

```bash
ollama create -f minimind.modelfile minimind-local
ollama run minimind-local
```

Có thể dùng nhanh mô hình đã cung cấp:

```bash
ollama run QuocBry/minimind-3
```

### MNN

MNN là engine suy luận cho thiết bị biên, hỗ trợ triển khai nhẹ và hiệu năng cao.

```bash
cd MNN/transformers/llm/export
python llmexport.py --path /path/to/model/ --export mnn --hqq --dst_path model-mnn
./llm_demo /path/to/model-mnn/config.json prompt.txt
```

## Ghi nhận và đóng góp

Dự án cảm ơn các cộng tác viên đã hỗ trợ ghi chép huấn luyện, xử lý dữ liệu, viết hướng dẫn và phân tích mã. Các dự án và nghiên cứu liên quan như Llama, llama2.c, TinyLlama, DeepSeek, ChatLM-mini-Chinese, Mistral-MoE và nhiều dự án LLM nhỏ khác đã tạo cảm hứng cho MiniMind.

## Thành quả liên quan

MiniMind đã được dùng hoặc nhắc tới trong một số nghiên cứu và tài liệu liên quan đến benchmark y tế, cân bằng tải chuyên gia trong MoE, đánh giá văn bản pháp lý do LLM sinh ra, khả năng tổng quát hóa của next-token prediction, học liên kết trên thiết bị không đồng nhất, dự đoán quỹ đạo tàu, AI trong mật mã học và khả năng chịu lỗi phần cứng.

## Trích dẫn

```bibtex
@misc{minimind,
  title = {MiniMind: Train a Tiny LLM from Scratch},
  author = {Jingyao Gong},
  year = {2024},
  note = {GitHub repository, accessed 2026}
}
```

## Giấy phép

Dự án sử dụng giấy phép Apache License 2.0.
