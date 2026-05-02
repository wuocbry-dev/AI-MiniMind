import time
import argparse
import random
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer

def main():
    parser = argparse.ArgumentParser(description="Suy luận và hội thoại MiniMind")
    parser.add_argument('--load_from', default='model', type=str, help="Đường dẫn tải mô hình (model=trọng số torch gốc, đường dẫn khác=định dạng transformers)")
    parser.add_argument('--save_dir', default='out', type=str, help="Thư mục trọng số mô hình")
    parser.add_argument('--weight', default='full_sft', type=str, help="Tiền tố tên trọng số (pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo)")
    parser.add_argument('--lora_weight', default='None', type=str, help="Tên trọng số LoRA (None=không dùng, có thể: lora_identity, lora_medical)")
    parser.add_argument('--hidden_size', default=768, type=int, help="Kích thước tầng ẩn")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="Số tầng ẩn")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="Có dùng kiến trúc MoE hay không (0=không, 1=có)")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="Bật ngoại suy mã hóa vị trí RoPE (4x, chỉ giải quyết vấn đề mã hóa vị trí)")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="Độ dài sinh tối đa (lưu ý: không phải năng lực văn bản dài thực tế)")
    parser.add_argument('--temperature', default=0.85, type=float, help="Nhiệt độ sinh, kiểm soát tính ngẫu nhiên (0-1, lớn hơn thì ngẫu nhiên hơn)")
    parser.add_argument('--top_p', default=0.95, type=float, help="Ngưỡng nucleus sampling (0-1)")
    parser.add_argument('--open_thinking', default=0, type=int, help="Có bật tư duy thích ứng hay không (0=không, 1=có)")
    parser.add_argument('--historys', default=0, type=int, help="Số lượt hội thoại mang theo (phải là số chẵn, 0 là không mang theo)")
    parser.add_argument('--show_speed', default=1, type=int, help="Hiện tốc độ decode (tokens/s)")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="Thiết bị chạy")
    args = parser.parse_args()
    
    prompts = [
        'Bạn có thế mạnh gì?',
        'Vì sao bầu trời màu xanh?',
        'Hãy viết hàm Python tính dãy Fibonacci.',
        'Giải thích quy trình cơ bản của "quang hợp".',
        'Nếu mai trời mưa, tôi nên ra ngoài thế nào?',
        'So sánh ưu/nhược điểm của mèo và chó khi làm thú cưng.',
        'Giải thích máy học là gì.',
        'Gợi ý một số món ăn Trung Quốc.'
    ]
    
    conversation = []
    model, tokenizer = init_model(args)
    input_mode = int(input('[0] Tự động kiểm thử\n[1] Nhập thủ công\n'))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    for prompt in prompt_iter:
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0: print(f'💬: {prompt}')
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})
        if 'pretrain' in args.weight:
            inputs = tokenizer.bos_token + prompt
        else:
            inputs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))
        
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print('🧠: ', end='')
        st = time.time()
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1
        )
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[Tốc độ]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')

if __name__ == "__main__":
    main()