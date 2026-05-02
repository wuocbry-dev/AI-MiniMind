import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import json
import time
import random
import argparse
import warnings
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from openai import OpenAI
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "Tính giá trị biểu thức toán học, hỗ trợ cộng trừ nhân chia, lũy thừa, căn bậc hai, ...", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "Biểu thức toán học, ví dụ 123+456, 2**10, sqrt(144)"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "Lấy ngày giờ hiện tại, hỗ trợ chỉ định múi giờ", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "description": "Tên múi giờ, ví dụ Asia/Shanghai, America/New_York", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "random_number", "description": "Tạo số ngẫu nhiên trong phạm vi chỉ định", "parameters": {"type": "object", "properties": {"min": {"type": "integer", "description": "Giá trị nhỏ nhất", "default": 0}, "max": {"type": "integer", "description": "Giá trị lớn nhất", "default": 100}}, "required": []}}},
    {"type": "function", "function": {"name": "text_length", "description": "Đếm số ký tự và số từ trong văn bản", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "Văn bản cần thống kê"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "Chuyển đổi đơn vị, hỗ trợ độ dài, khối lượng, nhiệt độ, ...", "parameters": {"type": "object", "properties": {"value": {"type": "number", "description": "Giá trị cần chuyển đổi"}, "from_unit": {"type": "string", "description": "Đơn vị nguồn, ví dụ km, miles, kg, pounds, celsius, fahrenheit"}, "to_unit": {"type": "string", "description": "Đơn vị đích"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "Lấy thời tiết hiện tại của thành phố, gồm nhiệt độ, độ ẩm và trạng thái", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "Tên thành phố, ví dụ Hà Nội, TP.HCM, New York"}, "unit": {"type": "string", "description": "Đơn vị nhiệt độ, celsius hoặc fahrenheit", "enum": ["celsius", "fahrenheit"], "default": "celsius"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "Tra cứu tỷ giá thời gian thực giữa hai loại tiền", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string", "description": "Mã tiền tệ nguồn, ví dụ USD, VND, EUR"}, "to_currency": {"type": "string", "description": "Mã tiền tệ đích, ví dụ USD, VND, EUR"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "Dịch văn bản sang ngôn ngữ đích", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "Văn bản cần dịch"}, "target_language": {"type": "string", "description": "Ngôn ngữ đích, ví dụ english, vietnamese, japanese, french"}}, "required": ["text", "target_language"]}}},
]

MOCK_RESULTS = {
    "calculate_math": lambda args: {"result": str(eval(str(args.get("expression", "0")).replace("^", "**").replace("×", "*").replace("÷", "/").replace("−", "-").replace("²", "**2").replace("³", "**3").replace("（", "(").replace("）", ")")))},
    "get_current_time": lambda args: {"datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "timezone": args.get("timezone", "Asia/Shanghai")},
    "random_number": lambda args: {"result": random.randint(int(args.get("min", 0)), int(args.get("max", 100)))},
    "text_length": lambda args: {"characters": len(args.get("text", "")), "words": len(args.get("text", "").split())},
    "unit_converter": lambda args: {"result": round(float(args.get("value", 0)) * 0.621371, 2), "from": f"{args.get('value', 0)} {args.get('from_unit', '')}", "to": args.get("to_unit", "")},
    "get_current_weather": lambda args: {"city": args.get("location"), "temperature": "22°C", "humidity": "65%", "condition": "Nắng"},
    "get_exchange_rate": lambda args: {"from": args.get("from_currency", ""), "to": args.get("to_currency", ""), "rate": 7.15},
    "translate_text": lambda args: {"translated": "hello world"},
}

TOOL_MAP = {t["function"]["name"]: t for t in TOOLS}

def get_tools(names):
    return [TOOL_MAP[n] for n in names]

TEST_CASES = [
    {"prompt": "Giúp tôi tính 256 nhân 37 bằng bao nhiêu", "tools": ["calculate_math", "get_current_time"]},
    {"prompt": "Bây giờ là mấy giờ?", "tools": ["get_current_time", "random_number"]},
    {"prompt": "Giúp tôi đổi 100 km sang dặm", "tools": ["unit_converter", "calculate_math"]},
    {"prompt": "Giúp tôi tạo một số ngẫu nhiên từ 1 đến 1000, sau đó tính bình phương", "tools": ["random_number", "calculate_math", "text_length"]},
    {"prompt": "Thời tiết hôm nay ở Hà Nội thế nào?", "tools": ["get_current_weather", "get_current_time"]},
    {"prompt": "Tra cứu tỷ giá USD sang VND", "tools": ["get_exchange_rate", "get_current_time"]},
    {"prompt": "Dịch 'Xin chào thế giới' sang tiếng Anh", "tools": ["translate_text", "text_length"]},
    {"prompt": "What is the weather in Tokyo? Also convert 30 celsius to fahrenheit.", "tools": ["get_current_weather", "unit_converter", "get_current_time"]},
]


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        model = MiniMindForCausalLM(MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe)))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer


def parse_tool_calls(text):
    matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    calls = []
    for m in matches:
        try:
            calls.append(json.loads(m.strip()))
        except Exception:
            pass
    return calls


def parse_tool_call_from_text(content):
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(pattern, content, re.DOTALL)
    if not matches:
        return None
    tool_calls = []
    for i, match in enumerate(matches):
        try:
            data = json.loads(match)
            tool_calls.append({
                "id": f"call_{i}",
                "function": {"name": data.get("name", ""), "arguments": json.dumps(data.get("arguments", {}), ensure_ascii=False)}
            })
        except Exception:
            pass
    return tool_calls if tool_calls else None


def execute_tool(call, arguments=None):
    name = call.get("name", "") if isinstance(call, dict) else call
    try:
        raw_args = call.get("arguments", {}) if isinstance(call, dict) else arguments
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    except Exception:
        args = {}
    fn = MOCK_RESULTS.get(name)
    if not fn:
        return {"error": f"Công cụ không tồn tại: {name}"}
    try:
        return fn(args)
    except Exception as e:
        return {"error": f"Thực thi công cụ thất bại: {str(e)[:80]}"}


def generate(model, tokenizer, messages, tools, args):
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=False)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(args.device)
    st = time.time()
    print('🧠: ', end='')
    generated_ids = model.generate(
        inputs["input_ids"], attention_mask=inputs["attention_mask"],
        max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        top_p=args.top_p, temperature=args.temperature
    )
    response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
    print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s') if args.show_speed else print()
    return response


def chat_api(client, messages, tools, args, stream=True):
    response = client.chat.completions.create(
        model=args.api_model, messages=messages, tools=tools,
        stream=stream, temperature=args.temperature,
        max_tokens=8192, top_p=args.top_p
    )
    if not stream:
        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls = choice.message.tool_calls
        if not tool_calls:
            tool_calls = parse_tool_call_from_text(content)
        print(f'🧠: {content}')
        return content, tool_calls
    print('🧠: ', end='', flush=True)
    content, tool_calls = "", None
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
            content += delta.content
        if delta.tool_calls:
            if tool_calls is None:
                tool_calls = []
            for tc_chunk in delta.tool_calls:
                idx = tc_chunk.index if tc_chunk.index is not None else len(tool_calls)
                while len(tool_calls) <= idx:
                    tool_calls.append({
                        "id": "",
                        "function": {"name": "", "arguments": ""}
                    })
                if tc_chunk.id:
                    tool_calls[idx]["id"] += tc_chunk.id
                if tc_chunk.function:
                    if tc_chunk.function.name:
                        tool_calls[idx]["function"]["name"] += tc_chunk.function.name
                    if tc_chunk.function.arguments:
                        tool_calls[idx]["function"]["arguments"] += tc_chunk.function.arguments
    print()
    if not tool_calls:
        tool_calls = parse_tool_call_from_text(content)
    return content, tool_calls


def run_case(prompt, tools, args, model=None, tokenizer=None, client=None):
    messages = [{"role": "user", "content": prompt}]
    while True:
        if args.backend == 'local':
            content = generate(model, tokenizer, messages, tools, args)
            tool_calls = parse_tool_calls(content)
        else:
            content, tool_calls = chat_api(client, messages, tools, args, stream=bool(args.stream))
        if not tool_calls:
            break
        tool_calls = [{
            "id": tc.id if hasattr(tc, 'id') else tc.get("id", ""),
            "name": tc.function.name if hasattr(tc, 'function') else tc["function"]["name"],
            "arguments": tc.function.arguments if hasattr(tc, 'function') else tc["function"]["arguments"]
        } for tc in tool_calls] if args.backend == 'api' else tool_calls
        messages.append({"role": "assistant", "content": content} if args.backend == 'local' else {"role": "assistant", "content": content, "tool_calls": [{"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}} for tc in tool_calls]})
        for tc in tool_calls:
            name = tc["name"]
            arguments = tc["arguments"]
            print(f'📞 [Tool Calling]: {name} | args={arguments}')
            result = execute_tool(tc if args.backend == 'local' else name, arguments)
            print(f'✅ [Tool Called]: {json.dumps(result, ensure_ascii=False)}')
            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)} if args.backend == 'local' else {"role": "tool", "content": json.dumps(result, ensure_ascii=False), "tool_call_id": tc["id"]})


def main():
    parser = argparse.ArgumentParser(description="Đánh giá ToolCall MiniMind")
    parser.add_argument('--backend', default='local', choices=['local', 'api'], type=str, help="Backend suy luận (local=mô hình cục bộ, api=giao diện tương thích OpenAI)")
    parser.add_argument('--load_from', default='../model', type=str, help="Đường dẫn tải mô hình (model=trọng số torch gốc, đường dẫn khác=định dạng transformers)")
    parser.add_argument('--save_dir', default='../out', type=str, help="Thư mục trọng số mô hình")
    parser.add_argument('--weight', default='full_sft', type=str, help="Tiền tố tên trọng số (pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo)")
    parser.add_argument('--hidden_size', default=768, type=int, help="Kích thước tầng ẩn")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="Số lớp ẩn")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="Có dùng kiến trúc MoE (0=không, 1=có)")
    parser.add_argument('--max_new_tokens', default=512, type=int, help="Độ dài sinh tối đa")
    parser.add_argument('--temperature', default=0.9, type=float, help="Nhiệt độ sinh, kiểm soát độ ngẫu nhiên (0-1, càng lớn càng ngẫu nhiên)")
    parser.add_argument('--top_p', default=0.9, type=float, help="Ngưỡng nucleus sampling (0-1)")
    parser.add_argument('--show_speed', default=0, type=int, help="Hiển thị tốc độ decode (tokens/s)")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="Thiết bị chạy")
    parser.add_argument('--api_base_url', default="http://localhost:11434/v1", type=str, help="base_url của API tương thích OpenAI")
    parser.add_argument('--api_key', default='sk-123', type=str, help="api_key của API tương thích OpenAI")
    parser.add_argument('--api_model', default='QuocBry/minimind-3:latest', type=str, help="Tên mô hình dùng khi gọi API")
    parser.add_argument('--stream', default=1, type=int, help="API mode có xuất theo luồng (0=không, 1=có)")
    args = parser.parse_args()

    model = tokenizer = client = None
    if args.backend == 'local': model, tokenizer = init_model(args)
    else: client = OpenAI(api_key=args.api_key, base_url=args.api_base_url)

    input_mode = int(input('[0] Tự động kiểm thử\n[1] Nhập tay\n'))

    cases = [{"prompt": case["prompt"], "tools": get_tools(case["tools"]), "tool_names": case["tools"]} for case in TEST_CASES] if input_mode == 0 else iter(lambda: {"prompt": input('💬: '), "tools": TOOLS, "tool_names": [t["function"]["name"] for t in TOOLS]}, {"prompt": "", "tools": TOOLS, "tool_names": []})
    for case in cases:
        if not case["prompt"]: break
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0:
            print(f'📦 Công cụ sẵn có: {case["tool_names"]}\n')
            print(f'💬: {case["prompt"]}')
        run_case(case["prompt"], case["tools"], args, model=model, tokenizer=tokenizer, client=client)
        print('\n' + '-' * 50 + '\n')


if __name__ == "__main__":
    main()
