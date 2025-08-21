# check_local_model.py
# 检查本地模型是否可以正常加载和推理
# 这里假设你下载了Qwen3-0.6B

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # 指定你刚刚下载的模型文件夹路径
    # 这个路径是相对路径
    model_path = "../models/Qwen3-0.6B"
    
    logging.info(f"开始从本地路径加载模型: {model_path}")

    try:
        # 加载分词器 (Tokenizer)
        logging.info("正在加载 Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        logging.info("Tokenizer 加载成功！")

        # 加载模型
        logging.info("正在加载模型...")
        # 如果你有GPU，模型会自动加载到GPU上
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16, # 在4090上使用bfloat16可以加速且保持精度
            device_map="auto",
            trust_remote_code=True
        )
        logging.info("模型加载成功！")
        logging.info(f"模型已加载到设备: {model.device}")

        # --- 进行一次简单的推理测试 ---
        logging.info("正在进行推理测试...")
        prompt = "你好，请自我介绍一下。"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        logging.info("推理测试完成！")
        print("\n" + "="*50)
        print(f"用户提问: {prompt}")
        print(f"模型回答: {response}")
        print("="*50 + "\n")
        logging.info("本地模型部署验证成功！")

    except Exception as e:
        logging.error(f"加载或测试模型时发生错误: {e}")
        logging.error("请检查模型文件是否已完整下载到 'models/Qwen3-0.6B' 文件夹中。")

if __name__ == "__main__":
    main()