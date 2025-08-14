# scripts/evaluate_model_qwenorpodis.py(包含回答细节,加速,padding修复)

import os
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import json

# --- 1. 配置与日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 2. 评估配置 ---
# ==============================================================================
# 1. 修改模型路径
#    - 评估SFT模型: ".../Qwen3-0.6B-SFT-ORPOw.dis.-Merged"
#    - 评估基础模型: ".../Qwen3-0.6B"
# MODEL_TO_EVALUATE_PATH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), "models", "Qwen3-0.6B-SFT-ORPOw.dis.-Merged")
# MODEL_TO_EVALUATE_PATH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), "models", "deepseek-math-7b-base")
MODEL_TO_EVALUATE_PATH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), "models", "Qwen3-4B")


# 2. 设置评估模式
#    - 评估SFT模型: EVALUATE_BASE_MODEL = False
#    - 评估基础模型: EVALUATE_BASE_MODEL = True
EVALUATE_BASE_MODEL = True

# ==============================================================================
QUICK_TEST = False
NUM_SAMPLES_QUICK_TEST = 100
# ==============================================================================

# --- 新增：批量推理配置 ---
# ==============================================================================
# 调整 BATCH_SIZE 可以影响速度和显存占用。对于0.6B模型在4090上，8或16是很好的起点。
BATCH_SIZE = 32
# ==============================================================================

# 从完整路径中提取出最后的文件夹名，即模型名
model_name = os.path.basename(MODEL_TO_EVALUATE_PATH)
# 使用f-string格式化新的文件名
OUTPUT_FILENAME = f"evaluation_details_{model_name}.json"

DATASET_NAME = "gsm8k"
DATASET_CONFIG = "main"
DATASET_SPLIT = "test"

# --- 3. 答案提取函数 ---
def extract_answer(text):
    match = re.search(r'####\s*([\d,]+\.?\d*)', text)
    if match:
        try: return float(match.group(1).replace(',', ''))
        except ValueError: return None
    
    numbers = re.findall(r'[\d,]+\.?\d+', text)
    if numbers:
        try: return float(numbers[-1].replace(',', ''))
        except ValueError: return None
    return None

# --- 4. Prompt模板定义 ---

# 这是为SFT模型定制的“指令”模板 (零样本)
sft_prompt_template = (
    "Below is a math problem. Please solve it step-by-step and provide the final answer in the format '#### <number>'.\n\n"
    "### Problem:\n"
    "{question}\n\n"
    "### Solution:\n"
)

# 这是为基础模型设计的、更公平的“示例”模板 (少样本)
base_model_few_shot_template = (
    "Please solve the following math problem step-by-step and provide the final answer in the format '#### <number>'.\n\n"
    "Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n"
    "A: Natalia sold 48/2 = 24 clips in May.\nIn April and May, she sold 48 + 24 = 72 clips.\n#### 72\n\n"
    "Q: {question}\n"
    "A:"
)

# --- 5. 主评估流程 ---
def main():
    logger.info(f"开始评估模型: {MODEL_TO_EVALUATE_PATH}")
    if EVALUATE_BASE_MODEL:
        prompt_template = base_model_few_shot_template
    else:
        prompt_template = sft_prompt_template

    logger.info("正在加载模型和Tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_TO_EVALUATE_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TO_EVALUATE_PATH, trust_remote_code=True)
    
    # --- 关键修复：确保总是设置正确的padding ---
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 无论pad_token是否存在，对于批量生成，都应将padding_side设为'left'
    tokenizer.padding_side = 'left'
    logger.info("Tokenizer已配置: pad_token=eos_token, padding_side='left'")
    # --- 修复结束 ---
    
    model.eval()
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        logger.info("模型已成功使用 torch.compile 进行优化！")
    except Exception as e:
        logger.warning(f"torch.compile 优化失败，将使用常规模式。错误: {e}")

    logger.info("模型加载完成。")

    logger.info("正在加载和准备数据集...")
    full_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    
    if QUICK_TEST:
        logger.warning(f"!!! 快速测试模式已开启，仅评估前 {NUM_SAMPLES_QUICK_TEST} 条数据 !!!")
        eval_dataset = full_dataset.select(range(NUM_SAMPLES_QUICK_TEST))
    else:
        eval_dataset = full_dataset

    def create_prompts(examples):
        return {"prompt": [prompt_template.format(question=q) for q in examples["question"]]}

    prompt_dataset = eval_dataset.map(create_prompts, batched=True, remove_columns=["answer"])
    data_loader = DataLoader(prompt_dataset, batch_size=BATCH_SIZE)

    logger.info(f"本次评估将运行 {len(eval_dataset)} 条数据，批次大小为 {BATCH_SIZE}。")

    detailed_results = []
    correct_indices = []
    incorrect_indices = []
    
    logger.info("开始批量进行推理和评估...")
    for i, batch in enumerate(tqdm(data_loader, desc="Evaluating in Batches")):
        prompts = batch["prompt"]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.pad_token_id,
            )
        
        generated_texts = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        start_index = i * BATCH_SIZE
        for j, gen_text in enumerate(generated_texts):
            idx = start_index + j
            if idx < len(eval_dataset):
                example = eval_dataset[idx]
                predicted_answer = extract_answer(gen_text)
                ground_truth_answer = extract_answer(example['answer'])
                
                is_correct = False
                if predicted_answer is not None and ground_truth_answer is not None:
                    if abs(predicted_answer - ground_truth_answer) < 1e-4:
                        is_correct = True
                
                if is_correct:
                    correct_indices.append(idx + 1)
                else:
                    incorrect_indices.append(idx + 1)

                detailed_results.append({
                    "index": idx + 1,
                    "question": example['question'],
                    "model_full_answer": gen_text,
                    "correct_full_answer": example['answer'],
                    "is_correct": is_correct
                })

    total_predictions = len(detailed_results)
    correct_predictions = len(correct_indices)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    final_output = {
        "summary": {
            "model_name": model_name, "total_questions": total_predictions,
            "correct_count": correct_predictions, "incorrect_count": len(incorrect_indices),
            "accuracy": f"{accuracy:.4f}", "correct_indices": correct_indices,
            "incorrect_indices": incorrect_indices
        },
        "detailed_results": detailed_results
    }
    
    logger.info(f"正在将详细评估结果写入文件: {OUTPUT_FILENAME}...")
    with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    logger.info("写入完成。")
    
    logger.info("="*50)
    logger.info("评估完成！")
    if QUICK_TEST:
        logger.warning("!!! 本次为快速测试结果，不代表最终性能 !!!")
    
    summary = final_output["summary"]
    logger.info(f"模型: {summary['model_name']}")
    logger.info(f"评估题目数: {summary['total_questions']}")
    logger.info(f"正确解答数: {summary['correct_count']}")
    logger.info(f"答对的题目序号: {', '.join(map(str, summary['correct_indices']))}")
    logger.info(f"答错的题目序号: {', '.join(map(str, summary['incorrect_indices']))}")
    logger.info(f"最终准确率 (Accuracy): {summary['accuracy']}")
    logger.info(f"详细结果已保存到 {OUTPUT_FILENAME}，请查看。")
    logger.info("="*50)

if __name__ == "__main__":
    main()
