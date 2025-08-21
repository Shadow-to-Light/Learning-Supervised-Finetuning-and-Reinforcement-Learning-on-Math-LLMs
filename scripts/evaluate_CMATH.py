# scripts/evaluate_model_on_cmath_official.py
# 适用于 CMATH 数据集，并集成了官方评估逻辑。
# 包含批量推理、torch.compile加速、padding修复和详细结果记录。

import os
import torch
import re
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

# --- 1. 配置与日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 2. 评估配置 ---
# ==============================================================================
# 1. 修改模型路径
# MODEL_TO_EVALUATE_PATH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), "models", "Qwen3-4B")
MODEL_TO_EVALUATE_PATH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), "models", "deepseek-math-7b-base-SFT-GSM8k-Merged")

# 2. 设置评估模式 (True: 基础模型, False: 指令/SFT模型)
EVALUATE_BASE_MODEL = True

# 3. 数据集与输出配置
DATASET_NAME = "weitianwen/cmath"
DATASET_SPLIT = "test"
model_name = os.path.basename(MODEL_TO_EVALUATE_PATH)
OUTPUT_FILENAME = f"evaluation_details_{model_name}_cmath_official.json"

# 4. 测试与推理配置
QUICK_TEST = False
NUM_SAMPLES_QUICK_TEST = 100
BATCH_SIZE = 96 # 对于7B模型，需要减小Batch Size
# ==============================================================================

# --- 3. CMath 官方评估函数 (来源: https://github.com/XiaoMi/cmath) ---
# ==============================================================================
REG_DIGITS_SINGLETON = re.compile(r"^\d+[./]?\d*%?$")
REG_DIGITS_BEGIN = re.compile(r"^(\d+[./]?\d*%?) ?(?=[\u4e00-\u9fa5,，。°℃$])")
REG_DIGITS_MIDDLE = re.compile(r"(?<=[\u4e00-\u9fa5$:：，＝≈=->{]) ?(\d+[./]?\d*%?) ?(?:[\u4e00-\u9fa5,，。°℃$}（]|\([^\d\s]|\.$)")
REG_DIGITS_END = re.compile(r"(?<=[\u4e00-\u9fa5$:：＝≈=\->]) ?(\d+[./]?\d*%?)$")

REG_NORM = re.compile(r"(?<=\d)[, ](?=\d)")
REG_LATEX_FRAC = re.compile(r'\\frac{([^}]+)}{([^}]+)}')
REG_CN_FRAC = re.compile(r'(\d+)分之(\d+)')

def has_exception(answer: str) -> bool:
    if answer is None or len(answer.strip()) == 0: return True
    reg_timeout = re.compile("(请求.*超时)|(timeout)")
    if bool(re.search(reg_timeout, answer)): return True
    reg_error = re.compile("error|异常|失败|content_filter")
    if "{" in answer and "}" in answer and (bool(re.search(reg_error, answer))): return True
    return False

def extract_cn_fractal(line):
    res = re.findall(REG_CN_FRAC, line)
    return ["{}/{}".format(b, a) for a, b in res] if len(res) != 0 else res

def extract_digits_from_line(line):
    res1 = re.findall(REG_DIGITS_BEGIN, line)
    res2 = re.findall(REG_DIGITS_MIDDLE, line)
    res3 = re.findall(REG_DIGITS_END, line)
    res4 = re.findall(REG_DIGITS_SINGLETON, line)
    res_cn_frac = extract_cn_fractal(line)
    concat = res1 + res2 + res_cn_frac + res3 + res4
    candidates = [s.strip() for s in concat]
    return [s for s in candidates if not (s.startswith("/") or s.endswith("/"))]

def extract_digits_prediction(response, truncation="t"):
    if has_exception(response): return ["ERROR"]
    response = REG_LATEX_FRAC.sub(r'\1/\2', response)
    response = re.sub(REG_NORM, "", response)
    candidates = [extract_digits_from_line(line) for line in response.splitlines()]
    candidates = [item for sublist in candidates for item in sublist]
    
    if truncation == "t": # 只考虑最后两个数字
        res = candidates[-2:] if len(candidates) > 2 else candidates
    else: # 其他截断模式，为简化我们只实现 "t"
        res = candidates
    return list(set(res))

def string2num(string: str):
    string = string.strip()
    if string.endswith("%"):
        return float(string.replace("%", "")) / 100
    if "/" in string:
        parts = string.split("/")
        if len(parts) != 2 or float(parts[1]) == 0.0: return 0
        return float(parts[0]) / float(parts[1])
    return float(string)

def match_digits(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    relative_diff = abs(a - b) / (min(abs(a), abs(b)) + 1e-6)
    return relative_diff < 1e-2

def match_digit_response(golden, responses: list) -> bool:
    if "ERROR" in responses: return False
    try:
        golden_num = string2num(golden)
    except ValueError:
        return False # 如果标准答案无法转换为数字，则无法比较

    for r in responses:
        try:
            num = string2num(r)
            if match_digits(golden_num, num):
                return True
        except (ValueError, ZeroDivisionError):
            pass
    return False
# ==============================================================================
# --- 官方评估函数结束 ---


# --- 4. Prompt模板定义 --- (保持中文模板)
sft_prompt_template = (
    "请逐步解决以下数学问题，并以“#### <答案>”的格式给出最终答案。\n\n"
    "### 问题:\n"
    "{question}\n\n"
    "### 解答:\n"
)
base_model_few_shot_template = (
    "请逐步解决以下数学问题，并以“#### <答案>”的格式给出最终答案。\n\n"
    "问题: 小明有48个弹珠，他在四月份卖给了他的朋友们。五月份他卖出的弹珠数量是四月份的一半。小明在四月和五月总共卖出了多少个弹珠？\n"
    "解答: 小明在五月份卖出了 48 / 2 = 24 个弹珠。\n在四月和五月，他总共卖出了 48 + 24 = 72 个弹珠。\n#### 72\n\n"
    "问题: {question}\n"
    "解答:"
)

# --- 5. 主评估流程 ---
def main():
    logger.info(f"开始使用官方评估逻辑评估模型: {MODEL_TO_EVALUATE_PATH}")
    prompt_template = base_model_few_shot_template if EVALUATE_BASE_MODEL else sft_prompt_template

    logger.info("正在加载模型和Tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_TO_EVALUATE_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TO_EVALUATE_PATH, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    logger.info("Tokenizer已配置: pad_token=eos_token, padding_side='left'")
    
    model.eval()
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
            logger.info("模型已成功使用 torch.compile 进行优化！")
        except Exception as e:
            logger.warning(f"torch.compile 优化失败，将使用常规模式。错误: {e}")

    logger.info("正在加载和准备数据集...")
    # **修正**: 使用 split=DATASET_SPLIT
    full_dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    
    if QUICK_TEST:
        logger.warning(f"!!! 快速测试模式已开启，仅评估前 {NUM_SAMPLES_QUICK_TEST} 条数据 !!!")
        eval_dataset = full_dataset.select(range(NUM_SAMPLES_QUICK_TEST))
    else:
        eval_dataset = full_dataset

    # **修正**: create_prompts 现在只使用 'question' 列
    def create_prompts(examples):
        return {"prompt": [prompt_template.format(question=q) for q in examples["question"]]}

    # **修正**: remove_columns 现在移除所有原始列，只保留 'prompt'
    prompt_dataset = eval_dataset.map(create_prompts, batched=True, remove_columns=eval_dataset.column_names)
    data_loader = DataLoader(prompt_dataset, batch_size=BATCH_SIZE)

    logger.info(f"本次评估将运行 {len(eval_dataset)} 条数据，批次大小为 {BATCH_SIZE}。")
    detailed_results = []
    correct_count = 0
    
    logger.info("开始批量进行推理和评估...")
    for i, batch in enumerate(tqdm(data_loader, desc="Evaluating in Batches")):
        prompts = batch["prompt"]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=1024, do_sample=False, pad_token_id=tokenizer.pad_token_id
            )
        
        generated_texts = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        start_index = i * BATCH_SIZE
        for j, gen_text in enumerate(generated_texts):
            idx = start_index + j
            if idx < len(eval_dataset):
                example = eval_dataset[idx]
                
                # --- 核心逻辑修改 ---
                # **修正**: 使用 'golden' 列作为标准答案
                ground_truth = example['golden'] 
                # **修正**: 使用官方函数提取候选答案列表
                # 修正后的代码
                predicted_candidates = extract_digits_prediction(gen_text, truncation=None)                # **修正**: 使用官方函数进行匹配
                is_correct = match_digit_response(ground_truth, predicted_candidates)
                
                if is_correct:
                    correct_count += 1

                detailed_results.append({
                    "index": idx + 1,
                    "question": example['question'],
                    "model_full_answer": gen_text,
                    "model_extracted_candidates": predicted_candidates, # 记录候选列表
                    "golden_answer": ground_truth,
                    "is_correct": is_correct
                })

    total_predictions = len(detailed_results)
    accuracy = correct_count / total_predictions if total_predictions > 0 else 0
    
    final_output = {
        "summary": {
            "model_name": model_name,
            "dataset": f"{DATASET_NAME} ({DATASET_SPLIT} split)",
            "total_questions": total_predictions,
            "correct_count": correct_count,
            "incorrect_count": total_predictions - correct_count,
            "accuracy": f"{accuracy:.4f}",
        },
        "detailed_results": detailed_results
    }
    
    logger.info(f"正在将详细评估结果写入文件: {OUTPUT_FILENAME}...")
    with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    
    logger.info("="*50)
    summary = final_output["summary"]
    logger.info(f"评估完成！结果摘要:")
    logger.info(f"  模型: {summary['model_name']}")
    logger.info(f"  数据集: {summary['dataset']}")
    logger.info(f"  题目总数: {summary['total_questions']}")
    logger.info(f"  正确数: {summary['correct_count']}")
    logger.info(f"  最终准确率 (Accuracy): {summary['accuracy']}")
    logger.info(f"  详细结果已保存至: {OUTPUT_FILENAME}")
    logger.info("="*50)

if __name__ == "__main__":
    main()