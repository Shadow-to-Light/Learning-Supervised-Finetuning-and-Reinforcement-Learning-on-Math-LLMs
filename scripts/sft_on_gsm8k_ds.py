# scripts/sft_on_gsm8k_ds.py (V3 - 终极版，同时解决VRAM和RAM溢出,并增加数据示例输出)

import os
import torch
import re
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import logging

# --- 1. 配置与日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 2. 路径与参数定义 ---
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "deepseek-math-7b-base")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "training_outputs", "deepseek-math-7b-base-sft-gsm8k")

# --- 3. 数据集处理 ---
def preprocess_dataset():
    """加载并预处理 GSM8k 数据集"""
    logger.info("开始加载和预处理 GSM8k 数据集...")
    
    # 加载数据集
    dataset = load_dataset("gsm8k", "main")
    
    # --- 新增：打印一条数据示例 ---
    logger.info("="*50)
    logger.info("GSM8k 数据集示例:")
    sample = dataset['train'][0]  # 取第一条训练数据作为示例
    logger.info(f"  问题 (Question): {sample['question']}")
    logger.info(f"  解答 (Answer): \n{sample['answer']}")
    logger.info("="*50)
    # --- 新增结束 ---

    # GSM8k 默认只有 train 和 test, 我们从 train 中划分出一部分作为验证集
    train_test_split = dataset['train'].train_test_split(test_size=0.05, seed=42)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    
    logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(eval_dataset)}")

    # 定义一个 Prompt 模板
    prompt_template = (
        "Below is a math problem. Please solve it step-by-step and provide the final answer in the format '#### <number>'.\n\n"
        "### Problem:\n"
        "{question}\n\n"
        "### Solution:\n"
        "{answer}"
    )

    def format_prompt(example):
        # 将问题和答案填充到模板中
        full_prompt = prompt_template.format(question=example['question'], answer=example['answer'])
        return {"text": full_prompt}

    # 应用格式化函数
    train_dataset = train_dataset.map(format_prompt)
    eval_dataset = eval_dataset.map(format_prompt)
    
    logger.info("数据集预处理完成。")
    return train_dataset, eval_dataset

# --- 4. 评估逻辑 ---

# 新增！这个函数在每一批评估数据上运行，大大减少内存占用
def preprocess_logits_for_metrics(logits, labels):
    """在累积前预处理logits，只保留预测的token ID。"""
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

def compute_metrics(eval_preds):
    """计算准确率，现在它接收的是预处理过的predictions"""
    # eval_preds 现在是 (predictions, labels) 的元组
    preds, labels = eval_preds
    
    # preds 已经是 token IDs，不再需要 argmax
    # 将 preds 中由 padding 产生的 -100 替换为 pad_token_id
    preds[preds == -100] = tokenizer.pad_token_id
    # labels 依然需要处理 padding token
    labels[labels == -100] = tokenizer.pad_token_id
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    def extract_answer(text):
        match = re.search(r'####\s*([\d,]+\.?\d*)', text)
        if match:
            try:
                return float(match.group(1).replace(',', ''))
            except ValueError:
                return None
        return None

    correct = 0
    total = 0
    for pred_str, label_str in zip(decoded_preds, decoded_labels):
        label_ans = extract_answer(label_str)
        pred_ans = extract_answer(pred_str)
        
        if label_ans is not None:
            total += 1
            if pred_ans is not None and abs(pred_ans - label_ans) < 1e-4:
                correct += 1
                
    accuracy = correct / total if total > 0 else 0
    return {"accuracy": accuracy}

# --- 5. 主训练流程 ---
if __name__ == "__main__":
    logger.info("开始 SFT 训练流程 (V3 - 内存终极优化版，且带有示例输出)...")
    
    train_dataset, eval_dataset = preprocess_dataset()

    logger.info(f"从本地路径加载 Tokenizer: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("设置 pad_token = eos_token")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    logger.info(f"从本地路径加载量化模型: {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    logger.info("应用 PEFT (LoRA) 来包装量化模型...")
    model.print_trainable_parameters()
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=16,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        bf16=True,
        logging_strategy="steps",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        compute_metrics=compute_metrics,
        # --- 终极修复！ ---
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    logger.info("一切准备就绪，开始训练！")
    trainer.train()

    logger.info("训练完成！模型和训练日志已保存在 " + OUTPUT_DIR)
