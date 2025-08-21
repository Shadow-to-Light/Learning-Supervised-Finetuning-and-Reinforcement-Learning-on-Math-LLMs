# scripts/dpo_on_mathdpopairs.py

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
import logging

# --- 1. 配置与日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 2. 路径定义 ---
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# 使用你SFT合并后的模型作为DPO的基础
BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Qwen3-0.6B-SFT-GSM8k-Merged")
# 新的DPO训练的输出路径
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "training_outputs", "qwen3-0.6B-math-dpo-pairs")

# --- 3. DPO 训练流程 ---
def main():
    logger.info("开始数学领域DPO训练流程 (数据集: reciprocate/math_dpo_pairs)...")
    
    logger.info("正在加载偏好数据集: reciprocate/math_dpo_pairs...")
    # 注意：reciprocate/math_dpo_pairs 数据集本身不大，我们可以直接使用完整训练集，共有2418条
    dataset = load_dataset("reciprocate/math_dpo_pairs", split="train")

    # 为了快速迭代，我们先使用一个子集。跑通后，你可以删除[:300]来使用完整数据集
    # dataset = load_dataset("reciprocate/math_dpo_pairs", split="train[:300]")

    # --- 关键修复：更鲁棒的数据清洗 ---
    logger.info("正在进行数据有效性检查...")
    original_size = len(dataset)
    
    # 一个更简单且鲁棒的过滤函数，只检查字段是否存在且为真（不为None或空字符串）
    def is_valid(example):
        return (
            example.get("prompt") and
            example.get("chosen") and
            example.get("rejected")
        )
    
    dataset = dataset.filter(is_valid)
    new_size = len(dataset)

    if original_size > new_size:
        logger.info(f"数据清洗完成。移除了 {original_size - new_size} 条不完整的行。")
    else:
        logger.info("数据质量很好，未发现不完整的行。")

    # --- 关键修复：安全地打印示例 ---
    if new_size > 0:
        logger.info("="*50)
        logger.info("math-dpo-pairs DPO 数据集示例:")
        sample = dataset[0]
        # 使用 .get() 访问，即使字段不存在也不会报错
        logger.info(f"  Prompt:\n{sample.get('prompt')}")
        logger.info(f"  Chosen (好答案):\n{sample.get('chosen')}")
        logger.info(f"  Rejected (坏答案):\n{sample.get('rejected')}")
        logger.info("="*50)
    else:
        logger.error("错误：数据清洗后，数据集为空！无法继续训练。请检查数据集或过滤逻辑。")
        return # 提前退出
    # --- 修复结束 ---

    # 查看一条数据示例
    logger.info("="*50)
    logger.info("math-dpo-pairs DPO 数据集示例:")
    sample = dataset[0]
    logger.info(f"  Prompt:\n{sample['prompt']}")
    logger.info(f"  Chosen (好答案):\n{sample['chosen']}")
    logger.info(f"  Rejected (坏答案):\n{sample['rejected']}")
    logger.info("="*50)

    logger.info(f"从本地路径加载SFT后的模型: {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("设置 pad_token = eos_token")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
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

    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        bf16=True,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to="tensorboard",
        beta=0.1,
        # DPOConfig不支持remove_unused_columns，DPOTrainer会自己处理
    )

    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=dpo_config,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        # max_prompt_length=1024, # 截断过长的prompt
        # max_length=2048,      # 截断prompt+completion的总长
    )

    logger.info("一切准备就绪，开始DPO训练！")
    dpo_trainer.train()

    logger.info("DPO训练完成！")
    
    final_model_path = os.path.join(OUTPUT_DIR, "final_checkpoint")
    dpo_trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"最终的DPO模型已保存在: {final_model_path}")

if __name__ == "__main__":
    main()