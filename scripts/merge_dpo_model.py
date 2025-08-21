# scripts/merge_dpo_model.py

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging

# --- 1. 配置与日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 2. 路径定义 ---
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# 基础模型是SFT合并后的模型
BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Qwen3-0.6B-SFT-GSM8k-Merged")

# LoRA适配器是你刚刚DPO训练产出的最终检查点
# !!! 请再次确认这个路径与你日志中显示的路径完全一致 !!!
# 根据你的日志，路径是 '.../qwen3-0.6B-stepdpo/final_checkpoint'
# 如果你在脚本里改了OUTPUT_DIR，请在这里同步修改
LORA_ADAPTER_PATH = os.path.join(PROJECT_ROOT, "training_outputs", "qwen3-0.6B-stepdpo", "final_checkpoint")

# 合并后最终模型的保存路径
MERGED_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Qwen3-0.6B-SFT-StepDPO-Merged")

# --- 3. 合并流程 ---
def main():
    logger.info("开始合并DPO LoRA权重...")
    
    # 检查LoRA路径是否存在
    if not os.path.exists(LORA_ADAPTER_PATH):
        logger.error(f"错误：LoRA适配器路径不存在！请检查路径：{LORA_ADAPTER_PATH}")
        return

    # 以全精度加载SFT模型作为基础
    logger.info(f"正在从 '{BASE_MODEL_PATH}' 加载SFT基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    # 将DPO的LoRA适配器加载到SFT模型上
    logger.info(f"正在从 '{LORA_ADAPTER_PATH}' 加载DPO LoRA适配器...")
    model_with_lora = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)

    # 执行合并
    logger.info("正在执行合并操作 (merge_and_unload)...")
    merged_model = model_with_lora.merge_and_unload()
    logger.info("合并完成！")

    # 创建保存目录
    os.makedirs(MERGED_MODEL_PATH, exist_ok=True)
    
    # 保存合并后的最终模型和tokenizer
    logger.info(f"正在将最终模型保存到 '{MERGED_MODEL_PATH}'...")
    merged_model.save_pretrained(MERGED_MODEL_PATH)
    tokenizer.save_pretrained(MERGED_MODEL_PATH)
    
    logger.info("="*50)
    logger.info("StepDPO模型合并成功！")
    logger.info(f"你的最终版模型已保存在: {MERGED_MODEL_PATH}")
    logger.info("下一步，我们就可以用 evaluate_model.py 来测试它的最终性能了！")
    logger.info("="*50)

if __name__ == "__main__":
    main()
