# scripts/orpo_draw_final.py (V3 - 最终可视化版)

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import LoraConfig, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import ORPOConfig, ORPOTrainer
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Union, Any

# --- 1. 配置与日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 2. 路径与参数定义 ---
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
# 我们直接在SFT模型的基础上做ORPO，看能否进一步提升
BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Qwen3-0.6B-SFT-GSM8k-Merged")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "training_outputs", "qwen3-0.6B-orpo-dis")

QUICK_TEST = True###
NUM_SAMPLES_QUICK_TEST = 100

# --- 3. 自定义训练器以记录梯度 (最终修正版) ---
class GradLoggingORPOTrainer(ORPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_gradients = []

    def training_step(self, *args, **kwargs):
        """
        使用 *args 和 **kwargs 来接收所有可能的参数，确保兼容性。
        """
        # 1. 将接收到的所有参数原封不动地传递给父类的 training_step 方法
        #    这会正常完成loss计算和反向传播
        loss = super().training_step(*args, **kwargs)

        # 2. 从参数中获取 model 对象 (它通常是第一个位置参数)
        #    args 是一个元组 (tuple)，包含所有位置参数
        model = args[0]
        
        # 3. 梯度记录逻辑保持不变
        for param in model.parameters():
            if param.grad is not None and param.requires_grad:
                grads = param.grad.detach().cpu().numpy().flatten()
                self.all_gradients.extend(grads)
                
        return loss

# --- 新增: 创建一个Callback来记录grad_norm ---
class GradNormLoggerCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.grad_norms = []
        self.steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        # on_log会在每次记录日志时被调用 (每 logging_steps 一次)
        if logs is not None and 'grad_norm' in logs:
            self.grad_norms.append(logs['grad_norm'])
            self.steps.append(state.global_step)


# --- 4. 主训练流程 ---
def main():
    logger.info("开始数学领域ORPO训练流程 (数据集: argilla/distilabel-math-preference-dpo)...")
    
    full_dataset = load_dataset("argilla/distilabel-math-preference-dpo", split="train")

    if QUICK_TEST:
        logger.warning(f"!!! 快速测试模式已开启，仅使用 {NUM_SAMPLES_QUICK_TEST} 条数据 !!!")
        dataset = full_dataset.select(range(NUM_SAMPLES_QUICK_TEST))
    else:
        dataset = full_dataset
        
    dataset = dataset.rename_column("instruction", "prompt")
    dataset = dataset.rename_column("chosen_response", "chosen")
    dataset = dataset.rename_column("rejected_response", "rejected")
    logger.info("已将数据集的列重命名为 'prompt', 'chosen', 'rejected'。")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        logger.info("设置 pad_token = eos_token 并设置 padding_side='left'")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, quantization_config=quantization_config,
        device_map="auto", trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        lora_alpha=16, lora_dropout=0.1, r=8, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    orpo_config = ORPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-6, # 从 8e-6 大幅降低
        bf16=True,
        logging_strategy="steps", logging_steps=10,
        save_strategy="steps", save_steps=50,
        save_total_limit=2,
        report_to="tensorboard",
        beta=0.1, #提高beta的值。这相当于告诉模型：“请更重视、更努力地区分好与坏的答案！”
        max_prompt_length=1024,
        max_length=2048,
        max_grad_norm=15, # 梯度裁剪的阈值（我这个很大）
        # removed_unused_columns=False,没有这个参数
    )
    
    # --- 新增: 初始化我们的Callback ---
    grad_norm_callback = GradNormLoggerCallback()

    orpo_trainer = GradLoggingORPOTrainer(
        model=model,
        args=orpo_config,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        # --- 新增: 将Callback传递给Trainer ---
        callbacks=[grad_norm_callback],
    )

    logger.info("一切准备就绪，开始ORPO训练！")
    orpo_trainer.train()

    logger.info("ORPO训练完成！")

    # --- 修改: 训练结束后，现在我们有两个来源的数据 ---
    
    # 1. 处理并保存所有单个梯度 (来自 GradLoggingORPOTrainer)
    logger.info("开始处理【单个梯度】的分布数据...")
    gradients = np.array(orpo_trainer.all_gradients)
    
    if len(gradients) == 0:
        logger.warning("没有收集到任何梯度数据，跳过保存和绘图。")
    else:
        # 剔除极端离群点
        lower_quantile = np.quantile(gradients, 0.001)
        upper_quantile = np.quantile(gradients, 0.999)
        gradients_for_plotting = gradients[(gradients > lower_quantile) & (gradients < upper_quantile)]
        logger.info(f"为绘图剔除了极端离群点，保留了 {len(gradients_for_plotting)} / {len(gradients)} 个数据点。")

        # 保存文本文件
        grad_log_file = os.path.join(PROJECT_ROOT, "scripts", "gradient_log.txt")
        logger.info(f"正在将 *全部* 梯度值保存到: {grad_log_file}")
        np.savetxt(grad_log_file, gradients, fmt='%.6f', newline=' ')
        with open(grad_log_file, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write('grad:' + content)
        logger.info(f"全部梯度值已保存到: {grad_log_file}")

        # --- 关键修改：使用新的绘图逻辑 ---
        sns.set_theme(style="whitegrid")

        # 1.1 绘制对数刻度直方图 (Log-Scale Histogram)
        plt.figure(figsize=(12, 8))
        # 使用 log=True 来启用Y轴对数刻度
        plt.hist(gradients_for_plotting, bins=200, color='darkblue', alpha=0.7, log=True)
        plt.title("Log-Scale Histogram of Gradients during ORPO Training", fontsize=16)
        plt.xlabel("Gradient Value", fontsize=12)
        plt.ylabel("Frequency (Log Scale)", fontsize=12)
        log_hist_file = os.path.join(PROJECT_ROOT, "scripts", "gradient_log_histogram.png")
        plt.savefig(log_hist_file)
        plt.close()
        logger.info(f"梯度对数直方图已保存到: {log_hist_file}")
        
        # 1.2 绘制小提琴图 (Violin Plot) - 保持不变，因为它本身就是一种有效的信息展示
        plt.figure(figsize=(12, 8))
        sns.violinplot(y=gradients_for_plotting, color='skyblue', inner='box')
        plt.title("Violin Plot of Gradients during ORPO Training", fontsize=16)
        plt.ylabel("Gradient Value", fontsize=12)
        violin_plot_file = os.path.join(PROJECT_ROOT, "scripts", "gradient_violin_plot.png")
        plt.savefig(violin_plot_file)
        plt.close()
        logger.info(f"梯度小提琴图已保存到: {violin_plot_file}")
    
    # 2. 绘制梯度范数曲线图 (来自 GradNormLoggerCallback)
    logger.info("开始处理【梯度范数】的时间序列数据...")
    if grad_norm_callback.steps:
    # 绘制并保存梯度范数曲线图
        plt.figure(figsize=(12, 8))
        sns.lineplot(x=grad_norm_callback.steps, y=grad_norm_callback.grad_norms)
        plt.title("Gradient Norm During ORPO Training", fontsize=16)
        plt.xlabel("Training Steps", fontsize=12)
        plt.ylabel("Gradient Norm", fontsize=12)
        plt.grid(True)
        grad_norm_curve_file = os.path.join(PROJECT_ROOT, "scripts", "gradient_norm_over_time.png")
        plt.savefig(grad_norm_curve_file)
        plt.close()
        logger.info(f"梯度范数曲线图已保存到: {grad_norm_curve_file}")
    else:
        logger.warning("未能收集到梯度范数数据。")

    # 保存最终模型
    final_model_path = os.path.join(OUTPUT_DIR, "final_checkpoint")
    orpo_trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logger.info(f"最终的ORPO模型已保存在: {final_model_path}")

if __name__ == "__main__":
    main()
