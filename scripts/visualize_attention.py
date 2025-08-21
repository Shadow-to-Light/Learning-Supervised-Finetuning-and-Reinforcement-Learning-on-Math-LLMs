# scripts/visualize_attention.py
# 这个脚本可以加载任何你训练好的模型（基础版、SFT版、DPO版、ORPO版），并针对一个具体的数学问题，画出它的注意力图。

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 配置 ---
# 1. 选择你要分析的模型
MODEL_PATH = "../models/Qwen3-4B" 
# 2. 准备一个你感兴趣的数学问题
EXAMPLE_PROMPT = "A bakery has 200 cookies. It sells 3/4 of them. How many cookies does the bakery have now?"

def plot_attention_map(attention_matrix, tokens, layer_index, output_filename):
    """绘制并保存注意力图"""
    # 将token中的特殊字符进行转义，避免显示问题
    escaped_tokens = [t.replace('$', '\\$') for t in tokens]
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(attention_matrix, xticklabels=escaped_tokens, yticklabels=escaped_tokens, cmap="viridis", cbar=True)
    
    # 加上因果掩码的视觉提示线
    plt.plot(np.arange(len(tokens)) + 0.5, np.arange(len(tokens)) + 0.5, color='red', linestyle='--', linewidth=1)
    
    # plt.title(f"Attention Map - Layer {layer_index}", fontsize=16)
    plt.xlabel("Key Tokens (attended to)", fontfamily='Arial', fontsize=24, fontweight='bold')
    plt.ylabel("Query Tokens (attending from)", fontfamily='Arial', fontsize=24, fontweight='bold')
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    logging.info(f"注意力图已保存到: {output_filename}")

def main():
    # --- 加载模型和Tokenizer ---
    # 【修正1】将 logger.info 改为 logging.info
    logging.info(f"正在加载模型: {MODEL_PATH}")
    # 检查模型路径是否存在
    if not os.path.isdir(MODEL_PATH):
        logging.error(f"模型路径不存在: {MODEL_PATH}")
        return
        
    model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.bfloat16, 
    device_map="auto", 
    trust_remote_code=True,
    attn_implementation="eager"  # <-- 添加这一行
)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model.eval()

    # --- 准备输入 ---
    # 【注意】这里的模板是为特定SFT模型硬编码的，如果更换模型，可能需要调整此模板
    sft_prompt_template = (
        "Below is a math problem. Please solve it step-by-step and provide the final answer in the format '#### <number>'.\n\n"
        "### Problem:\n"
        "{question}\n\n"
        "### Solution:\n"
    )
    full_prompt = sft_prompt_template.format(question=EXAMPLE_PROMPT)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    # 获取输入prompt对应的tokens
    input_ids = inputs['input_ids'][0]
    tokens = [tokenizer.decode(token_id) for token_id in input_ids]
    
    # --- 【修正2】执行标准的前向传播，而不是 model.generate() ---
    logging.info("正在执行前向传播以获取Prompt的自注意力权重...")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # --- 提取和处理注意力权重 ---
    # `outputs.attentions` 是一个元组，长度为模型层数
    # 每个元素的形状: (batch_size, num_heads, seq_len, seq_len)
    attentions_all_layers = outputs.attentions
    
    # 选择要可视化的层 (例如: 第一层, 中间层, 最后一层)
    num_layers = len(attentions_all_layers)
    layers_to_plot = [0, num_layers // 2, num_layers - 1]
    
    logging.info(f"模型共有 {num_layers} 层，将可视化第 {layers_to_plot} 层的注意力。")
    
    for layer_idx in layers_to_plot:
        # 提取指定层的注意力矩阵
        # (batch_size, num_heads, seq_len, seq_len)
        attention_layer = attentions_all_layers[layer_idx].cpu()
        
        # 在多头注意力中取平均值
        # (batch_size, seq_len, seq_len) -> (seq_len, seq_len)
        # 【修正】在调用.numpy()之前，使用.float()将bfloat16转换为float32
        attention_matrix = attention_layer.mean(dim=1)[0].float().numpy()
        
        # 绘图
        output_filename = f"attention_map_layer_{layer_idx}.svg"
        plot_attention_map(attention_matrix, tokens, layer_idx, output_filename)

if __name__ == "__main__":
    main()