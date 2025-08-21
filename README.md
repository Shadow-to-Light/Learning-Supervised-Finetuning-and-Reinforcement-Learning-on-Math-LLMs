# Learning Supervised Finetuning and Reinforcement Learning on Math LLMs / 数学大模型有监督微调与强化学习实践

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

A collection of practical scripts and implementations for Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) on mathematical reasoning tasks, primarily using the Qwen3 model and evaluated on GSM8K/CMATH datasets.

本项目收集了针对数学推理任务的有监督微调（SFT）和强化学习（RL）的实践脚本与实现，主要基于Qwen3模型并在GSM8K/CMATH数据集上进行评估。

## 📖 Overview / 项目概述

This repository provides implemented and validated code examples for Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) techniques applied to mathematical reasoning tasks using large language models. The codebase includes visualization tools for attention mechanisms in text-to-text generation scenarios and serves as both a learning resource and practical reference for reproducible experiments.

本仓库旨在备份和分享最近学习并实践成功的脚本，以及"文本生成文本"的Attention map简易绘图，主要关注有监督微调和强化学习在数学推理任务中的应用。所有代码均经过测试验证。

⚠️ **Note / 注意**  
Only key code snippets are shared here. Some training code and results are temporarily not publicly available. The existing code is sufficient for learning and communication purposes. Feel free to contact me if you have any questions.

注意，只收集了部分关键的代码，部分训练代码与训练结果暂不公开，已有的代码已足够学习交流使用，有疑问可以与我联系。

## 🏷️ Keywords / 关键词

- **Supervised Fine-Tuning (SFT)** / 有监督微调
- **Reinforcement Learning (RL)** / 强化学习  
- **Qwen3** / 通义千问3
- **GSM8K** (Grade School Math 8K)
- **CMATH** (Chinese Math Dataset)
- **Mathematical Reasoning** / 数学推理
- **Large Language Models** / 大语言模型

## 📁 Project Structure / 项目结构
```
.
├── README.md
├── figure
│   ├── attention_map_layer_0.svg
│   ├── attention_map_layer_18.svg
│   └── attention_map_layer_35.svg
├── model
│   └── model_download.py
├── results
│   └── evaluation_details_deepseek-math-7b-base_cmath_official_corrected.json
└── scripts
    ├── check_local_model.py
    ├── dpo_on_distilabelmathpreferencedpo.py
    ├── dpo_on_mathdpopairs.py
    ├── evaluate_CMATH.py
    ├── evaluate_GSM8K.py
    ├── merge_dpo_model.py
    ├── sft_on_gsm8k_ds.py
    ├── sft_on_gsm8k_qwen3_0.6B.py
    ├── sft_on_gsm8k_qwen3_4B.py
    └── visualize_attention.py
```
## 📊 Datasets / 数据集

- **GSM8K**: Grade School Math 8K - English math word problems  
  [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/openai/gsm8k)

- **CMATH**: Chinese Math Dataset - Chinese math problems for evaluation  
  [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/weitianwen/cmath) • 
  [![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/XiaoMi/cmath)

- **distilabel-math-preference-dpo**:  
  [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/argilla/distilabel-math-preference-dpo)

- **Math-Step-DPO-10K**:  
  [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/xinlai/Math-Step-DPO-10K)

## 📜 License / 许可证

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 Acknowledgments / 致谢

- Thanks to the Qwen team for the open-source models
- Appreciation to the GSM8K and CMATH dataset creators
- Inspired by various research papers and open-source projects in the field, especially:  
  **"DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"**  
  [![arXiv](https://img.shields.io/badge/arXiv-2402.03300-b31b1b.svg)](https://arxiv.org/abs/2402.03300)

- 感谢 Qwen 团队开源模型
- 感谢 GSM8K 和 CMATH 数据集的创建者
- 灵感来源于该领域的多篇研究论文和开源项目，特别是：  
  **"DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"**  
  [![arXiv](https://img.shields.io/badge/arXiv-2402.03300-b31b1b.svg)](https://arxiv.org/abs/2402.03300)

## 📧 Contact / 联系方式

For questions or suggestions, please open an issue.

如有问题或建议，请提交 issue。