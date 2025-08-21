# 从 ModelScope 下载模型

from modelscope.hub.snapshot_download import snapshot_download
import os

# 指定模型下载到的目标目录
model_dir = './Qwen3-4B'

# 使用 local_dir 参数直接指定下载位置
model_path = snapshot_download('Qwen/Qwen3-4B', 
                              local_dir=model_dir,
                              local_dir_use_symlinks=False)  # 不使用符号链接，直接复制文件

print(f"模型已下载到: {model_path}")