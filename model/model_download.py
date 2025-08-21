# 从 ModelScope 下载模型
# 这种下载方式可能需要手动移动文件位置

from modelscope.hub.snapshot_download import snapshot_download
import os

# 创建一个名为 'Qwen3-4B' 的目录来存放模型文件
model_dir_base = './Qwen3-4B'
if not os.path.exists(model_dir_base):
    os.makedirs(model_dir_base)

# 从 ModelScope 下载模型，并指定缓存目录
model_dir = snapshot_download('Qwen/Qwen3-4B', cache_dir=model_dir_base)

print(f"模型已下载到: {model_dir}")