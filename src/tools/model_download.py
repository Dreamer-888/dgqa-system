import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graphrag.settings import apply_runtime_environment

apply_runtime_environment()

from huggingface_hub import snapshot_download

def download_hf_model():
    # 配置要下载的模型列表 (追加了 bge-m3)
    models_to_download = [
        {"repo_id": "BAAI/bge-m3", "local_dir": "./models/bge-m3"},
        {"repo_id": "BAAI/bge-reranker-base", "local_dir": "./models/bge-reranker-base"}
    ]
    
    for model in models_to_download:
        repo_id = model["repo_id"]
        local_dir = model["local_dir"]
        
        print(f"\n==================================================")
        print(f"开始从镜像站下载模型: {repo_id}")
        print(f"目标本地路径: {os.path.abspath(local_dir)}")
        print(f"==================================================")
        
        start_time = time.time()
        try:
            # 执行断点续传下载
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,  # 禁用符号链接
                resume_download=True,          # 开启断点续传
                ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.flax"] # 忽略不需要的框架权重
            )
            
            elapsed_time = time.time() - start_time
            print(f"🎉 {repo_id} 下载成功！耗时: {elapsed_time:.2f} 秒")
            
        except Exception as e:
            print(f"❌ {repo_id} 下载过程中遇到错误: {e}")
            print("提示: 可能是网络偶发波动，请重新运行本脚本尝试断点续传。")

if __name__ == "__main__":
    download_hf_model()
