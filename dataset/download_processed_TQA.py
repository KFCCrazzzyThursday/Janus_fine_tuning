import os
import tarfile
from huggingface_hub import hf_hub_download

# 下载 .tar.gz 文件（自动保存到 cache 目录）
tar_path = hf_hub_download(
    repo_id="Billyshears/TQA",
    filename="TQA.tar.gz",
    repo_type="dataset"
)

# 解压到目标目录
extract_dir = "/workspace/Janus_fine_tuning/dataset/unpacked_dataset/TQA"
os.makedirs(extract_dir, exist_ok=True)

with tarfile.open(tar_path, "r:gz") as tar:
    tar.extractall(path=extract_dir)

print(f"Done: {extract_dir}")
