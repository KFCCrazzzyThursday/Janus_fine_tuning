# download_unpack_TQA.py

import os
import json
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
import concurrent.futures

def main():
    # 1) 从HF Hub下载，缓存到 /workspace/Janus_fine_tuning/dataset/raw_dataset/
    ds = load_dataset("yyyyifan/TQA", cache_dir="/workspace/Janus_fine_tuning/dataset/raw_dataset/TQA")

    # 2) 创建输出目录（unpacked_dataset/ 下存放解包后的文件）
    out_dir = "/workspace/Janus_fine_tuning/dataset/unpacked_dataset/TQA"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

    # ========== 多线程任务函数 ==========
    def process_example(example, idx, split_name):
        """
        单条样本处理逻辑：保存图像到磁盘并构造记录 dict 返回。
        """
        img_obj = example["image"]
        if img_obj is not None:
            img_path = os.path.join(out_dir, "images", f"{split_name}_{idx}.png")
            img_obj.save(img_path)
        else:
            img_path = None

        record = {
            "id": f"{split_name}_{idx}",
            "question": example["question"],
            "options": example["options"],
            "answer": example["answer"],
            "image_path": img_path
        }
        return record

    # ========== 导出函数 (多线程) ==========
    def export_split(split_ds, split_name):
        """
        将 split_ds 中的所有样本并行处理后，写入 JSON Lines 文件和对应的 PNG 图片。
        """
        jsonl_path = os.path.join(out_dir, f"{split_name}.jsonl")

        futures = []
        with open(jsonl_path, "w", encoding="utf-8") as f_json, \
                concurrent.futures.ThreadPoolExecutor(max_workers=48) as executor:
            # 提交多线程任务
            for idx, example in enumerate(split_ds):
                futures.append(executor.submit(process_example, example, idx, split_name))

            # 等待任务完成，并将返回的 record 写入 JSONL
            # tqdm 用来显示整体进度
            for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures),
                                            total=len(futures),
                                            desc=f"Exporting {split_name}")):
                record = future.result()

                # 在这里增加一句简单的打印，告诉用户处理到了第几条
                # 如果觉得过多，可以在一定间隔下再打印
                print(f"[{split_name}] 已完成 {i+1}/{len(futures)} 条")

                f_json.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Done => {jsonl_path}")

    # 3) 分别导出 train/val/test
    export_split(ds["train"], "train")
    export_split(ds["val"], "val")
    export_split(ds["test"], "test")

if __name__ == "__main__":
    main()
