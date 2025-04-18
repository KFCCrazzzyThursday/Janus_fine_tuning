# 🚀 Personal Server Setup Guide (Vast.ai)
[中文](README.zh-CN.md) | [English](README.md)
> A quick and clean setup for a deep learning server on Vast.ai, tailored for personal use.

---

## ✅ 1. Add SSH Public Key

Add the following public key to the server’s SSH key list:

```bash
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKJhes/9LAx0dQ882EdAuA3G1+pfW5k6ovpudq7aKsAh liyj@DESKTOP-LOH1NAO
```

---

## ✅ 2. Clone the GitHub Repository

```bash
cd /workspace
git clone https://github.com/KFCCrazzzyThursday/Janus_fine_tuning.git
```

---

## ✅ 3. Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
rm ./Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

---

## ✅ 4. Create Conda Environment & Install Dependencies

```bash
conda create -n Janus python=3.10 -y
conda activate Janus

cd /workspace/Janus_fine_tuning/Janus

pip install -e .
pip install transformers==4.38.2
pip install gpustat ipykernel
```

> 💡 To monitor GPU usage in real time:

```bash
watch --color -n 1 gpustat --color
```

---

## ✅ 5. Configure GitHub SSH Access

```bash
git config --global user.email "liyj323@mail2.sysu.edu.cn"
git config --global user.name "liyj"

# Set SSH remote if not already set:
git remote set-url origin git@github.com:KFCCrazzzyThursday/Janus_fine_tuning.git

ssh-keygen -t ed25519 -C "liyj323@mail2.sysu.edu.cn"
cat ~/.ssh/id_ed25519.pub
```

> 🔐 Copy the output and add it to your GitHub SSH keys:  
> [https://github.com/settings/keys](https://github.com/settings/keys)

---

## ✅ 6. Use Jupyter Notebook in VSCode

Install the Python and Jupyter extensions in VSCode, then run:

```bash
python -m ipykernel install --user --name Janus --display-name "Python (Janus)"
```

---

## ✅ 7. TensorBoard Setup Tip

To avoid version-related issues, lock the protobuf version:

```bash
pip install tensorboard
pip install protobuf==4.25
```

> See [this issue](https://github.com/tensorflow/tensorboard/issues/6808) for context.

---

## ✅ 8. FlashAttention Installation Notes

⚠️ **You must install PyTorch and torchvision first, or installation will fail!**

```bash
pip install bitsandbytes accelerate
pip install flash-attn --no-build-isolation
```

---

## ✅ 9. Install HuggingFace `trl` (for fine-tuning)

```bash
pip install git+https://github.com/huggingface/trl.git
```

---

## ✅ 10. Download the Preprocessed Dataset

```bash
cd /workspace/dataset
python download_processed_TQA.py
```

You’ll find the processed data at:

```
/workspace/Janus_fine_tuning/dataset/processed_dataset
```
