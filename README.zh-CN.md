# 🚀 个人服务器部署指南（Vast.ai）
[中文](README.zh-CN.md) | [English](README.md)
> 快速、简洁地在租用 GPU 服务器（如 Vast.ai）上部署深度学习环境。

---

## ✅ 1. 添加 SSH 公钥

将以下 SSH 公钥添加到服务器的 `authorized_keys` 中，用于远程 SSH 登录：

```bash
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKJhes/9LAx0dQ882EdAuA3G1+pfW5k6ovpudq7aKsAh liyj@DESKTOP-LOH1NAO
```

---

## ✅ 2. 克隆 GitHub 仓库

```bash
cd /workspace
git clone https://github.com/KFCCrazzzyThursday/Janus_fine_tuning.git
```

---

## ✅ 3. 安装 Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
rm ./Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

---

## ✅ 4. 创建 Conda 环境并安装依赖项

```bash
conda create -n Janus python=3.10 -y
conda activate Janus

cd /workspace/Janus_fine_tuning/Janus

pip install -e .
pip install transformers==4.38.2
pip install gpustat ipykernel
```

> 💡 实时监控 GPU 状态：

```bash
watch --color -n 1 gpustat --color
```

---

## ✅ 5. 配置 GitHub SSH 访问权限

```bash
git config --global user.email "liyj323@mail2.sysu.edu.cn"
git config --global user.name "liyj"

# 如果你尚未添加 SSH 远程地址：
git remote set-url origin git@github.com:KFCCrazzzyThursday/Janus_fine_tuning.git

ssh-keygen -t ed25519 -C "liyj323@mail2.sysu.edu.cn"
cat ~/.ssh/id_ed25519.pub
```

> 🔐 将生成的公钥内容复制到 GitHub：  
> [https://github.com/settings/keys](https://github.com/settings/keys)

---

## ✅ 6. 在 VSCode 中使用 Jupyter Notebook

在 VSCode 安装 Python 与 Jupyter 扩展插件后，执行以下命令以注册内核：

```bash
python -m ipykernel install --user --name Janus --display-name "Python (Janus)"
```

---

## ✅ 7. TensorBoard 安装与配置建议

某些版本的 TensorBoard 需锁定 `protobuf` 版本以避免错误：

```bash
pip install tensorboard
pip install protobuf==4.25
```

> 参考该 [GitHub issue](https://github.com/tensorflow/tensorboard/issues/6808)

---

## ✅ 8. 安装 FlashAttention

⚠️ **安装 FlashAttention 前，必须先安装 PyTorch 和 torchvision，否则会失败！**

```bash
pip install bitsandbytes accelerate
pip install flash-attn --no-build-isolation
```

---

## ✅ 9. 安装 HuggingFace 微调工具 `trl`

使用最新版 `trl` 进行强化学习微调（如 PPO）：

```bash
pip install git+https://github.com/huggingface/trl.git
```

---

## ✅ 10. 下载已处理数据集

```bash
cd /workspace/dataset
python download_processed_TQA.py
```

处理好的数据集路径为：

```
/workspace/Janus_fine_tuning/dataset/processed_dataset
```
