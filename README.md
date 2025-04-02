#  Server Setup Guide (Personal Use, for Vast.ai)

## 1. SSH Public Key

Add the following SSH public key to server key list:

```bash
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKJhes/9LAx0dQ882EdAuA3G1+pfW5k6ovpudq7aKsAh liyj@DESKTOP-LOH1NAO
```

## 2. Clone the GitHub Repository

```bash
cd /workspace
git clone https://github.com/KFCCrazzzyThursday/Janus_fine_tuning.git
```

## 3. Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
rm ./Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

## 4. Create Environment and Install Dependencies

```bash
conda create -n Janus python==3.10
conda activate Janus

cd /workspace/Janus_fine_tuning/Janus

pip install -e .
pip install transformers==4.38.2
pip install gpustat ipykernel

watch --color -n 1 gpustat --color
```

## 5. SSH Key Setup for GitHub

```bash
git config --global user.email "liyj323@mail2.sysu.edu.cn"
git config --global user.name "liyj"  
# Only if you haven't added SSH remote already
git remote set-url origin git@github.com:KFCCrazzzyThursday/Janus_fine_tuning.git

ssh-keygen -t ed25519 -C "liyj323@mail2.sysu.edu.cn"
cat ~/.ssh/id_ed25519.pub
```

> 🔐 Copy the output and add it to your GitHub SSH keys:  
> https://github.com/settings/keys

## 6. Jupyter Notebook with VSCode

Install Python and Jupyter extensions on VSCode. Then
```
python -m ipykernel install --user --name Janus --display-name "Python (Janus)"
```

## 7.TensorBoard Tips
Refer to this [Github issue](https://github.com/tensorflow/tensorboard/issues/6808).
```
pip install tensorboard
pip install protobuf==4.25
```