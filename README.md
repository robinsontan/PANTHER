# PANTHER: Generative Pretraining Beyond Language for Sequential User Behavior Modeling

## Description
This repository contains the implementation of PANTHER and the scripts for the experiments on credit card transaction dataset, including:
- Data preprocessing pipelines
- Pre-training on CCT behavior sequence data
- Fraud detection model training

## Dataset
The raw dataset is available on Kaggle:
[Credit Card Transactions Dataset](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset)

## Prerequisites
- Python 3.8+
- Required packages (install via `pip install -r requirements.txt`)

## Workflow

### 1. Data Preprocessing
```bash
python cct_preprocess.py
```
Configure the raw data path in `cct_preprocess.py` before running.

### 2. Model Pretraining
Pretrain PANTHER on CCT behavior sequence data:
```bash
export CONFIG_GROUPS=credit-card-transactions
export CONFIG=tf_patternrec_v5

python train_behseq.py \
    --gin_config_file="configs/${CONFIG_GROUPS}/${CONFIG}.gin" \
    --model_save_path="./ckpt/${CONFIG_GROUPS}/${CONFIG}" \
    2>&1 | tee "logs/${CONFIG_GROUPS}/${CONFIG}.log"
```

### 3. Generate Embeddings
Extract pretrained embeddings for downstream tasks:
```bash
python inference.py \
    --gin_config_file="configs/${CONFIG_GROUPS}/${CONFIG}.gin" \
    --model_save_path="./ckpt/${CONFIG_GROUPS}/${CONFIG}"
```

### 4. Fraud Detection Model
Train the DCN fraud detection model:
```bash
python cct_fraud_detection.py
```
