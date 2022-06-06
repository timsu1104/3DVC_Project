# 3D Visual Computing - Final Project
### Author
Zhengyuan Su, 2021010812, Yao Class 12, Tsinghua University

## Introduction
This repo contains source code of Final Project of the course 3D Visual Computing 2022. 

## Step 1. Preparing Data
To prepare the data, we generally obey the process of origin. Here are the steps: 

1. Download the data from the course webdisk. Assume we have already been mounted at the root directory of this project. Then run the commands below to download the data. 

```bash
cd datasets
wget https://cloud.tsinghua.edu.cn/seafhttp/files/64616fee-3f4f-49c8-992d-d6d5e5e3e0da/testing_data.zip
wget https://cloud.tsinghua.edu.cn/seafhttp/files/3b325cde-8f9a-43f5-bbd2-1b0977b6ee93/training_data_1.zip
wget https://cloud.tsinghua.edu.cn/seafhttp/files/4ba9c959-de03-4653-b28d-6333a14da020/training_data_2.zip
wget https://cloud.tsinghua.edu.cn/seafhttp/files/a51b63d4-7e04-4b6a-97bd-086d4f9a1700/training_data_3.zip
wget https://cloud.tsinghua.edu.cn/seafhttp/files/27be6d81-0bb7-4a94-bbe7-695a368c46ed/training_data_4.zip
wget https://cloud.tsinghua.edu.cn/seafhttp/files/9220db99-b2d8-4a0a-8e7c-502bc1dc9d32/training_data_5.zip
cd ..
```

2. Extract and aggregate data 

```bash
cd datasets
# Unzip
for zip in *.zip; do unzip -q $zip; done 
# Aggregate
mkdir training_data
mkdir training_data/data
cp -r training_data_1/split training_data
for i in {1..5}; do find training_data_$i/data/ -name "*" | xargs -i cp -r {} training_data/data/; done
cd ..
```

3. Process the data to fit the dataloader input. 

```bash
cd datasets
python prepare_data.py
cd ..
```

## Step 2. Environment Requirement
The code is tested on PyTorch 1.9.0+cu111. 

## Step 3. Training
To train the model, run
```bash
python scripts/train.py
```
To evaluate on test set, run
```bash
python scripts/test.py
```