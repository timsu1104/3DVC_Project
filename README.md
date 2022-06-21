# 3D Visual Computing - Final Project
### Author
Zhengyuan Su, 2021010812, Yao Class 12, Tsinghua University

## Introduction
This repo contains source code of Final Project of the course 3D Visual Computing 2022. 

I use Fast R-CNN to generate boxes and Frustum PointNet to segment the instances. 

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
python prepare_data.py --data_split train
python prepare_data_detection.py --data_split train
# python Aggregate_data.py --data_split train
# python Aggregate_data.py --data_split val

python prepare_data.py --data_split test
python prepare_data_detection.py --data_split test
# python Aggregate_data.py --data_split test
cd ..
```

## Step 2. Environment Requirement
The code is tested on PyTorch 1.9.0+cu111 with python 3.7.0. 

Detectron2 by facebookAIResearch is used for 2D detectron. To install it, run
```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
or to install locally, run
```bash
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```
Please refer to [Detectron2 0.6 documentation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) for more detail information. 

## Step 3. Training
The model consists of two part, a 2D detection part and a 3D segmentation part. Both are trained with full supervision. 

To train the model, run
```bash
python scripts/detection.py
python scripts/train.py --tag complexer_pointnet
```

Tensorboard monitoring is supported. To use it, run 
```bash
tensorboard --logdir exp/FrustumSegmentationNet --bind_all
tensorboard --logdir output --bind_all
```

To evaluate on test set, run
```bash
python scripts/generate_test.py > detection_output.log
python scripts/test.py --tag complexer_pointnet --epoch 32
```
Note: when training, passing a ```--toy``` can force the model to use a reduced set of data (100 for training and 20 for validation). 


## Step 4. Visualization
To visualize the result, run 
```bash
python scripts/visual.py --id $ID
```
where $ID should be replaced with the prefix of a specific testing data (e.g. 1-4-25). If $ID=```''```, then all testing result would be visualized. 
The output directory is ```visualization/```