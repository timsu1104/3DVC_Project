# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, random, torch, glob, sys
from tqdm import tqdm

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.engine import DefaultTrainer

print("Import complete.")

def get_data_train():
    dataset_dicts = []
    files = sorted(glob.glob(os.path.join('datasets/training_data/data/*_datas.json')))
    print("Begin loading.")
    for fn in tqdm(files):
        with open(fn, 'r') as f:
            dict = json.load(f)
            dict["file_name"] = os.path.join('datasets', dict["file_name"])
            dataset_dicts.append(dict)
    return dataset_dicts

def get_data_test():
    dataset_dicts = []
    files = sorted(glob.glob(os.path.join('datasets/testing_data/data/*_datas.json')))
    print("Begin loading.")
    for fn in tqdm(files):
        with open(fn, 'r') as f:
            dict = json.load(f)
            dict["file_name"] = os.path.join('datasets', dict["file_name"])
            dataset_dicts.append(dict)
    return dataset_dicts

DatasetCatalog.register("train", get_data_train)
DatasetCatalog.register("test", get_data_test)

print("Register Complete.")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 4  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 10000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 79  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
print("Configuration Complete.")

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
print("Begin Training.")
trainer.train()
