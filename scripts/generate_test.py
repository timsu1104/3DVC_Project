import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, random, torch, glob, sys
from PIL import Image
from tqdm import tqdm

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.engine import DefaultTrainer

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

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 4  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 10000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 79  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, 'retina_net')
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

print("Finish_loading_model")
dataset_dicts = get_data_test()
preds = []
print("Begin Loading data")
for d in tqdm(dataset_dicts):
    im = np.array(Image.open(d["file_name"])) # / 255
    outputs = predictor(im)
    boxes, scores, pred_labels = outputs['instances'].pred_boxes, outputs['instances'].scores, outputs['instances'].pred_classes
    boxes = np.array(boxes.tensor.cpu()).tolist()
    scores = np.array(scores.cpu()).tolist()
    pred_labels = np.array(pred_labels.cpu()).tolist()
    print(scores)
    preds.append([boxes, pred_labels, scores])

with open(os.path.join('datasets/2Dproposal.json'), "w") as f:
    json.dump(preds, f)