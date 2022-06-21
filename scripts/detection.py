# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os, json, glob
from tqdm import tqdm

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.engine import DefaultTrainer

def get_data_train():
    dataset_dicts = []
    files = sorted(glob.glob(os.path.join('datasets/training_data/data/*_datas.json')))
    for fn in tqdm(files):
        with open(fn, 'r') as f:
            dict = json.load(f)
        dict["file_name"] = os.path.join('datasets', dict["file_name"])
        for obj in dict['annotations']:
            ymin, xmin, ymax, xmax = obj['bbox']
            obj['bbox'] = [xmin, ymin, xmax, ymax]
        dataset_dicts.append(dict)
    return dataset_dicts

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("train", )
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 20000
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 79 
cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, 'retina_net')

DatasetCatalog.register("train", get_data_train)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()