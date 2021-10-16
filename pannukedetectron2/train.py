from pannukedetectron2.customtrainer import PannukeTrainer
from pannukedetectron2 import labels as pannuke_labels
from pannukedetectron2.parser import get_pannuke_coco_datadict

from detectron2.config import get_cfg
import os
from pannukedetectron2.customtrainer import register

def train(data_folder, fold, output_dir):
    coco_datadict = get_pannuke_coco_datadict(data_folder, fold)
    register(fold, coco_datadict)
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    )
    cfg.DATASETS.TRAIN = ("detection_dataset2",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl"  # Let training initialize from model zoo
    # cfg.MODEL.WEIGHTS = None
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.001  # pick a good LR
    cfg.SOLVER.MAX_ITER = 200000  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        64  # faster, and good enough for this toy dataset (default: 512)
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(pannuke_labels.LABEL_MAP)
    cfg.OUTPUT_DIR = output_dir
    cfg.SOLVER.STEPS = (1000, 10000, 20000, 50000, 100000)
    cfg.SOLVER.WARMUP_ITERS = 0
    cfg.SOLVER.GAMMA = 0.5
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = PannukeTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
