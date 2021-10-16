from detectron2.engine import DefaultTrainer
from detectron2.data import transforms as T
from detectron2.data import build_detection_train_loader
from pannukedetectron2.mapper import CustomMapper
from pannukedetectron2 import labels as pannuke_labels
from detectron2.data import MetadataCatalog, DatasetCatalog

class PannukeTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=CustomMapper())

def get_dataset(coco_datadict):
    return coco_datadict

def register(fold: int, coco_datadict):
    dataset_name = f'pannuke_fold{fold}'
    DatasetCatalog.register(dataset_name, lambda d=coco_datadict: get_dataset(coco_datadict))
    MetadataCatalog.get(dataset_name).set(thing_classes=list(pannuke_labels.LABEL_MAP.keys()))
    coco_metadata = MetadataCatalog.get(dataset_name)
    return coco_metadata