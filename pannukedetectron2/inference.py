from pannukedetectron2 import labels as pannuke_labels

from pannukedetectron2.parser import get_pannuke_coco_datadict
from detectron2 import model_zoo
from detectron2.config import get_cfg
import os
from pannukedetectron2.customtrainer import register
import argparse
from detectron2.engine import DefaultPredictor
from pathlib import Path
import csv
import cv2

def inference(weights_path, data_folder, fold, output_dir, threshold=0.4):
    coco_datadict = get_pannuke_coco_datadict(data_folder, fold)
    register(fold, coco_datadict)
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    )
    cfg.DATASETS.TRAIN = ("detection_dataset2",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 1
    
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.001  # pick a good LR
    cfg.SOLVER.MAX_ITER = 200000  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.SOLVER.STEPS = (1000, 10000, 20000, 50000, 100000)
    cfg.SOLVER.WARMUP_ITERS = 0
    cfg.SOLVER.GAMMA = 0.5
    
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        64  # faster, and good enough for this toy dataset (default: 512)
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(pannuke_labels.LABEL_MAP)
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    cfg.MODEL.WEIGHTS = os.path.join(weights_path)  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    inference_ouput = []
    
    header = ['image_id', 'x', 'y', 'confidence', 'label']
    negative_image_ids = 0

    output_file = Path(output_dir) / f'inference_output_fold{fold}.csv'
    
    with open(output_file, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        
        for idx, d in enumerate(coco_datadict):
            im = cv2.imread(d["file_name"])
            outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            pred_boxes = outputs['instances'].get('pred_boxes')
            scores= outputs['instances'].get('scores')
            classes = outputs['instances'].get('pred_classes')
            centers = pred_boxes.get_centers()
            for idx, center in enumerate(centers):
                x, y = center.cpu().detach().numpy()
                confidence = scores[idx].cpu().detach().numpy()
                label = pannuke_labels.INV_LABEL_MAP[int(classes[idx].cpu().detach())]
                writer.writerow((d["file_name"], x,y,confidence, label))
                
                

def run():
    # create argument parser
    argument_parser = argparse.ArgumentParser(description="Experiment")
    argument_parser.add_argument("--weights_path", required=True)
    argument_parser.add_argument("--output_dir", required=True)
    argument_parser.add_argument("--data_folder", required=True)
    argument_parser.add_argument("--fold", required=True)
    args = vars(argument_parser.parse_args())
    inference(weights_path=args['weights_path'], data_folder=args['data_folder'], fold=args['fold'], output_dir=args['output_dir'])
    
    

if __name__ == "__main__":
    run()