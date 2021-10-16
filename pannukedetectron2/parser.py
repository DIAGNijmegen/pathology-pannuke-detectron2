import numpy as np
from wholeslidedata.annotation import AnnotationStructure, annotation_factory

from pannukedetectron2 import labels as pannuke_labels
from pathlib import Path
from detectron2.structures import BoxMode
import cv2

def get_bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [rmin, rmax, cmin, cmax]

def pannuke_bounding_box_parser(pannuke_data_path: str, fold: int):
    """ Expected data folder structure
        
        pannuke_data:
            fold1:
                images:
                    fold1_image0.png
                    fold1_image1.png
                    ...
                masks:
                    fold1_masks.npy
            fold2:
                ...
            fold3:
                ...

    Args:
        pannuke_data_path (str): path to pannuke_data
        fold (int): pannuke fold index


    Returns:
        dict: a mapping of images and corresponding list of annotations
    """


    masks_path = Path(pannuke_data_path) /  f'fold_{fold}' / f'Fold_{fold}' / 'masks' / f'fold{fold}' / 'masks.npy'
    masks = np.load(masks_path, mmap_mode="r")
    data = {}
    for idx, mask in enumerate(masks):
        filename = Path(pannuke_data_path) /  f'fold_{fold}' / f'Fold_{fold}' / 'images' /  f'fold{fold}' / 'images' / f"pannuke_fold{fold}_image_{idx}.png"
        annotations = []
        for category_id in range(mask.shape[-1]):
            inst_map = mask[..., category_id]
            inst_list = list(np.unique(inst_map))  # get list of instances

            if 0 in inst_list:
                inst_list.remove(0)  # remove background

            for inst_id in inst_list:
                inst_map_mask = np.array(
                    inst_map == inst_id, np.uint8
                )  # get single object
                y1, y2, x1, x2 = get_bounding_box(inst_map_mask)
                y1 = y1 - 2 if y1 - 2 >= 0 else y1
                x1 = x1 - 2 if x1 - 2 >= 0 else x1
                x2 = x2 + 2 if x2 + 2 <= inst_map.shape[1] - 1 else x2
                y2 = y2 + 2 if y2 + 2 <= inst_map.shape[0] - 1 else y2

                coords = [[x1, y1], [x1, y2], [x2, y1], [x2, y2]]

                structure = AnnotationStructure(
                    masks_path,
                    len(annotations) + 1,
                    "polygon",
                    pannuke_labels.INV_LABEL_MAP[category_id].lower(),
                    coords,
                    [],
                )
                annotations.append(annotation_factory(structure, pannuke_labels.LABEL_MAP))

        data[str(filename)] = annotations
    return data


def convert_annotations_to_coco_bbox(data):    
    '''
    data should be a dict with {image_path: [annotattions]}
    '''
    coco_dataset = []
    for image_index, (image_path, annotations) in enumerate(data.items()):
        boxes = []
        for annotation in annotations:
           

            boxes.append({
                "bbox_mode": BoxMode.XYXY_ABS,
                'category_id': annotation.label.value,
                'bbox': annotation.bounds,
            })

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f'cant load image {image_path}')
            
        height, width = cv2.imread(image_path).shape[:2]
        record = {}
        record["file_name"] = image_path
        record["image_id"] = image_index
        record["height"] = height
        record["width"] = width
        record["annotations"] = boxes
        coco_dataset.append(record)    

    return coco_dataset


def get_pannuke_coco_datadict(pannuke_data_path, fold):
    pannuke_parsed_data = pannuke_bounding_box_parser(pannuke_data_path=pannuke_data_path, fold=fold)
    coco_dataset = {}
    coco_dataset.update(pannuke_parsed_data)
    coco_datadict = convert_annotations_to_coco_bbox(coco_dataset)
    return coco_datadict