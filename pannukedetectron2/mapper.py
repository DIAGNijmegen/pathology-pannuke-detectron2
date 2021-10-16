from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

import copy
import torch
from dataaugmentation.augmentation import image_augmentations, SampleAugmentor
import numpy as np


class CustomMapper:
    def __init__(self):
        self._sa = SampleAugmentor(augmentations=image_augmentations())

    def __call__(self, dataset_dict):
        # Implement a mapper, similar to the default DatasetMapper, but with your own customizations

        # it will be modified by code below
        dataset_dict = copy.deepcopy(dataset_dict)

        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        transform_list = [
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        ]
        image, transforms = T.apply_transform_gens(transform_list, image)


        dummy_y = np.empty((1, 1))

        image, dummy_y = self._sa(image, dummy_y)
        self._sa.randomize()

        dataset_dict["image"] = torch.as_tensor(
            image.transpose(2, 0, 1).astype("float32")
        )

        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(annos, image.shape[:2])
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict
