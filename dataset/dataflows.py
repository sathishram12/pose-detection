import os
import cv2
import numpy as np
import functools
import tensorflow as tf
from tensorpack.dataflow import MultiProcessMapDataZMQ, TestDataSpeed
from tensorpack.dataflow.common import MapData

#import sys
#sys.path.append("../")
from dataset.augmentors import CropAug, FlipAug, ScaleAug, RotateAug, ResizeAug
from dataset.base_dataflow import CocoDataFlow, JointsLoader
from dataset.dataflow_steps import create_all_mask, augment, read_img, apply_mask, gen_mask


def build_sample(components):
    """
    Builds a sample for a model.

    :param components: components
    :return: list of final components of a sample.
    """
    return [components[10],
            components[13]]


def get_dataflow(annot_path, img_dir, strict, x_size = (324,324)):
    """
    This function initializes the tensorpack dataflow and serves generator
    for training operation.

    :param annot_path: path to the annotation file
    :param img_dir: path to the images
    :return: dataflow object
    """
    coco_crop_size = 368

    # configure augmentors

    augmentors = [
        ScaleAug(scale_min=0.8,
                 scale_max=2.0,
                 target_dist=0.8,
                 interp=cv2.INTER_CUBIC),

        RotateAug(rotate_max_deg=30,
                  interp=cv2.INTER_CUBIC,
                  border=cv2.BORDER_CONSTANT,
                  border_value=(128, 128, 128), mask_border_val=1),

        #CropAug(coco_crop_size, coco_crop_size, center_perterb_max=40, border_value=128,
        #        mask_border_val=1),

        FlipAug(num_parts=17, prob=0.5),

        ResizeAug(x_size[1], x_size[0])

    ]

    # prepare augment function

    augment_func = functools.partial(augment,
                                     augmentors=augmentors)

    # prepare building sample function

    build_sample_func = functools.partial(build_sample)

    df = CocoDataFlow((coco_crop_size, coco_crop_size), annot_path, img_dir)
    df.prepare()
    size = df.size()
    df = MapData(df, read_img)
    df = MapData(df, augment_func)
    df = MultiProcessMapDataZMQ(df, num_proc=4, map_func=build_sample_func, buffer_size=200, strict=strict)

    return df, size


if __name__ == '__main__':
    """
    Run this script to check speed of generating samples. Tweak the nr_proc
    parameter of PrefetchDataZMQ. Ideally it should reflect the number of cores 
    in your hardware
    """
    
    batch_size = 10
    curr_dir = os.path.dirname(__file__)
    annot_path = os.path.join(curr_dir, '../../data/coco2017/annotations/person_keypoints_train2017.json')
    img_dir = os.path.abspath(os.path.join(curr_dir, '../../data/coco2017/train2017/'))

    df1, size1 = get_dataflow(annot_path, img_dir, False, x_size=(324,324))

    TestDataSpeed(df1, size=100).start()
