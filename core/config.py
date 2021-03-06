#! /usr/bin/env python
# coding=utf-8


from easydict import EasyDict as edict


__C                             = edict()
# Consumers can get config by: from config import cfg

cfg                             = __C

# YOLO options
__C.YOLO                        = edict()

# Set the class name
__C.YOLO.CLASSES                = "./data/classes/waymo.names"
__C.YOLO.ANCHORS                = "./data/anchors/yolov4_waymo_800_anchors.txt"


__C.YOLO.MOVING_AVE_DECAY       = 0.9995
__C.YOLO.STRIDES                = [8, 16, 32]
__C.YOLO.XYSCALES               = [1.05, 1.05, 1.05]
__C.YOLO.ANCHOR_PER_SCALE       = 3
__C.YOLO.IOU_LOSS_THRESH        = 0.5
__C.YOLO.UPSAMPLE_METHOD        = "resize"

# Train options
__C.TRAIN                       = edict()
__C.TRAIN.ANNOT_PATH            = "./data/dataset/waymo_train.txt"


# __C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.INPUT_SIZE            = [800]
__C.TRAIN.DATA_AUG              = True
__C.TRAIN.BATCH_SIZE            = 3
__C.TRAIN.LEARN_RATE_INIT       = 1e-4
__C.TRAIN.LEARN_RATE_END        = 1e-6
__C.TRAIN.WARMUP_EPOCHS         = 1
__C.TRAIN.FISRT_STAGE_EPOCHS    = 1
__C.TRAIN.SECOND_STAGE_EPOCHS   = 30

__C.TRAIN.INITIAL_WEIGHT        = "./checkpoint/yolov4_waymo_loss=24.4961.ckpt-10"              # pre-trained weights

# TEST options
__C.TEST                        = edict()
__C.TEST.ANNOT_PATH             = "./data/dataset/waymo_val.txt"
__C.TEST.BATCH_SIZE             = 2
__C.TEST.INPUT_SIZE             = 800
__C.TEST.DATA_AUG               = False
__C.TEST.WRITE_IMAGE            = True
__C.TEST.WRITE_IMAGE_PATH       = "./data/detection/"
__C.TEST.WRITE_IMAGE_SHOW_LABEL = True
__C.TEST.WEIGHT_FILE            = "./checkpoint/yolov3_test_loss=9.2099.ckpt-5"
__C.TEST.SHOW_LABEL             = True
__C.TEST.SCORE_THRESHOLD        = 0.3
__C.TEST.IOU_THRESHOLD          = 0.45






