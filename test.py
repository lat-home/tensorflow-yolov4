import os
import cv2
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
from core.yolov4 import YOLOv4
import glob

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


ckpt_file = './checkpoint/yolov4_loss=27.0843.ckpt-13'                               # trained weights
detection_result_dir = './texts/yolo_m_bigger_waymo_resized_data_800_epoch13'        # output as txt file
conf_scr = 0.05                                                                      # confidence score


input_dir = 'datapath/val'                                                           # path to input images
val_names = glob.glob(f"{input_dir}/*.jpg")
classes = utils.read_class_names(cfg.YOLO.CLASSES)
num_classes = len(classes)


class YOLOv4(object):
    def __init__(self):
        self.input_size       = cfg.TEST.INPUT_SIZE
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes          = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes      = len(self.classes)
        self.anchors          = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.iou_threshold    = cfg.TEST.IOU_THRESHOLD
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.weight_file      = ckpt_file

        with tf.name_scope('input'):
            self.input_data = tf.placeholder(dtype=tf.float32, name='input_data')
            self.trainable  = tf.placeholder(dtype=tf.bool,    name='trainable')

        model = YOLO_M_Bigger(self.input_data, self.trainable)
        self.pred_sbbox, self.pred_mbbox, self.pred_lbbox = model.pred_sbbox, model.pred_mbbox, model.pred_lbbox

        with tf.name_scope('ema'):
            ema_obj = tf.train.ExponentialMovingAverage(self.moving_ave_decay)

        self.sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.saver = tf.train.Saver(ema_obj.variables_to_restore())
        self.saver.restore(self.sess, self.weight_file)

    def predict(self, image):
        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape

        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run(
            [self.pred_sbbox, self.pred_mbbox, self.pred_lbbox],
            feed_dict={
                self.input_data: image_data,
                self.trainable: False
            }
        )

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)


        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, conf_scr)
        bboxes = utils.nms(bboxes, self.iou_threshold)

        f = open(os.path.join(detection_result_dir, fname.replace('.jpg', '.txt')), "w")
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = np.array(bbox[:4], dtype=np.int32)
            score = bbox[4]
            cls_ind = int(bbox[5])
            cls = classes[int(cls_ind)]
            f.write('{} {:.6f} {} {} {} {}\n'.format(cls.replace(' ', '_'), score, x1, y1, x2, y2))

        return org_image


if __name__ == '__main__':
    test = YoloTest()

    if not os.path.isdir(detection_result_dir):
        os.makedirs(detection_result_dir)

    filenames = []

    for name in val_names:
        img_basename = os.path.basename(name)
        img_onlyname = os.path.splitext(img_basename)
        filenames.append(img_onlyname[0])

    i=0
    for fname in filenames:
        i+=1
        fname += ".jpg"
        img = cv2.imread(os.path.join(input_dir, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        det = test.predict(img)
        print(i)
