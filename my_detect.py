from __future__ import division

import threading

from models import *
from utils.utils import *
from utils.datasets import *

import os
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader

from torch.autograd import Variable



def delete_person(file_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, default='data/samples', help='path to dataset')
    parser.add_argument('--config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
    parser.add_argument('--weights_path', type=str, default='weights/yolov3.weights', help='path to weights file')
    parser.add_argument('--class_path', type=str, default='data/coco.names', help='path to class label file')
    parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
    parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
    opt = parser.parse_args()
    print(opt)

    cuda = torch.cuda.is_available() and opt.use_cuda

    os.makedirs('output', exist_ok=True)

    # Set up model
    model = Darknet(opt.config_path, img_size=opt.img_size)
    model.load_weights(opt.weights_path)

    if cuda:
        model.cuda()

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(ImageFolder(file_path, img_size=opt.img_size),
                            batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print ('\nPerforming object detection:')
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)


        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print('\t Thread %d Batch %d, Inference Time: %s' % (threading.get_ident(), batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths[0])
        img_detections.extend(detections)
        if detections[0] is not None:
            for detection in detections:
                x1, y1, x2, y2, conf, cls_conf, cls_pred = detection.numpy()[0,:]
                print ('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))
                if classes[int(cls_pred)] == 'person' and cls_conf.item() > 0.7 and abs(y2-y1) >= 128:
                    os.remove(img_paths[0])
                    print('delete')
                    break



# delete_person(file_path="/Users/rl/Desktop/data 2/wild/desert/sand")
# delete_person(file_path="data/samples")