import torchvision
import torch
import argparse
import cv2
import object_detectors.faster_rcnn.detect_utils
from PIL import Image

class FasterRCNNMobilnetv3:

    @staticmethod
    def detect_vehicles_in_dataset(paths, conf_thres=0.8) -> 'Annotations':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # load the model
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            pretrained=True)  # Pretrained True, means that the model is pretrained on the MS COCO 2017
        # load the model on to the computation device
        model.eval().to(device)

        annos = []
        for img_path in paths:
            image = Image.open(img_path)
            boxes, classes, labels = object_detectors.faster_rcnn.detect_utils.predict(image, model, device, conf_thres)
            annos.append(object_detectors.faster_rcnn.detect_utils.create_annotaions(boxes, classes, conf_thres, labels, image))

        return annos

    def detect_img(self, img, conf_thres=0.8):
        # construct the argument parser
        parser = argparse.ArgumentParser()
        args = vars(parser.parse_args())
        # define the computation device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load the model
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True) # Pretrained True, means that the model is pretrained on the MS COCO 2017
        # load the model on to the computation device
        model.eval().to(device)

        # read the image and run the inference for detections
        image = Image.open(args['input'])
        boxes, classes, labels = object_detectors.faster_rcnn.detect_utils.predict(image, model, device, 0.9)
        image = object_detectors.faster_rcnn.detect_utils.draw_boxes(boxes, classes, labels, image)
        cv2.imshow('Image', image)
        save_name = f"image"
        cv2.imwrite(f"outputs/{save_name}.jpg", image)
        cv2.waitKey(0)