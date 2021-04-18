import torch

from dataset_merger.bbox import BBox
from typing import Iterable
from dataset_merger.dataset import *
from object_detectors.yolov3 import *
from object_detectors.yolov3  import detect
from dataset_merger.merging import DatasetMerger


class Annotation:

    def __init__(self, bbox: BBox, confidence: float, label: str) -> None:
        self.bbox, self.confidence, self.label = bbox, confidence, label

    def build_dictionary(self) -> dict:
        x, y, w, h = self.bbox.as_xywh()
        return {
            'bbox': {
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h)
            },
            'label': self.label,
            'confidence': self.confidence
        }

    def __repr__(self) -> str:
        return f'{self.label}, confidence={self.confidence}, {self.bbox.__repr__()}'


class ObjectDetector:
    """
    This class do something awesome, but not yet.
    Funkcionalita:
    1.

    """
    @staticmethod
    def create_files(dataset: Dataset, detector: str) -> None:
        path_dst = dataset.path.joinpath('Our_detections')
        path_dst.mkdir(exist_ok=True)
        path_dst = path_dst.joinpath(detector)
        path_dst.mkdir(exist_ok=True)
        path_dst.joinpath('detections.json').touch(exist_ok=True)

    @staticmethod
    def yolov3_vehicle_detector(dataset: Dataset):
        print('Yolov3 vehicle detector.')
        ObjectDetector.create_files(dataset, 'yolov3')
        annos = detect.detect_vehicles_in_dataset(dataset.path)

        annos = DatasetMerger.create_dict_from_annotations_detected(dataset, annos)
        DatasetMerger.file_write(dataset.path.joinpath('Our_detections').joinpath('yolov3').joinpath('detections.json'), annos, 4)


        pass

    @staticmethod
    def yolov3_LPN_detector(dataset: Dataset):
        print('Yolov3 LPN detector.')
        ObjectDetector.create_files(dataset, 'yolov3')
        pass

    @staticmethod
    def SSD_vehicle_detector(dataset: Dataset):
        print('SSD vehicle detector.')
        ObjectDetector.create_files(dataset, 'ssd')
        annos = SSD300()
        annos = annos.start(dataset.imgs_path, 0.8)
        annos = DatasetMerger.create_dict_from_annotations_detected(dataset, annos)

        DatasetMerger.file_write(dataset.path.joinpath('Our_detections').joinpath('ssd').joinpath('detections.json'), annos, 4)
        pass

    @staticmethod
    def SSD_LPN_detector(dataset: Dataset):
        print('SSD LPN detector.')
        ObjectDetector.create_files(dataset, 'ssd')


        pass





Annotations = list[list[Annotation]]


class SSD300:
    """
    Labels -> [bicycle, car, motorcycle, bus, truck]
    """

    def __init__(self):
        self.precision = 'fp32'
        self.ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=self.precision)
        self.utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
        self.ssd_model.to('cuda')
        self.ssd_model.eval()
        self.classes_to_labels = self.utils.get_coco_object_dictionary()

    def start(self, uris: Iterable, const_confidence: float) -> Annotations:
        """
        return type class Detection[] -> bbboxes + confidences + labels
        """
        annotations = []
        inputs = [self.utils.prepare_input(uri) for uri in uris]
        tensor = self.utils.prepare_tensor(inputs, self.precision == 'fp16')
        with torch.no_grad():
            detections_batch = self.ssd_model(tensor)

        results_per_input = self.utils.decode_results(detections_batch)
        best_results_per_input = [self.utils.pick_best(results, const_confidence) for results in results_per_input]
        # best_results_per_input = [self.utils.pick_best(results, 0.40) for results in results_per_input]
        for image_idx in range(len(best_results_per_input)):
            bboxes, classes, confidences = best_results_per_input[image_idx]
            bboxes_on_picture = []
            for idx in range(len(bboxes)):
                left, bot, right, top = bboxes[idx]
                x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
                bboxes_on_picture.append \
                    (Annotation(BBox(int(x), int(y), int(w), int(h)), confidences[idx] * 100, self.classes_to_labels[classes[idx] - 1]))
            annotations.append(bboxes_on_picture)
        return annotations
