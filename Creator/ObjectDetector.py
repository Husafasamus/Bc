import torch

from matplotlib import pyplot as plt
import matplotlib.patches as patches
from typing import Iterable

class BBox:

    def __init__(self, x: float, y: float, w: float, h: float) -> None:
        """

            :param x:
            :param y:
            :param w:
            :param h:
            """
        # Kontrola zapornych cisiel?
        self.x, self.y, self.w, self.h = x, y, w, h


class Detection:

    def __init__(self, bbox: BBox, confidence: float, label: str):
        self.bbox = bbox
        self.confidence = confidence
        self.label = label


class ObjectDetector:
    """
    This class do somethign awesome, but not yet.
    """
    pass


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

    def start(self, uris: Iterable, const_confidence: float) -> 'Detection':
        """
        return type class Detection[] -> bbboxes + confidences + labels
        """
        detections = []
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
                    (Detection(BBox(x, y, w, h), confidences[idx] * 100, self.classes_to_labels[classes[idx] - 1]))
            detections.append(bboxes_on_picture)

        return detections
