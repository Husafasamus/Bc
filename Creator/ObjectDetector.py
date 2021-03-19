import torch

from Creator import bbox
from typing import Iterable


class Annotation:

    """
    Labels -> [bicycle, car, motorcycle, bus, truck]
    """

    def __init__(self, bbox: bbox.BBox, confidence: float, label: str) -> None:
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
    This class do somethign awesome, but not yet.
    """
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
                    (Annotation(bbox.BBox(x, y, w, h), confidences[idx] * 100,
                                self.classes_to_labels[classes[idx] - 1]))
            annotations.append(bboxes_on_picture)
        return annotations
