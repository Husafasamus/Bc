from dataset_merger.dataset import Dataset
from dataset_merger import object_detector as od
from dataset_merger.bbox import BBox

import pandas
from PIL import Image
import numpy as np


class CarsDataset(Dataset):
    """reference to: https://ai.stanford.edu/~jkrause/cars/car_dataset.html"""

    def __init__(self, path: str) -> None:
        super().__init__(path)

    def find_images(self) -> None:
        for x in self.path.joinpath('car_ims').iterdir():
            self.imgs_path.append(x)

    def get_labels(self) -> 'Annotations':
        """
        Reading .csv file
        delimiter = ';'
        example:
        relative_img_path;bbox_x1;bbox_y1;bbox_x2;bbox_y2;class;test
        car_ims/000001.jpg;112;7;853;717;1;0
        """
        df = pandas.read_csv(self.path.joinpath('cars_annos.txt'), delimiter=';')

        annotations = []
        for img in range(len(df)):
            bboxes_on_picture = [
                od.Annotation(
                    BBox(int(df['bbox_x1'][img]), df['bbox_y1'][img], df['bbox_x2'][img], df['bbox_y2'][img]), 1.0,
                    'car')]
            annotations.append(bboxes_on_picture)
        del df
        return annotations


class UFPRALPRDataset(Dataset):
    """"""

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.annotations_path = []

    def find_images(self) -> None:
        for dir_in_ds in self.path.iterdir():  # testing, training, validation
            for dir_in_sub_ds in dir_in_ds.iterdir():  # ---> track0091, track0092, track0093, ...
                for img_w_ants in dir_in_sub_ds.iterdir():  # ---> track0091[01].png, track0091[01].txt, ...
                    if img_w_ants.name.__contains__('.png'):
                        self.imgs_path.append(img_w_ants)
                    elif img_w_ants.name.__contains__('.txt'):
                        self.annotations_path.append(img_w_ants)

    def get_labels(self) -> 'Annotations':

        annotations = []
        for txt_file in self.annotations_path:
            annotations.append(self.get_annotation_from_ufpr_txt_files(txt_file))

        return annotations

    @staticmethod
    def get_annotation_from_ufpr_txt_files(path: str) -> 'Annotations':
        with open(path, 'r') as f:
            # txt = f.readlines()
            txt = [line.strip() for line in f.readlines()]
            picture_bboxes = [od.Annotation(
                BBox(int(txt[1].split(' ')[1]), int(txt[1].split(' ')[2]), int(txt[1].split(' ')[3]),
                     int(txt[1].split(' ')[4])), 1.0, txt[2].split(' ')[1]), od.Annotation(
                BBox(int(txt[7].split(' ')[1]), int(txt[7].split(' ')[2]), int(txt[7].split(' ')[3]),
                     int(txt[7].split(' ')[4])), 1.0, 'license_plate')]

        return picture_bboxes


class ArtificialMercosurLicensePlates(Dataset):
    """
    Path: D:\Downloads\nx9xbs4rgx-2
    License plates dataset
    """

    def __init__(self, path: str) -> None:
        super().__init__(path)

    def find_images(self) -> None:
        df = pandas.read_csv(self.path.joinpath('dataset.csv'), delimiter=',')
        self.imgs_path = [
            self.path.joinpath('images').joinpath(img) for img in df['image']
        ]
        del df

    def get_labels(self) -> 'Annotations':
        df = pandas.read_csv(self.path.joinpath('dataset.csv'), delimiter=',')
        annotations = []
        for annotations in range(len(df)):
            # h, w, _ = cv2.imread(str(self.paths_to_images[anno])).shape
            with Image.open(str(self.imgs_path[annotations]), 'r') as img:
                w, h = img.size
            annotations.append([
                od.Annotation(BBox.build_from_center_and_size(
                    np.array([int(df['x_center'][annotations] * w), int(df['y_center'][annotations] * h)]),
                    np.array([int(df['width'][annotations] * w), int(df['height'][annotations] * h)])),
                    1.0, 'license_plate')
            ])
        del df
        return annotations
