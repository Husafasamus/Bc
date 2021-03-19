from Creator import Dataset as ds
from Creator import ObjectDetector as od
from Creator import bbox
import pandas

"""
    reference to: https://ai.stanford.edu/~jkrause/cars/car_dataset.html
"""

Annotations = list[list[od.Annotation]]


class CarsDataset(ds.Dataset):

    def __init__(self, path: str) -> None:
        super().__init__(path)

    def find_images(self) -> None:
        for x in self.path.joinpath('car_ims').iterdir():
            self.paths_to_images.append(x)

    def get_labels(self) -> Annotations:
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
            od.Annotation(bbox.BBox(int(df['bbox_x1'][img]), df['bbox_y1'][img],df['bbox_x2'][img],df['bbox_y2'][img]), 1.0, 'car')]
            annotations.append(bboxes_on_picture)
        return annotations



class UFPRALPRDataset(ds.Dataset):

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.paths_to_annos = []

    def find_images(self) -> None:
        for dir_in_ds in self.path.iterdir():     # testing, training, validation
            for dir_in_sub_ds in dir_in_ds.iterdir():      #   ---> track0091, track0092, track0093, ...
                for img_w_ants in dir_in_sub_ds.iterdir(): # ---> track0091[01].png, track0091[01].txt, ...
                    if img_w_ants.name.__contains__('.png'):
                        self.paths_to_images.append(img_w_ants)
                    elif img_w_ants.name.__contains__('.txt'):
                        self.paths_to_annos.append(img_w_ants)

    def get_labels(self) -> Annotations:

        annotations = []
        for textFile in self.paths_to_annos:
            annotations.append(self.get_annotation_from_txt_file(textFile))

        return annotations

    def get_annotation_from_txt_file(self, path: str):
        with open(path, 'r') as f:
            txt = f.readlines()
            bboxes_on_picture = []
            bboxes_on_picture.append(od.Annotation(bbox.BBox(int(txt[1].split(' ')[1]), int(txt[1].split(' ')[2]), int(txt[1].split(' ')[3]), int(txt[1].split(' ')[4])), 1.0, txt[2].split(' ')[1].replace('\n', '')))
            bboxes_on_picture.append(od.Annotation(bbox.BBox(int(txt[7].split(' ')[1]), int(txt[7].split(' ')[2]), int(txt[7].split(' ')[3]), int(txt[7].split(' ')[4])), 1.0, 'license_plate'))

        return bboxes_on_picture

