from Creator import Dataset as ds
from Creator import ObjectDetector as od
import pathlib
import csv
import pandas

"""
    https://ai.stanford.edu/~jkrause/cars/car_dataset.html
"""

class CarsDataset(ds.Dataset):

    def __init__(self, path: str):
        super().__init__(path)

    def find_images(self):
        #self.paths_to_images = self.path.joinpath('car_ims').iterdir()
        for x in self.path.joinpath('car_ims').iterdir():
            self.paths_to_images.append(x)

    def get_labels(self):
        """
        Reading .csv file
        delimiter = ';'
        example:
        relative_img_path;bbox_x1;bbox_y1;bbox_x2;bbox_y2;class;test
        car_ims/000001.jpg;112;7;853;717;1;0
        """
        df = pandas.read_csv(self.path.joinpath('cars_annos.txt'), delimiter=';')

        detections = []
        for img in range(len(df)):
            bboxes_on_picture = []
            bboxes_on_picture.append(od.Detection(od.BBox(df['bbox_x1'][img], df['bbox_y1'][img],df['bbox_x2'][img],df['bbox_y2'][img]), 1.0, 'car'))
            detections.append(bboxes_on_picture)
        return detections
