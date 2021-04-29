from dataset_merger.dataset import Dataset
from dataset_merger import object_detector as od
from dataset_merger.bbox import BBox

import pandas
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
from benedict import benedict as bdict
import dataset_merger.merging
import json

class CarsDataset(Dataset):
    """reference to: https://ai.stanford.edu/~jkrause/cars/car_dataset.html"""

    def __init__(self, path: str) -> None:
        super().__init__(path)

    def find_images(self) -> None:
        for x in self.path.joinpath('car_ims').iterdir():
            self.imgs_path.append(x)
        #for x in self.path.ite

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

        #complete

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
        for index_annotations in range(len(df)):
            # h, w, _ = cv2.imread(str(self.paths_to_images[anno])).shape
            with Image.open(str(self.imgs_path[index_annotations]), 'r') as img:
                w, h = img.size
            annotations.append([
                od.Annotation(BBox.build_from_center_and_size(
                    np.array([int(df['x_center'][index_annotations] * w), int(df['y_center'][index_annotations] * h)]),
                    np.array([int(df['width'][index_annotations] * w), int(df['height'][index_annotations] * h)])),
                    1.0, 'license_plate')
            ])
        del df

        annos = dataset_merger.merging.DatasetMerger.create_dict_from_annotations_detected(self, annotations)
        with open(self.path.joinpath('detections.json')) as det:
            detections_vehicles = json.load(det)

        for img_ann in annos['content']:
            ann = self.find_anos(detections_vehicles, img_ann['file_name'])
            for a in ann:
                img_ann['annotations'].append(
                    a
                )

        return annos

    @staticmethod
    def find_anos(annos, file_name) -> 'Annotations':
        ret = []
        for file_annotations in annos['content']:
            if file_annotations['file_name'] == file_name:
                for ann in file_annotations['annotations']:
                    ret.append(ann)
        return ret

class CarLicensePlates(Dataset):
    def __init__(self, path: str) -> None:
        super().__init__(path)

    def find_images(self) -> None:
        for img in self.path.joinpath('images').iterdir():
            self.imgs_path.append(img)

    def get_labels(self) -> 'Annotations':
        annotations_path = self.path.joinpath('annotations')
        annotations = []
        for img_xml in annotations_path.iterdir():
            picture_bboxes = []
            data = bdict.from_xml(img_xml.__str__())
            if len(data['annotation']['object']) != 6:
                for object in data['annotation']['object']:
                    picture_bboxes.append(
                        od.Annotation(BBox(int(object['bndbox']['xmin']),
                                                              int(object['bndbox']['ymin']),
                                                              int(object['bndbox']['xmax']) - int(
                                                                  object['bndbox']['xmin']),
                                                              int(object['bndbox']['ymax']) - int(
                                                                  object['bndbox']['ymin'])), 1,
                                                         'license_plate'))
                    #print(object)
            else:
                #print(data['annotation']['object']['bndbox']['xmin'])
                try:
                    picture_bboxes.append(od.Annotation(BBox(int(data['annotation']['object']['bndbox']['xmin']),
                                                      int(data['annotation']['object']['bndbox']['ymin']), int(data['annotation']['object']['bndbox']['xmax']) - int(data['annotation']['object']['bndbox']['xmin']),
                                                      int(data['annotation']['object']['bndbox']['ymax']) - int(data['annotation']['object']['bndbox']['ymin'])), 1, 'license_plate'))
                except:
                    for object in data['annotation']['object']:
                        picture_bboxes.append(
                            od.Annotation(BBox(int(object['bndbox']['xmin']),
                                                                  int(object['bndbox']['ymin']),
                                                                  int(object['bndbox']['xmax']) - int(
                                                                      object['bndbox']['xmin']),
                                                                  int(object['bndbox']['ymax']) - int(
                                                                      object['bndbox']['ymin'])), 1,
                                                             'license_plate'))

            annotations.append(picture_bboxes)

        annos = dataset_merger.merging.DatasetMerger.create_dict_from_annotations_detected(self, annotations)
        with open(self.path.joinpath('Our_detections').joinpath('detections.json')) as file:
            detections = json.load(file)
        for index in range(len(detections['content'])):
            #add annootations

            for x in CarLicensePlates.find_anos(annos, detections['content'][index]['file_name']):
                detections['content'][index]['annotations'].append(x)

        dataset_merger.merging.DatasetMerger.file_write('detections.json', detections, indent=4)

            #Add our annotations


        return detections

    @staticmethod
    def find_anos(annos, file_name) -> 'Annotations':
        ret = []
        for file_annotations in annos['content']:
            if file_annotations['file_name'] == file_name:
                for ann in file_annotations['annotations']:
                    ret.append(ann)
        return ret


class SPZdataset(Dataset):

    def __init__(self, path: str) -> None:
        super().__init__(path)

    def find_images(self) -> None:
        for img in self.path.joinpath('images').iterdir():
            self.imgs_path.append(img)

    def build_dict(self) -> dict:
        # with Image.open(str(self.imgs_path[index_annotations]), 'r') as img:
        #    w, h = img.size
        annotations_path = self.path.joinpath('annotations')
        annotations = []
        data = {'content': []}

        # n = self.path.joinpath('annotations').iterdir()
        # print(n[0])
        for img in self.imgs_path:
            data['content'].append({
                'file_name': f'{img.name}',
                'annotations': []
            })
            picture_bboxes = []
            img_name = img.name.removesuffix('.jpg')
            txt_path = annotations_path.joinpath(f"{img_name}.txt")

            with Image.open(img.__str__(), 'r') as img:
                w, h = img.size

            with open(txt_path, "r") as file:
                picture_bboxes = []
                lines = file.readlines()
                for line in lines:
                    atrs = line.split()
                    width = int((float(atrs[3]) * w))
                    height = int((float(atrs[4]) * h))
                    t_x = int((float(atrs[0]) * w) - (width / 2))
                    t_y = int((float(atrs[1]) * h) - (height / 2))
                    if t_x < 0:
                        t_x = 0
                    if t_y < 0:
                        t_y = 0

                    data['content'][len(data['content']) - 1]['annotations'].append(
                        od.Annotation(
                            BBox(t_x, t_y, width, height), 1.0, "license_plate"
                        ).build_dictionary())
        return data

    def get_labels(self) -> 'Annotations':
        with open(self.path.joinpath('tomerge.json')) as file:
            detections_lpn = json.load(file)
        with open(self.path.joinpath('detections.json')) as det:
            detections_vehicles = json.load(det)

        for det in detections_lpn['content']:
            annos = self.find_anos(detections_vehicles, det['file_name'])
            for an in annos:
                det['annotations'].append(an)

        return detections_lpn # all

    @staticmethod
    def find_anos(annos, file_name) -> 'Annotations':
        ret = []
        for file_annotations in annos['content']:
            if file_annotations['file_name'] == file_name:
                for ann in file_annotations['annotations']:
                    ret.append(ann)
        return ret

   # def _annos(self, file_name):
     #   for imgs in self.path.joinpath('Out_detections').joinpath('detections.json'):

       # pass