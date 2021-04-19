from typing import Optional, Tuple
from pathlib import Path, PurePosixPath
from PIL import Image
import csv
import pandas
import shutil
import json
import random
import statistics
from dataset_merger.dataset import Dataset
from dataset_merger.bbox import BBox
#from dataset_merger.object_detector import Annotation
from dataset_merger import object_detector

"""

Files in MergedDatasets:
-Merged_datasets>
                --images
                --annotations
                --what_is_inside.csv
"""


class DatasetMerger:

    def __init__(self, destination_path: str) -> None:
        self.destination_path = Path(destination_path)
        self.wsi_path = self.destination_path.joinpath('what_is_inside.txt')

        self.destination_path.mkdir(exist_ok=True)  # Dest. path e.g. = D:\......\Merged_datasets
        # self.path_to_destination.joinpath('images').mkdir(exist_ok=True)  # dir
        self.destination_path.joinpath('annotations').mkdir(exist_ok=True)  # dir
        self.create_wsi(self.wsi_path)  # csv

    @staticmethod
    def compare_detections(dataset: Dataset):
        """
        Dataset: yolov3 -> detections.json
                 ssd    -> detections.json
        Vlastnosti, ktore treba zohladnit:
            1. Confidence
            2. Rozdiel toho ako sa lisia dva bboxy

        TO DO:
        1. Najst cesty k detections
        2. Nacitat jsony
        3. Loop v jsone a nacitanie obrazkov
        4. Vytvorit funkciu, na porovnanie annotacii v jednom obrazku
        5.

        """
        # Paths to detections.json, for each detector
        detections_yolov3_path = dataset.path.joinpath('Our_detections').joinpath('yolov3').joinpath('detections.json')
        detections_ssd_path = dataset.path.joinpath('Our_detections').joinpath('ssd').joinpath('detections.json')

        data_manual = {'content': []}
        data_completed = {'content': []}

        # Load jsons
        with open(detections_yolov3_path) as file:
            detection_yolov3 = json.load(file)

        with open(detections_ssd_path) as file:
            detection_ssd = json.load(file)

        # Loop in json and load separate imgs
        for index in range(len(detection_yolov3['content'])):

            #img annotations
            img_yolov3 = detection_yolov3['content'][index]
            img_ssd = detection_ssd['content'][index]

            # Compare 'license_plate' labels
            # Lists of lpn annotations
            list_yolov3_lpn = []
            list_ssd_lpn = []

            # List of vehicle annotations
            list_yolov3_vehicle = []
            list_ssd_vehicle = []

            # yolov3 check
            for index_annotation in range(len(img_yolov3['annotations'])):
                # Check if img contains 'license_plate' else add to vehicle list
                if img_yolov3['annotations'][index_annotation]['label'] == 'license_plate':
                    list_yolov3_lpn.append(img_yolov3['annotations'][index_annotation])
                else:
                    list_yolov3_vehicle.append(img_yolov3['annotations'][index_annotation])

            # ssd check
            for index_annotation in range(len(img_ssd['annotations'])):
                # Check if img contains 'license_plate' else add to vehicle list
                if img_ssd['annotations'][index_annotation]['label'] == 'license_plate':
                    list_ssd_lpn.append(img_ssd['annotations'][index_annotation])
                else:
                    list_ssd_vehicle.append(img_ssd['annotations'][index_annotation])

            # Funkcia, ktora mi prehodnoti detekovane anotacie na obrazku
            """
                Kedy, by sa mala anotacia uchovat?
                Kedy, by sa mala anotacia odstranit?
                Kedy, by  sa mal obrazok prehodnotit a vlozit do separatneho priecinku?
            """

            #print('yolo',list_yolov3_vehicle)
            #print('ssd',list_ssd_vehicle)

            # Lists of new annotations
            new_vehicle_annotations = []
            new_lpn_annotations = []

            new_vehicle_manual_annotations = []

            # Cyklus ktory porovna kazde jedno auto s druhym
            for vehicle_yolo in list_yolov3_vehicle:
                bbox_yolo = BBox(vehicle_yolo['bbox']['x'], vehicle_yolo['bbox']['y'], vehicle_yolo['bbox']['width'], vehicle_yolo['bbox']['height'])

                for vehicle_ssd in list_ssd_vehicle:
                    bbox_ssd = BBox(vehicle_ssd['bbox']['x'], vehicle_ssd['bbox']['y'], vehicle_ssd['bbox']['width'], vehicle_ssd['bbox']['height'])

                    # Check if intersection in bboxes does exist
                    bbox_intersection = BBox.intersection(bbox_yolo, bbox_ssd)
                    if bbox_intersection is not None:
                        print('Intersection in: ', vehicle_yolo, vehicle_ssd)
                        # Check for difference in confidences
                        capacity_yolo = bbox_yolo.capacity()
                        capacity_ssd = bbox_ssd.capacity()
                        capacity_intersection = bbox_intersection.capacity()

                        if (capacity_intersection / capacity_yolo) > 0.6 or (capacity_intersection / capacity_ssd) > 0.6: # Check if the annotations does annotate same object

                            if vehicle_yolo['confidence'] > 0.8 and vehicle_ssd['confidence'] > 0.8: # Check if confidences are above 0.8

                                if abs(vehicle_yolo['confidence'] - vehicle_ssd['confidence']) <= 0.05: # Check difference in confidences
                                    # If yes, rescale it to bigger annotation
                                    bbox_intersection.rescale(1.2, 1.2)
                                    new_vehicle_annotations.append(object_detector.Annotation(bbox_intersection, statistics.mean([vehicle_yolo['confidence'], vehicle_ssd['confidence']]), vehicle_yolo['label']))
                                    break
                                    # --> Final annotation
                                else:
                                    # If not, check for confidences, find max confidence from two of them and rescale to 1.1
                                    if vehicle_yolo['confidence'] > vehicle_ssd['confidence']:

                                        bbox_yolo.rescale(1.1, 1.1)
                                        new_vehicle_annotations.append(object_detector.Annotation(bbox_yolo,vehicle_yolo['confidence'], vehicle_yolo['label']))
                                        break
                                        # --> Final annotation
                                    else:
                                        bbox_ssd.rescale(1.1, 1.1)
                                        new_vehicle_annotations.append(object_detector.Annotation(bbox_ssd, vehicle_ssd['confidence'], vehicle_ssd['label']))
                                        break
                                        # --> Final annotation
                            else:
                                # If confidences is less, then 0.8
                                # maybe nothing
                                # add to manual detection !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                new_vehicle_manual_annotations.append(object_detector.Annotation(bbox_ssd, vehicle_ssd['confidence'], vehicle_ssd['label']))
                                new_vehicle_manual_annotations.append(object_detector.Annotation(bbox_yolo,vehicle_yolo['confidence'], vehicle_yolo['label']))
                                pass
                        else:
                            # Annotations probably does not annotate same object currently
                            continue
                    else:
                        # Intersection does not exists
                        continue
                        pass

                    pass

                pass




            print('manual: ', new_vehicle_manual_annotations)
            print('good: ', new_vehicle_annotations)

            if len(new_vehicle_manual_annotations) > 0:
                # Tak zapis do json detections ktore treba opravit a nasledne si ich pouzivatel otvori v label toole
                data_manual['content'].append(detection_yolov3['content'][index])
            else:
                pass



        # LPN check
        pass




    @staticmethod
    def create_wsi(path_to: str) -> bool:
        if Path(path_to).exists():
            return False
        else:
            Path(path_to).touch(exist_ok=True)  # text file
            with open(path_to, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["dataset_name", "dataset_path", "number_of_pictures", "index_from", "index_to"])
            return True

    @staticmethod
    def file_write(path: str, data, indent=0) -> None:
        with open(path, 'w') as f:
            # f.write(json.dumps(data))
            json.dump(data, f, indent=indent)
            f.close()

    @staticmethod
    def create_dict_from_annotations(dataset: Dataset) -> dict:
        data = {'content': []}
        imgs = dataset.imgs_path
        annotations = dataset.get_labels()

        for i_img in range(len(imgs)):
            data['content'].append({
                'file_name': f'{Path(imgs[i_img]).name}',
                'annotations': []
            })
            for annotation in annotations[i_img]:
                data['content'][i_img]['annotations'].append(annotation.build_dictionary())

        return data

    @staticmethod
    def create_dict_from_annotations_detected(dataset: Dataset, annos) -> dict:
        data = {'content': []}
        imgs = dataset.imgs_path
        annotations = annos

        for i_img in range(len(imgs)):
            data['content'].append({
                'file_name': f'{Path(imgs[i_img]).name}',
                'annotations': []
            })
            for annotation in annotations[i_img]:
                data['content'][i_img]['annotations'].append(annotation.build_dictionary())

        return data


    def is_ds_inside(self, dataset: Dataset) -> bool:
        wsi = pandas.read_csv(self.wsi_path)
        for name in wsi["dataset_name"]:
            if dataset.name == name:
                del wsi
                return True
        for path in wsi["dataset_path"]:
            if dataset.path == path:
                del wsi
                return True

        return False

    def wsi_last_index(self) -> str:
        with open(str(self.wsi_path), 'r') as file:
            data = file.readlines()
            if data[-1].split(',')[-1].rstrip() == 'index_to':
                return '-1'
            else:
                return data[-1].split(',')[-1].rstrip()

    def import_dataset(self, new_dataset: Dataset) -> int:
        if self.is_ds_inside(new_dataset):  # If dataset is inside the 'what_is_inside.txt', function returns False
            print(f"Dataset: {new_dataset} is already in!")
            return 3

        self.destination_path.joinpath('images').mkdir(exist_ok=True)  # dir

        # If not
        # <> 1. check for last index in ds which is in what is inside.txt
        # 2. copy imgs from dataset
        # 3. copy his annotations
        # 4. update what is inside.txt with new dataset props
        # 5. return 0 if it was succesfully

        # 1. Last known index in Merged_datasets
        last_file = 1
        if self.wsi_last_index() != '-1':
            last_file = int(self.wsi_last_index()) + 1

        last_file_copy = last_file  # For annotations
        # 2. copy images from dataset
        for img_path in new_dataset.imgs_path:
            with Image.open(img_path) as img:
                img.convert('RGB'). \
                    save(f"{self.destination_path.joinpath('images').joinpath(f'{last_file:05d}.jpg')}")
            last_file += 1

        # 3. Copy annotations
        #   - read annotations get labels()
        #   - append to new merged labels

        # last file index
        last_file = last_file_copy
        data_annotations = self.create_dict_from_annotations(new_dataset)
        for num in range(len(data_annotations['content'])):
            data_annotations['content'][num]['file_name'] = f'{last_file:05d}.jpg'
            last_file += 1

        # detections json path
        destination_json_path = self.destination_path.joinpath('annotations').joinpath('detections.json')
        if destination_json_path.exists():
            with open(destination_json_path) as file:
                detections = json.load(file)
                for img_num in range(len(data_annotations)):
                    detections['content'].append(data_annotations['content'][img_num])
            self.file_write(destination_json_path, detections, 4)
        else:
            self.file_write(destination_json_path, data_annotations, 4)

        # 4. Update what is inside txt with new dataset props
        with open(self.wsi_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f"{new_dataset.name}", f"{new_dataset.path}",
                             f"{len(new_dataset.imgs_path)}",
                             f"{last_file_copy}", last_file - 1])

        # 5. Return
        return 0

    def get_status(self) -> Tuple[bool, bool]:
        """
        return
        -> (False, True): without images
        -> (True, False): file images without train, test, validation
              is inside merged ds. -> Merge train,test, validation
              into images again and split it again.
        -> (True, True): images with train, test, validation
        """
        imgs = False
        train = False
        for inside in self.destination_path.iterdir():
            if inside.name == 'images':
                imgs = True
        for inside in self.destination_path.iterdir():
            if inside.name == 'train':
                train = True
        return imgs, train

    def split_train_test_validation(self, train_size=.6, test_size=.2, validation_size=.2) -> None:
        # Check if train, test, validation sizing is correct
        if not (train_size + test_size + validation_size) == 1.:
            print('You have choose bad ratio of sizes train, test and validation!')
            return None

        # Check if file images is in, which has to be splited
        is_imgs = self.get_status()

        # If ds include images dir
        if is_imgs == (True, False):
            print('images in!')
            self._split_train_test_validation(train_size, test_size)
        elif is_imgs == (True, True):
            # Create imgs dir
            # Recopy all images from train, test and validation images into images dir and then split it again
            self.copy_imgs_dir_and_remove('train')
            self.copy_imgs_dir_and_remove('test')
            self.copy_imgs_dir_and_remove('validation')
            self._split_train_test_validation(train_size, test_size)

        return None

    def copy_imgs_dir_and_remove(self, dir_name: str) -> None:
        imgs_path = self.destination_path.joinpath('images')
        for img_path in self.destination_path.joinpath(dir_name).iterdir():
            shutil.copy(img_path, imgs_path.joinpath(img_path.name))
        shutil.rmtree(self.destination_path.joinpath(dir_name))

        return None

    def _split_train_test_validation(self, train_size: float, test_size: float) -> None:
        imgs_list = list(self.destination_path.joinpath('images').iterdir())
        c_imgs = len(imgs_list)
        c_train_imgs = int(c_imgs * train_size)
        c_test_imgs = int(c_imgs * test_size)
        # c_validation_imgs = int(c_imgs * validation_size)

        # Shuffle imgs_list
        random.shuffle(imgs_list)
        # Copy into train dir
        train_path = self.destination_path.joinpath('train')
        train_path.mkdir()

        for i in range(c_train_imgs):
            shutil.copy(imgs_list[i], train_path.joinpath(imgs_list[i].name))

        # Copy into test dir
        test_path = self.destination_path.joinpath('test')
        test_path.mkdir()

        for i in range(c_train_imgs, c_train_imgs + c_test_imgs):
            shutil.copy(imgs_list[i], test_path.joinpath(imgs_list[i].name))

        # Copy into validation dir
        validation_path = self.destination_path.joinpath('validation')
        validation_path.mkdir()

        for i in range(c_train_imgs + c_test_imgs, c_imgs):
            shutil.copy(imgs_list[i], validation_path.joinpath(imgs_list[i].name))

        # Remove images dir
        shutil.rmtree(self.destination_path.joinpath('images'))

        return None
