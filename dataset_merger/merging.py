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
import itertools

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
    def compare_detections_n(dataset: Dataset, confidence_treshold=0.8, difference_in_confidence=0.05, bbox_perc_intersection=0.6) -> Tuple[int, int]:
        """
        Work: Load annotations from detections
        Note: Every object detector has its own directory with detections.json
        detections_json_from_detectors path for each detector dir.
        """
        detections_json_from_detectors = []
        for detections in dataset.path.joinpath('Our_detections').iterdir():
            if detections.name != 'Our_detections' or detections.name != 'detections.json':
                detections_json_from_detectors.append(detections.joinpath('detections.json'))

        # Check if annotations exists
        if len(detections_json_from_detectors) == 0:
            return None

        """
        Work: Load jsons to lists 
        Note: Variable list will contain annotations from each 
              detector.
        detections_detectors: each index contains detections from one detector
        """
        detections_detectors = []

        for path_json in detections_json_from_detectors:
            with open(path_json) as detections:
                detections_detectors.append(json.load(detections))

        """
        Work: Create dictionary, where each img will contain annotations from every detector
        Note: Variable dict, will contain every annotations
        detections_all:
        """
        detections_all = {'content': []}

        # Append every img_file to detections_all
        for index_img in range(len(dataset.imgs_path)):
            detections_all['content'].append({
                'file_name': f"{dataset.imgs_path[index_img].name}",
                'annotations': []
            })
            for index_detector in range(len(detections_detectors)):
                for index_detection in range(len(detections_detectors[index_detector]['content'])):
                    if dataset.imgs_path[index_img].name == detections_detectors[index_detector]['content'][index_detection]['file_name']:
                        for index_annotation in range(len(detections_detectors[index_detector]['content'][index_detection]['annotations'])):
                            detections_all['content'][index_img]['annotations'].append(detections_detectors[index_detector]['content'][index_detection]['annotations'][index_annotation])

        # Delete garbage
        del detections_json_from_detectors
        del detections_detectors

        """
        Work: Compare annotations in one separate img
        Note: 
        """
        # Images for manual annotation
        imgs_manual = []
        # Images done
        imgs_done = {'content': []}

        index_done = 0


        for index_img in range(len(detections_all['content'])):
            result = DatasetMerger.compare_annotations_in_img(detections_all['content'][index_img], confidence_treshold, difference_in_confidence, bbox_perc_intersection) # Result None as manual annotations or Annotations for successfull annotation
            if result is None:
                imgs_manual.append(detections_all['content'][index_img])
                continue

            file_name = detections_all['content'][index_img]['file_name']
            imgs_done['content'].append({
                'file_name': f'{file_name}',
                'annotations': []
            })
            for anno in result:
                imgs_done['content'][index_done]['annotations'].append(anno.build_dictionary())
            index_done += 1

        del detections_all

       # print('manual', imgs_manual)
        #print('done', imgs_done)
        manual_data = {'content': []}



        c_done = len(imgs_done['content'])

        for img_detection in imgs_manual:
            manual_data['content'].append(img_detection)
            pass

        c_manual = len(manual_data['content'])

        DatasetMerger.file_write('detections_manual.json', manual_data, indent=4)
        DatasetMerger.file_write('detections_completed.json', imgs_done, indent=4)

        # Return Tuple[count_manual, count_done]
        return (c_manual, c_done)

    @staticmethod
    def compare_annotations_in_img(img_detection: dict, confidence_treshold=0.8, difference_in_confidence=0.05, bbox_perc_intersection=0.6) -> 'Annotations':
        """
        Work: Separate vehicles and lpns
        Note:
        vehicles
        lpns
        """
        vehicles = []
        lpns = []

        for index_annotation in range(len(img_detection['annotations'])):
            # Check if img contains 'license_plate' else add to vehicle list
            if img_detection['annotations'][index_annotation]['label'] == 'license_plate':
                lpns.append(img_detection['annotations'][index_annotation])
            else:
                vehicles.append(img_detection['annotations'][index_annotation])

        # If img does not contain vehicles, img is automaticaly added to manual annotation list
        #if len(vehicles) == 0:
        #    return None # None for manual annotations or I return new annotations

        """
        Work: Compare vehicles
        Note:
        result successful Annotations
        """
        # index, which were added
        added = []
        successful_annotations_vehicles = []

        if len(vehicles) == 1:
            if vehicles[0]['confidence'] >  confidence_treshold:
                bbox_ = BBox(vehicles[0]['bbox']['x'], vehicles[0]['bbox']['y'], vehicles[0]['bbox']['width'],
                              vehicles[0]['bbox']['height'])
                successful_annotations_vehicles.append(object_detector.Annotation(bbox_, vehicles[0]['label'], vehicles[0]['confidence']))
            else:
                return None
        else:
            for x, y in itertools.combinations(range(len(vehicles)), 2):
                if not x in added:
                    if not y in added:
                        index, result = DatasetMerger.compare_two_annotations(vehicles[x], vehicles[y], confidence_treshold, difference_in_confidence, bbox_perc_intersection) # If none is for manual annotations

                        if index == -1:
                            continue
                        if result is None:
                            return None

                        successful_annotations_vehicles.append(result)
                        if index == 3:
                            added.append(x)
                            added.append(y)
                        elif index == 1:
                            added.append(x)
                        else:
                            added.append(y)

            del added
        #return successful_annotations

        if len(successful_annotations_vehicles) == 0:
            return None

        """
        Work: Compare LPNs
        Note: 
        """
        added = []
        successful_annotations_lpns = []

        if len(lpns) == 1:
            if lpns[0]['confidence'] >  confidence_treshold:
                bbox_ = BBox(lpns[0]['bbox']['x'], lpns[0]['bbox']['y'], lpns[0]['bbox']['width'],
                              lpns[0]['bbox']['height'])
                successful_annotations_lpns.append(object_detector.Annotation(bbox_, lpns[0]['label'], lpns[0]['confidence']))
        else:
            for x, y in itertools.combinations(range(len(lpns)), 2):
                if not x in added:
                    if not y in added:
                        index, result = DatasetMerger.compare_two_annotations(lpns[x], lpns[y], confidence_treshold, difference_in_confidence, bbox_perc_intersection) # If none is for manual annotations

                        if index == -1:
                            continue
                        if result is None:
                            continue
                            #return None # -----------------------?-----------------------------

                        successful_annotations_lpns.append(result)
                        if index == 3:
                            added.append(x)
                            added.append(y)
                        elif index == 1:
                            added.append(x)
                        else:
                            added.append(y)

            del added

        """
        Work: Detect LPNs on vehicles.
        Note: LPN should be in the detected vehicle.
              LPN can't be without vehicle 
        append to successful_annotations_vehicles for return
        """
        #

        for lpn_detection in successful_annotations_lpns:
            bbox_lpn = lpn_detection.bbox

            for vehicle_detection in successful_annotations_vehicles:
                bbox_vehicle = vehicle_detection.bbox
                bbox_i = BBox.intersection(bbox_lpn, bbox_vehicle)
                if bbox_i is not None:
                    if bbox_i.capacity() / bbox_lpn.capacity() > 0.9:   # Define number, which define us when the bounding boxe is in another
                        successful_annotations_vehicles.append(lpn_detection)
                        break

        return successful_annotations_vehicles

    @staticmethod
    def compare_two_annotations(annotation1: dict, annotation2: dict, confidence_treshold=0.8, difference_in_confidence=0.05, bbox_perc_intersection=0.6):
        bbox_1 = BBox(annotation1['bbox']['x'], annotation1['bbox']['y'], annotation1['bbox']['width'], annotation1['bbox']['height'])
        bbox_2 = BBox(annotation2['bbox']['x'], annotation2['bbox']['y'], annotation2['bbox']['width'], annotation2['bbox']['height'])
        bbox_i = BBox.intersection(bbox_1, bbox_2)

        if bbox_i is not None:
            bbox_1_c, bbox_2_c, bbox_i_c  = bbox_1.capacity(), bbox_2.capacity(), bbox_i.capacity()

            if bbox_i_c / bbox_1_c > bbox_perc_intersection or bbox_i_c / bbox_2_c > bbox_perc_intersection: # Maybe I will change this one
                del bbox_1_c, bbox_2_c, bbox_i_c

                if annotation1['confidence'] > confidence_treshold and annotation2['confidence'] > confidence_treshold:
                    if abs(annotation1['confidence'] - annotation2['confidence']) <= difference_in_confidence:
                        bbox_i.rescale(1.1, 1.1)
                        return 3, object_detector.Annotation(bbox_i, min(annotation1['confidence'], annotation2['confidence']), annotation1['label'])
                else:
                    if annotation1['confidence'] > annotation2['confidence']:
                        bbox_1.rescale(1.1, 1.1)
                        return 1, object_detector.Annotation(bbox_1, annotation1['confidence'], annotation1['label'])
                    else:
                        bbox_2.rescale(1.1, 1.1)
                        return 2, object_detector.Annotation(bbox_2, annotation2['confidence'], annotation2['label'])

        return -1, None

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
