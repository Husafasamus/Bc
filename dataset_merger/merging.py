from typing import Optional, Tuple
from pathlib import Path, PurePosixPath
import csv
import pandas
import shutil
import json
import random

from dataset_merger.dataset import Dataset

"""

Files in MergetDatasets:
-Merget datasets>
                --images
                --annotations
                --what_is_inside.csv
"""


class DatasetMerger:

    def __init__(self, path_to_destination: str) -> None:
        self.path_to_destination = Path(path_to_destination)
        self.path_to_what_is_inside = self.path_to_destination.joinpath('what_is_inside.txt')

        self.path_to_destination.mkdir(exist_ok=True)  # Dest. path e.g. = D:\......\Merget_datasets
        #self.path_to_destination.joinpath('images').mkdir(exist_ok=True)  # dir
        self.path_to_destination.joinpath('annotations').mkdir(exist_ok=True)  # dir
        self._create_what_is_inside(self.path_to_what_is_inside)  # csv

    @staticmethod
    def _create_what_is_inside(path_to: str) -> bool:
        if Path(path_to).exists():
            return False
        else:
            Path(path_to).touch(exist_ok=True)  # text file
            with open(path_to, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["dataset_name", "dataset_path", "number_of_pictures", "index_from", "index_to"])
            return True

    @staticmethod
    def _write_to_file(path: str, data, indent=0) -> None:
        with open(path, 'w') as f:
            # f.write(json.dumps(data))
            json.dump(data, f, indent=indent)
            f.close()

    @staticmethod
    def _create_dict_from_annotations(dataset: Dataset) -> dict:
        data = {'content': []}
        imgs = dataset.paths_to_images
        annotations = dataset.get_labels()

        for i_img in range(len(imgs)):
            data['content'].append({
                'file_name': f'{Path(imgs[i_img]).name}',
                'annotations': []
            })
            for annotation in annotations[i_img]:
                data['content'][i_img]['annotations'].append(annotation.build_dictionary())

        return data

    def is_ds_inside(self, dataset: Dataset) -> bool:
        wsi = pandas.read_csv(self.path_to_what_is_inside)
        for name in wsi["dataset_name"]:
            if dataset.name == name:
                del wsi
                return True
        for path in wsi["dataset_path"]:
            if dataset.path == path:
                del wsi
                return True

        return False

    def last_index_in_merget_ds(self) -> str:
        with open(str(self.path_to_what_is_inside), 'r') as file:
            data = file.readlines()
            if data[-1].split(',')[-1].rstrip() == 'index_to':
                return '-1'
            else:
                return data[-1].split(',')[-1].rstrip()

    def import_dataset(self, new_dataset: Dataset) -> int:
        if self.is_ds_inside(new_dataset):  # If dataset is inside the 'what_is_inside.txt', function returns False
            print(f"Dataset: {new_dataset} is already in!")
            return 3

        self.path_to_destination.joinpath('images').mkdir(exist_ok=True)  # dir


        # If not
        # <> 1. check for last index in ds which is in what is inside.txt
        # 2. copy imgs from dataset
        # 3. copy his annotations
        # 4. update what is inside.txt with new dataset props
        # 5. return 0 if it was succesfully

        # 1. Last known index in merget_datasets
        last_file = 1
        if self.last_index_in_merget_ds() != '-1':
            last_file = int(self.last_index_in_merget_ds()) + 1

        last_file_copy = last_file  # For annotations
        # 2. copy images from dataset
        for img_path in new_dataset.paths_to_images:
            shutil.copy(img_path, self.path_to_destination.joinpath('images')
                        .joinpath(f'{last_file}{PurePosixPath(img_path.name).suffix}'))
            last_file += 1

        # Cez pillow ak je png tak jpg a zmenit format 00001.jpg 5 miest



        # 3. Copy annotations
        #   - read annos get labels()
        #   - append to new merget labels

        #last file index
        last_file = last_file_copy
        data_annotations = self._create_dict_from_annotations(new_dataset)
        for num in range(len(data_annotations['content'])):
            data_annotations['content'][num]['file_name'] = f'{last_file}.jpg'
            last_file += 1

        #detections json path
        path_to_detections_json = self.path_to_destination.joinpath('annotations').joinpath('detections.json')
        if path_to_detections_json.exists():
            with open(path_to_detections_json) as file:
                detections = json.load(file)
                for img_num in range(len(data_annotations)):
                    detections['content'].append(data_annotations['content'][img_num])
            self._write_to_file(path_to_detections_json, detections, 4)
        else:
            self._write_to_file(path_to_detections_json, data_annotations, 4)

        # 4. Update what is inside txt with new dataset props
        with open(self.path_to_what_is_inside, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f"{new_dataset.name}", f"{new_dataset.path}",
                             f"{len(new_dataset.paths_to_images)}",
                             f"{ last_file_copy}", last_file-1])

        # 5. Return
        return 0

    def get_status(self) -> Tuple[bool, bool]:
        """
        return
        -> (False, True): without images
        -> (True, False): file images without train, test, validation
              is inside merget ds. -> Merge train,test, validation
              into images again and split it again.
        -> (True, True): images with train, test, validation
        """
        tmp_imgs = False
        tmp_train = False
        for inside in self.path_to_destination.iterdir():
            if inside.name == 'images':
                tmp_imgs = True
        for inside in self.path_to_destination.iterdir():
            if inside.name == 'train':
                tmp_train = True
        return tmp_imgs, tmp_train

    def split_train_test_validation(self, train_size = .6, test_size = .2, validation_size = .2) -> None:
        # Check if train, test, validation sizing is correct
        if not (train_size + test_size + validation_size) == 1.:
            print('You have choose bad ratio of sizes train, test and validation!')
            return None

        # Check if file images is in, which has to be splited
        is_images = self.get_status()

        # If ds include images dir
        if is_images == (True, False):
            print('images in!')
            self._split_train_test_validation(train_size, test_size, validation_size)
        elif is_images == (True, True):
            # Creaste images dir
            # Recopy all images from train, test and validation images into images dir and then split it again

            # From train
            self.copy_dir_to_imgs_and_remove('train')
            self.copy_dir_to_imgs_and_remove('test')
            self.copy_dir_to_imgs_and_remove('validation')
            self._split_train_test_validation(train_size, test_size, validation_size)
        else:
            return None

        return None

    def copy_dir_to_imgs_and_remove(self, dir_name: str) -> None:
        imgs_path = self.path_to_destination.joinpath('images')
        for img_path in self.path_to_destination.joinpath(dir_name).iterdir():
            shutil.copy(img_path, imgs_path.joinpath(img_path.name))
        shutil.rmtree(self.path_to_destination.joinpath(dir_name))
        return None

    def _split_train_test_validation(self, train_size: float, test_size: float, validation_size: float) -> None:
        imgs_list = list(self.path_to_destination.joinpath('images').iterdir())
        c_imgs = len(imgs_list)
        c_train_imgs = int(c_imgs * train_size)
        c_test_imgs = int(c_imgs * test_size)
        c_validation_imgs = int(c_imgs * validation_size)

        # Shuffle imgs_list
        random.shuffle(imgs_list)
        print(imgs_list)
        # Copy into train dir
        train_path = self.path_to_destination.joinpath('train')
        train_path.mkdir()

        for i in range(c_train_imgs):
            shutil.copy(imgs_list[i], train_path.joinpath(imgs_list[i].name))

        # Copy into test dir
        test_path = self.path_to_destination.joinpath('test')
        test_path.mkdir()

        for i in range(c_train_imgs, c_train_imgs + c_test_imgs):
            shutil.copy(imgs_list[i], test_path.joinpath(imgs_list[i].name))

        # Copy into validation dir
        validation_path = self.path_to_destination.joinpath('validation')
        validation_path.mkdir()

        for i in range(c_train_imgs + c_test_imgs, c_imgs):
            shutil.copy(imgs_list[i], validation_path.joinpath(imgs_list[i].name))

        # Remove images dir
        shutil.rmtree(self.path_to_destination.joinpath('images'))

        return None