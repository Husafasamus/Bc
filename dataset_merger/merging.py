from typing import Optional
from pathlib import Path, PurePosixPath
import csv
import pandas
import shutil
import json

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
        self.path_to_destination.joinpath('images').mkdir(exist_ok=True)  # dir
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

        # If not
        # <> 1. check for last index in ds which is in what is inside.txt
        # 2. copy imgs from dataset
        # 3. copy his annotations
        # 4. update what is inside.txt with new dataset props
        # 5. return True if it was succesfully

        # 1. Last known index in merget_datasets
        last_file = 1
        if self.last_index_in_merget_ds() != '-1':
            last_file = int(self.last_index_in_merget_ds()) + 1

        last_file_copy = last_file  # For annotations
        # 2. copy images from dataset
        for path_of_img in new_dataset.paths_to_images:
            shutil.copy(path_of_img, self.path_to_destination.joinpath('images')
                        .joinpath(f'{last_file}{PurePosixPath(path_of_img.name).suffix}'))
            last_file += 1

        # 3. Copy annotations
        #   - read annos get labels()
        #   - append to new merget labels
        last_file = last_file_copy
        data_annotations = self._create_dict_from_annotations(new_dataset)
        for num in range(len(data_annotations['content'])):
            data_annotations['content'][num]['file_name'] = f'{last_file}.jpg'
            last_file += 1

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

        return 0
