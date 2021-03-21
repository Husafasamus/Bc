import pathlib
import shutil
import json

from typing import Iterable, Optional


class Dataset:

    # def __init__(self, name: str, path: pathlib.Path):
    #    self.name, self.path = name, path;

    def __init__(self, path: str) -> None:
        self.path = pathlib.Path(path)
        self.name = self.path.name
        self.paths_to_images = []

    def find_images(self) -> None:
        print(f"{self.name}'s function findImages() has not been defined!")


class FakeDataset1(Dataset):

    # def __init__(self, name: str, path: pathlib.Path):
    #   super().__init__(name, path)
    #  self.paths_to_images = []

    def __init__(self, path: str) -> None:
        super().__init__(path)
        # self.paths_to_images = []

    """
    This method gives me paths to all images, which dataset contains.
    """

    def find_images(self) -> None:
        for path in pathlib.Path(self.path).iterdir():
            if 'labels' not in path.name:
                self.paths_to_images.append(path)

    """
    This method gives me labels, which are already created.
    And transfer them into our convention.
    """

    def get_labels(self) -> str:
        labels_path = self.path.joinpath('labels.json')
        f = open(labels_path)
        data = json.load(f)
        f.close()
        return data

















class Unit:

    def __init__(self, path: str) -> None:
        """
        self.index ->
        """
        self.path = pathlib.Path(path)
        self.index = 1

    def copy_imgs_from_ds(self, dataset: Dataset) -> None:

        if not self.path.joinpath('imgs').exists():
            self.path.joinpath('imgs').mkdir(exist_ok=True)

        # Zmenit
        for path_img in dataset.paths_to_images:
            shutil.copy(path_img, self.path.joinpath('imgs').joinpath(f'img_{self.index}.jpg'))
            self.index += 1

    def add_labels_from_ds(self, labels: Iterable) -> None:
        """
        Create new dir for merget dataset and json file, where new or labels, that have been already created add to new
        this json file.
        """

        # New dir and json file
        json_path = self.path.joinpath('annotations').joinpath('detections.json')

        # Create new JSON file in case when, it already exists because of data loss.
        if not json_path.exists():
            self.create_json_file()

        for annotation in labels:
            self.append_item_to_json_file(annotation, json_path)

    def create_json_file(self) -> None:
        """
        Method which creates JSON file with included our structure of annotations.
        """
        self.path.joinpath('annotations').mkdir(exist_ok=True)
        json_path = self.path.joinpath('annotations').joinpath('detections.json')
        json_path.touch(exist_ok=True)

        # Basic structure of JSON string
        json_str = '{"content": []}'

        # Writing of the JSON structure
        f = open(json_path, 'w')
        f.write(json_str)
        f.close()

    def append_item_to_json_file(self, label: str, path: str) -> None:
        """
        Method which appends new lables to the JSON file.
        """

        with open(path) as json_file:
            data = json.load(json_file)
            temp = data['content']
            temp.append(label)

        self.write_to_file(path, data)
        """
            #with open(path, 'w') as f:
            #json.dump(data, f, indent=4)
            #json.dump(data, f)
        """

    def create_json_from_detections(self, annotations, imgs, path: str, indent=None) -> Optional[str]:

        data = {'content': []}

        for i_img in range(len(imgs)):
            data['content'].append({
                'file_name': f'{pathlib.Path(imgs[i_img]).name}',
                'annotations': []
            })
            for annotation in annotations[i_img]:
                data['content'][i_img]['annotations'].append(annotation.build_dictionary())

        # print(json.dumps(data, indent=4))
        print(json.dumps(data, indent=indent))
        #self.write_to_file(path, data, indent=indent)
        #return json.dumps(data)

    @staticmethod
    def write_to_file(path: str, data, indent=0) -> None:
        with open(path, 'w') as f:
            # f.write(json.dumps(data))
            json.dump(data, f, indent=indent)
            f.close()

