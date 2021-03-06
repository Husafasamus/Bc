import pathlib
import json
from typing import Optional

class Dataset:

    def __init__(self, path: str) -> None:
        self.path = pathlib.Path(path)
        self.name = self.path.name
        self.imgs_path = []

    def find_images(self) -> None:
        print(f"{self.name}'s function findImages() has not been defined!")

    def get_labels(self) -> Optional[bool]:
        """ If this method return None, it means it has to be annotated."""
        return None

    def __repr__(self) -> str:
        return f'{self.name} in: {self.path}'


class FakeDataset1(Dataset):

    def __init__(self, path: str) -> None:
        super().__init__(path)
        # self.paths_to_images = []

    """ This method gives me paths to all images, which dataset contains."""

    def find_images(self) -> None:
        for path in pathlib.Path(self.path).iterdir():
            if 'labels' not in path.name:
                self.imgs_path.append(path)

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

class FakeDataset2(Dataset):
    """ Typ datasetu, cez ktory musia prebehnut detektory"""
    def __init__(self, path: str) -> None:
        super().__init__(path)
        # self.paths_to_images = []

    """ This method gives me paths to all images, which dataset contains."""

    def find_images(self) -> None:
        for path in pathlib.Path(self.path).iterdir():
            if 'Our_detections' not in path.name:
                self.imgs_path.append(path)

    """
    This method gives me labels, which are already created.
    And transfer them into our convention.
    """

    def get_labels(self) -> Optional[bool]:
        return None

