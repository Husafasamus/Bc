from Creator import Dataset as ds
from Creator.Reader import FileManager as fm
import os
import json

"""
{
    "images": [
        {
        "file_name": "1.jpg",
        "height": 427,
        "width": 640,
        "bbox": [x,y,width,height], ----> ak ich tu je viac?
        "id": 1
        },
        {
        "file_name": "2.jpg",
        "height": 427,
        "width": 640,
        "bbox": [x,y,width,height],
        "id": 2
        },
    ...
    ]

}
"""

class Handler:

    def __init__(self, path, dataset1, dataset2, objDet1):
        self.path = path
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.objdetector = objDet1


    # Zlucenie, prekopirovanie existujucich fotiek z datasetov do novej zlozky
    def merge(self):
        fm.create_dir(rf"{self.path}")

        self.dataset1.find_images()
        self.dataset2.find_images()

        last_index = self.copy_dataset(self.dataset1, self.path, 1)
        self.copy_dataset(self.dataset2, self.path, last_index)
        print("Copy done!")
        print("Labeling has started...")
        self.label_photos()


    def copy_dataset(self, ds, path_to_dir, index):
        index = index
        for img in self.dataset1.images:
            fm.copy_move(img, fr"{path_to_dir}/{index}.jpg")
            index += 1
        return index


    def getpaths_to_images(self):
        images = os.listdir(self.path)
        for x in range(len(images)):
            images[x] = f"{self.path}/{images[x]}"
        return images

    def label_photos(self):
        #m.create_file(f"{self.path}/labels.json")

        #JSON
        data = {}
        data['images'] = []


        for img in self. getpaths_to_images():
            data['images'].append({
                'file_name': f'{img}',
                'isthere': 'true'
            })

        with open(fr'{self.path}/data.json', 'w') as outfile:
            json.dump(data, outfile)

        print("Labeling has been finished!")
