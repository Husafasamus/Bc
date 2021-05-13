import sys
import json
from pathlib import Path
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

images_path_dir       = Path(r'D:\Vehicles_with_lpns_dataset\images')
annotations_path_file = Path(r'D:\Vehicles_with_lpns_dataset\annotations\detections.json')

# TODO Rozdelenia confidences historgram, aku distribuciu ma dataset
def confidences_histogram() -> None:
    plt.style.use('fivethirtyeight')
    confidences = []

    with open(annotations_path_file) as detetections:
        images = json.load(detetections)

    for img in images['content']:
        for annotations in img['annotations']:
            confidences.append(annotations['confidence'])

    bins = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 , 1.0]
    plt.hist(confidences, bins=bins, edgecolor='black' )

    plt.title('Confidences of bboxes')
    plt.xlabel('Confidences')
    plt.ylabel('Total bboxes')

    plt.tight_layout()
    plt.show()

# TODO Velkosti obrazkov histogram, aspect ratio
def aspect_ratio_images_histogram():
    aspect_ratio_images = []

    for img_path in images_path_dir.iterdir():
        # image w, h
        with Image.open(img_path.__str__(), 'r') as img:
            w, h = img.size
        aspect_ratio_images.append(get_aspect_ratio(w, h))

    bins = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    plt.hist(aspect_ratio_images, bins=bins, edgecolor='black')

    plt.title('Aspect ratio of images')
    plt.xlabel('aspect ratio')
    plt.ylabel('Total images')

    plt.tight_layout()
    plt.show()

    pass

# TODO BBOX / obsah obrazkov histogram
def bboxes_area_in_image_histogram():
    plt.style.use('fivethirtyeight')
    all_bboxes_area_img = []

    with open(annotations_path_file) as detetections:
        images = json.load(detetections)

    for img_path in images_path_dir.iterdir():
        # image w, h
        with Image.open(img_path.__str__(), 'r') as img:
            w, h = img.size
        img_area = get_area(w, h)
        # bbox w,h
        img_annotations = get_annotations(img_path.name, images)
        if img_annotations is not None:
            for annotation in img_annotations:
                bbox_area = get_area(annotation['bbox']['width'], annotation['bbox']['height'])
                all_bboxes_area_img.append((bbox_area/img_area))

    bins = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    plt.hist(all_bboxes_area_img, bins=bins, edgecolor='black')

    plt.title('Areas of bboxes')
    plt.xlabel('BBox_area / Img_area')
    plt.ylabel('Total bboxes')

    plt.tight_layout()
    plt.show()

def get_aspect_ratio(width: int, height: int):
    return height / width

def get_area(width: int, height: int):
    return width * height

def get_annotations(img, annotations):
    for file_annotations in annotations['content']:
        if file_annotations['file_name'] == img:
            return file_annotations['annotations']

    return None

def main():
    #confidences_histogram()
    #bboxes_area_in_image_histogram()
    #aspect_ratio_images_histogram()

    return 0

if __name__ == "__main__":
    print('Roztrhaj to!')
    sys.exit(main())




