from dataset_merger import dataset as ds
from dataset_merger import dataset_others as ods
from dataset_merger import bbox
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import numbers
from typing import Optional
import pathlib
from PIL import Image
import time
import statistics
import math
from dataset_merger.merging import DatasetMerger

def plot_img_with_xywh(img: str, x: int, y: int, w: int, h: int) -> None:
    fig, ax = plt.subplots(1)
    img = mpimg.imread(img)
    ax.imshow(img)
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    #   rect = patches.Rectangle((764, 335), 260, 230, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)
    # Intersection
    # rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='b', facecolor='blue')
    ax.add_patch(rect)
    plt.show()
def plot_img_with_bbox3(img, bbox1: bbox.BBox, bbox2: bbox.BBox, bbox3: bbox.BBox) -> None:
    x, y, w, h = bbox1.as_xywh()
    fig, ax = plt.subplots(1)
    img = mpimg.imread(img)
    ax.imshow(img)
    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    x, y, w, h = bbox2.as_xywh()
    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)
    # Intersection

    x, y, w, h = bbox3.as_xywh()
    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='b', facecolor='blue')
    ax.add_patch(rect)
    plt.show()
def plot_img_with_bbox(img, bbox1: bbox.BBox) -> None:
    x, y, w, h = bbox1.as_xywh()
    fig, ax = plt.subplots(1)
    img = mpimg.imread(img)
    ax.imshow(img)
    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()




def compute_2D_heat_map(dataset: ds.Dataset=0, conf_from=0.3,
                        conf_to=1, conf_step=0.1, intersection_from=0.4, intersection_to=1,intersection_step=0.1):

    conf_dif = (conf_to - conf_from)
    num_conf_steps = math.ceil(conf_dif / conf_step) + 1
    inters_dif = (intersection_to - intersection_from)
    num_inters_steps = math.ceil(inters_dif / conf_step) + 1
    c_imgs = 0
    data = [[0 for j in range(num_inters_steps)] for i in range(num_conf_steps)]
    confidences = []
    intersections = []
    tmp_inter_step = intersection_from
    for innter_ann in range(num_inters_steps):
        intersections.append(round(tmp_inter_step, 2))
        tmp_inter_step += intersection_step

    del tmp_inter_step


    conf_act_step = conf_from
    intersection_act_step = intersection_from

    # Confidence
    for index_conf_step in range(num_conf_steps):
        confidences.append(round(conf_act_step, 2))
        for index_inters_step in range(num_inters_steps):
            result = DatasetMerger.compare_detections_n(dataset, confidence_treshold=conf_act_step,
                                               bbox_perc_intersection=intersection_act_step)
            intersection_act_step += intersection_step
            data[index_conf_step][index_inters_step] = result[0]/ (result[0] + result[1])# 0 - Count of imgs manual, 1 - done imgs

        intersection_act_step = intersection_from
        conf_act_step += conf_step


    data_np = np.array(data)

    # Create 2D - Heat map
    fig, ax = plt.subplots()
    im = ax.imshow(data_np)
    fig.colorbar(im, ax=ax, label='Ratio')


    # We want to show all ticks...
    ax.set_xticks(np.arange(len(intersections)))
    ax.set_yticks(np.arange(len(confidences)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(intersections)
    ax.set_yticklabels(confidences)


    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    plt.xlabel('Intersection')
    plt.ylabel('Confidence')



    # Loop over data dimensions and create text annotations.
    for i in range(len(confidences)):
        for j in range(len(intersections)):
            text = ax.text(j, i, data_np[i, j],
                         ha="center", va="center", color="w")


    ax.set_title(f"Relatívny počet obrázkov, potrebných manuálne anotovať\nv závislosti od confidence a intersection")

    fig.tight_layout()

    plt.show()




def main() -> int:
    #merger = DatasetMerger(r'D:\bakalarkaaaa\Merged_datasets')
    #ds1 = ods.ArtificialMercosurLicensePlates(r'D:\Downloads\nx9xbs4rgx-2')
    #ds1.find_images()

    #t0 = time.time()
    #merger.import_dataset(ds1)
    #merger.split_train_test_validation()
    #t1 = time.time() - t0

    d = ds.FakeDataset2(r'D:\bakalarkaaaa\Datasets\Fake_dataset_2')
    d.find_images()
    #print(bbox.BBox(1,2,10,10).capacity())

    compute_2D_heat_map(d)


    #DatasetMerger.compare_detections_n(d, bbox_perc_intersection=0.8)
    #DatasetMerger.compare_detections(d)

    #ObjectDetector.yolov3_vehicle_detector(d)
    #ObjectDetector.SSD_vehicle_detector(d)


    return 0

def main_alt() -> int:
    pass

if __name__ == "__main__":
    import sys


    sys.exit(main())
#   sys.exit(main_alt())

#ds1 = ods.UFPRALPRDataset(r"D:\Downloads\UFPR-ALPR dataset")
    #ds1.find_images()

    #dest = pathlib.Path(r'C:\Users\soky5\Desktop\try')
    #for img_index in range(len(ds1.imgs_path)):
     #   with Image.open(ds1.imgs_path[img_index]) as img:
      #      img.convert('RGB'). \
       #         save(f"{dest.joinpath(f'{img_index:05d}.jpg')}")
        #with Image.open(str(ds1.imgs_path[img_index]), 'r') as img:
         #   w, h = img.size
        #annot = ods.UFPRALPRDataset.get_annotation_from_ufpr_txt_files(ds1.annotations_path[img_index])
        #with open(dest.joinpath(f'{img_index:05d}.txt'), "w") as write:
         #   write.write(f"0 {annot[1].bbox.center[0]/w} {annot[1].bbox.center[1]/h} {annot[1].bbox.size[0]/w} {annot[1].bbox.size[1]/h}")

    #print(f"Time elapsed: {t1}")


import cv2

# h, w, _ = cv2.imread(r'D:\Downloads\nx9xbs4rgx-2\images\cropped_parking_lot_1.JPG').shape
# bbx2 = bbox.BBox.build_from_center_and_size(np.array([int(0.468825*w), int(0.609544*h)]), np.array([int(0.489209*w), int(0.225597*h)]))

# plot_img_with_bbox(r'D:\Downloads\nx9xbs4rgx-2\images\cropped_parking_lot_1.JPG', bbx2)
# ds1 = ods.ArtificialMercosurLicensePlates(r'D:\Downloads\nx9xbs4rgx-2')
# ds1.find_images()
# ds1.get_labels()

# Kordinacie su v v strede a nie top left
# plot_img_with_xywh(r'D:\Downloads\nx9xbs4rgx-2\images\cropped_parking_lot_1.JPG', int(0.468825*834), int(0.609544*461), int(0.489209*834), int(0.225597*461))


unit = ds.Unit(r'D:\bakalarkaaaa\Merget_datasets')
ds1 = ods.ArtificialMercosurLicensePlates(r'D:\Downloads\nx9xbs4rgx-2')
ds1.find_images()
# ds1.get_labels()

# print(ds1.paths_to_images[0])
# print(type(ds1.paths_to_images[0]))

# print(type(r'D:\Downloads\nx9xbs4rgx-2\images\cropped_parking_lot_1.JPG'))
unit.create_json_from_detections(ds1.get_labels(), ds1.paths_to_images, '', indent=4)

"""pathlib.Path(r'D:\bakalarkaaaa\Datasets\Detections').mkdir(exist_ok=True)
pathlib.Path(r'D:\bakalarkaaaa\Datasets\Detections\Fake_dataset_1').mkdir(exist_ok=True)
p = pathlib.Path(r'D:\bakalarkaaaa\Datasets\Detections\Fake_dataset_1\ssd')
p.mkdir(exist_ok=True)
p.joinpath('detections.json').touch(exist_ok=True)
p = pathlib.Path(r'D:\bakalarkaaaa\Datasets\Detections\Fake_dataset_1\ssd\detections.json')
unit.create_json_from_detections(ssd.start(ds1.paths_to_images, 0.4), ds1.paths_to_images,p)"""

"""

# zachovat
""""""
unit = ds.Unit(r'D:\bakalarkaaaa\Merget_datasets')
ds1 = ds.FakeDataset1(r'D:\bakalarkaaaa\Datasets\Fake_dataset_1')
ds1.find_images()

ssd = od.SSD300()

pathlib.Path(r'D:\bakalarkaaaa\Datasets\Detections').mkdir(exist_ok=True)
pathlib.Path(r'D:\bakalarkaaaa\Datasets\Detections\Fake_dataset_1').mkdir(exist_ok=True)
p = pathlib.Path(r'D:\bakalarkaaaa\Datasets\Detections\Fake_dataset_1\ssd')
p.mkdir(exist_ok=True)
p.joinpath('detections.json').touch(exist_ok=True)
p = pathlib.Path(r'D:\bakalarkaaaa\Datasets\Detections\Fake_dataset_1\ssd\detections.json')
unit.create_json_from_detections(ssd.start(ds1.paths_to_images, 0.4), ds1.paths_to_images,p)

"""

"""

import torch
precision = 'fp32'
ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision)

utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

ssd_model.to('cuda')
ssd_model.eval()


uris = ds1.paths_to_images

inputs = [utils.prepare_input(uri) for uri in uris]
tensor = utils.prepare_tensor(inputs, precision == 'fp16')

with torch.no_grad():
    detections_batch = ssd_model(tensor)

results_per_input = utils.decode_results(detections_batch)
best_results_per_input = [utils.pick_best(results, 0.1) for results in results_per_input]

classes_to_labels = utils.get_coco_object_dictionary()


from matplotlib import pyplot as plt
import matplotlib.patches as patches

for image_idx in range(len(best_results_per_input)):
    fig, ax = plt.subplots(1)
    # Show original, denormalized image...
    image = inputs[image_idx] / 2 + 0.5
    ax.imshow(image)
    # ...with detections
    bboxes, classes, confidences = best_results_per_input[image_idx]
    print(bboxes)
    print(classes)
    print(confidences)
    print(image_idx)
    for idx in range(len(bboxes)):
        left, bot, right, top = bboxes[idx]
        x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, "{} {:.0f}%".format(classes_to_labels[classes[idx] - 1], confidences[idx]*100), bbox=dict(facecolor='white', alpha=0.5))
plt.show()
"""
