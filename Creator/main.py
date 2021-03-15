from Creator import Dataset as ds
from Creator import ObjectDetector as od

import pathlib
import json
import shutil


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




r"""
path_to_datasets = pathlib.Path(r'D:\bakalarkaaaa\Datasets')

pathlib.Path()


if not path_to_datasets.exists():
    exit("Path to datasets does not exists!")

# Vytvorenie priecinku pre mergnuty dataset
pathlib.Path(r'D:\bakalarkaaaa\Merget_datasets').mkdir(exist_ok=True)

#
unit = ds.Unit(r'D:\bakalarkaaaa\Merget_datasets')

# Inicializacia datasetov
ds1 = ds.FakeDataset1(r'D:\bakalarkaaaa\Datasets\Fake_dataset_1')
ds2 = ds.FakeDataset1(r'D:\bakalarkaaaa\Datasets\Fake_dataset_2')

# Najde si cesty ku fotkam
ds1.find_images()
ds2.find_images()

# Provizorne vycuca lable uz vytvoreneho datasetu
unit.add_labels_from_ds(ds1.get_labels()['content'])
unit.add_labels_from_ds(ds2.get_labels()['content'])

# Prekopiruje obrazky z datasetu
unit.copy_imgs_from_ds(ds1)
unit.copy_imgs_from_ds(ds2)

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
plt.show()"""