from osgeo import gdal
from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import numpy as np
import skimage.io
import tifffile
import matplotlib.pyplot as plt
import pylab
import random
import os
import pdb
pylab.rcParams['figure.figsize'] = (8.0, 10.0)


data_directory = "/nr/samba/jodata10/pro/autokart/usr/maria/framefield/data/mapping_challenge_dataset/raw/train"
annotation_file_template = "{}/{}/annotation{}.json"

coco = COCO(data_directory + "/annotation-small.json")
for key in coco.dataset:
	print(key)

# This generates a list of all `image_ids` available in the dataset
image_ids = coco.getImgIds()

random_image_id = random.choice(image_ids)
img = coco.loadImgs(random_image_id)[0]
annotation_ids = coco.getAnnIds(imgIds=img['id'])
annotations = coco.loadAnns(annotation_ids)

print(annotations)
