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
"""
data_directory = "/nr/samba/jodata10/pro/autokart/usr/maria/autokart/building_8cm/building/rgb8cm+lidar-obj-heigt+lidar-obj-int/work/test/ground_truth"
annotation_file_template = "{}/{}/annotation{}.json"

coco = COCO(data_directory + "/annotation.json")


# This generates a list of all `image_ids` available in the dataset
image_ids = coco.getImgIds()

random_image_id = random.choice(image_ids)
img = coco.loadImgs(random_image_id)[0]
annotation_ids = coco.getAnnIds(imgIds=img['id'])
annotations = coco.loadAnns(annotation_ids)
"""



image_dir = "/nr/samba/jodata10/pro/autokart/usr/trier/sandvika_2020_buildings_8cm/building/rgb8cm+lidar-obj-heigt+lidar-obj-int/work/train/orig_size/0004_orig-size.tif"
"""
I = skimage.io.imread(image_dir)
plt.imshow(I)
plt.show()
"""

I = tifffile.imread(image_dir)
plt.imshow(I)
plt.axis("off")
#coco.showAnns(annotations)
plt.show()
