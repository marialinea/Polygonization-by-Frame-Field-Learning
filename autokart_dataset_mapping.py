import torch


from torch_lydorn.torch.utils.data import Dataset as LydornDataset, makedirs, files_exist, __repr__

from torch_lydorn.torchvision.datasets import utils


from lydorn_utils import print_utils
from lydorn_utils import python_utils

from multiprocess import Pool
import shapely.geometry

from functools import partial
from pycocotools.coco import COCO
from tqdm import tqdm

import os
import warnings
import numpy as np
import pathlib
import tifffile
import pdb

class Autokart(LydornDataset):
    def __init__(self, root, transform=None, pre_transform=None, fold="train", small=False, pool_size=1):
        assert fold in ["train", "val", "test_images"], "Input fold={} should be in [\"train\", \"val\", \"test_images\"]".format(fold)
        if fold == "test_images":
            print_utils.print_error("ERROR: fold {} not yet implemented!".format(fold))
            exit()
        self.root = root
        self.fold = fold
        makedirs(self.processed_dir)
        self.small = small
        self.pool_size = pool_size


        self.coco = None
        self.image_id_list = self.load_image_ids()
        self.stats_filepath = os.path.join(self.processed_dir, "stats.pt")
        self.stats = None
        if os.path.exists(self.stats_filepath):
            self.stats = torch.load(self.stats_filepath)
        self.processed_flag_filepath = os.path.join(self.processed_dir, "processed-flag")

        super(Autokart, self).__init__(root, transform, pre_transform)

    def load_image_ids(self):
        image_id_list_filepath = os.path.join(self.processed_dir, "image_id_list.json")

        if os.path.exists(image_id_list_filepath):
            image_id_list = python_utils.load_json(image_id_list_filepath)
        else:
            coco = self.get_coco()
            image_id_list = coco.getImgIds()
        # Save for later so that the whole coco object doesn't have to be instantiated
        # when just reading processed samples with multiple workers:
        python_utils.save_json(image_id_list_filepath, image_id_list)
        return image_id_list

    def get_coco(self):
        if self.coco is None:
            annotation_filename =  "annotation.json"
            annotations_filepath = os.path.join(self.root,"raw", self.fold, annotation_filename)
            self.coco = COCO(annotations_filepath)
        return self.coco

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed', self.fold)

    @property
    def processed_file_names(self):
        l = []
        for image_id in self.image_id_list:
            l.append(os.path.join("data_{:012d}.pt".format(image_id)))
        return l

    def __len__(self):
        return len(self.image_id_list)

    def _download(self):
        pass

    def download(self):
        pass

    def _process(self):
        f = os.path.join(self.processed_dir, 'pre_transform.pt')
        if os.path.exists(f) and torch.load(f) != __repr__(self.pre_transform):
            warnings.warn(
                'The `pre_transform` argument differs from the one used in '
                'the pre-processed version of this dataset. If you really '
                'want to make use of another pre-processing technique, make '
                'sure to delete `{}` first.'.format(self.processed_dir))
        f = os.path.join(self.processed_dir, 'pre_filter.pt')
        if os.path.exists(f) and torch.load(f) != __repr__(self.pre_filter):
            warnings.warn(
                'The `pre_filter` argument differs from the one used in the '
                'pre-processed version of this dataset. If you really want to '
                'make use of another pre-fitering technique, make sure to '
                'delete `{}` first.'.format(self.processed_dir))

        if os.path.exists(self.processed_flag_filepath):
            return

        print('Processing...')

        makedirs(self.processed_dir)
        self.process()

        path = os.path.join(self.processed_dir, 'pre_transprocessed_dirform.pt')
        torch.save(__repr__(self.pre_transform), path)
        path = os.path.join(self.processed_dir, 'pre_filter.pt')
        torch.save(__repr__(self.pre_filter), path)

        print('Done!')

    def process(self):
        images_relative_dirpath = os.path.join("raw", self.fold, "images")

        image_info_list = []
        coco = self.get_coco()
        for image_id in self.image_id_list:
            filename = coco.loadImgs(image_id)[0]["file name"]
            annotation_ids = coco.getAnnIds(imgIds=image_id)
            annotation_list = coco.loadAnns(annotation_ids)
            image_info = {
                "image_id": image_id,
                "image_filepath": os.path.join(self.root, images_relative_dirpath, filename),
                "image_relative_filepath": os.path.join(images_relative_dirpath, filename),
                "annotation_list": annotation_list
            }
            image_info_list.append(image_info)


        partial_preprocess_one = partial(preprocess_one, pre_filter=self.pre_filter, pre_transform=self.pre_transform, processed_dir=self.processed_dir)

        with Pool(self.pool_size) as p:
            #pdb.set_trace()
            sample_stats_list = list(tqdm(p.imap(partial_preprocess_one, image_info_list), total=len(image_info_list)))

        """
        count = 0
        for image_info in image_info_list:
            print("{}/{}".format(count + 1, len(image_info_list)), end="\r")
            sample_stats_list = list(preprocess_one(image_info, pre_filter=self.pre_filter, pre_transform=self.pre_transform, processed_dir=self.processed_dir))
            count += 1
        """
        # Aggregate sample_stats_list
        image_s0_list, image_s1_list, image_s2_list, class_freq_list = zip(*sample_stats_list)
        image_s0_array = np.stack(image_s0_list, axis=0)
        image_s1_array = np.stack(image_s1_list, axis=0)
        image_s2_array = np.stack(image_s2_list, axis=0)
        class_freq_array = np.stack(class_freq_list, axis=0)

        image_s0_total = np.sum(image_s0_array, axis=0)
        image_s1_total = np.sum(image_s1_array, axis=0)
        image_s2_total = np.sum(image_s2_array, axis=0)

        image_mean = image_s1_total / image_s0_total
        image_std = np.sqrt(image_s2_total/image_s0_total - np.power(image_mean, 2))
        class_freq = np.sum(class_freq_array*image_s0_array[:, None], axis=0) / image_s0_total

        # Save aggregated stats
        self.stats = {
            "image_mean": image_mean,
            "image_std": image_std,
            "class_freq": class_freq,
        }
        torch.save(self.stats, self.stats_filepath)

        # Indicates that processing has been performed:
        pathlib.Path(self.processed_flag_filepath).touch()

    def get(self, idx):
        image_id = self.image_id_list[idx]
        data = torch.load(os.path.join(self.processed_dir, "data_{:012d}.pt".format(image_id)))
        data["image_mean"] = self.stats["image_mean"]
        data["image_std"] = self.stats["image_std"]
        data["class_freq"] = self.stats["class_freq"]
        return data


def preprocess_one(image_info, pre_filter, pre_transform, processed_dir):
    #pdb.set_trace()
    out_filepath = os.path.join(processed_dir, "data_{:012d}.pt".format(image_info["image_id"]))
    data = None
    #pdb.set_trace()

    if os.path.exists(out_filepath):
        # Load already-processed sample
        try:
            data = torch.load(out_filepath)
        except EOFError:
            pass

    if data is None:
        #pdb.set_trace()
        # Process sample:
        image = tifffile.imread(image_info["image_filepath"])
        gt_polygons = []
        for annotation in image_info["annotation_list"]:
            flattened_segmentation_list = annotation["segmentation"]
            if len(flattened_segmentation_list) != 1:
                print("WHAT!?!, len(flattened_segmentation_list = {}".format(len(flattened_segmentation_list)))
                raise NotImplementedError
            flattened_arrays = np.array(flattened_segmentation_list)
            coords = np.reshape(flattened_arrays, (-1, 2))
            polygon = shapely.geometry.Polygon(coords)

            # Filter out degenerate polygons (area is lower than 2.0)
            if 2.0 < polygon.area:
                gt_polygons.append(polygon)

        data = {
            "image": image,
            "gt_polygons": gt_polygons,
            "image_relative_filepath": image_info["image_relative_filepath"],
            "name": os.path.splitext(os.path.basename(image_info["image_relative_filepath"]))[0],
            "image_id": image_info["image_id"]
        }

        if pre_filter is not None and not pre_filter(data):
            return

        if pre_transform is not None:
            data = pre_transform(data)

        # masked_angles = data["gt_crossfield_angle"].astype(np.float) * data["gt_polygons_image"][:, :, 1].astype(np.float)
        # skimage.io.imsave("gt_crossfield_angle.png", data["gt_crossfield_angle"]
        # skimage.io.imsave("masked_angles.png", masked_angles)
        # exit()


        torch.save(data, out_filepath)
        print("saved = ", image_info["image_id"])

    # Compute stats for later aggregation for the whole dataset

    normed_image = data["image"] / 255
    image_s0 = data["image"].shape[0] * data["image"].shape[1]  # Number of pixels
    image_s1 = np.sum(normed_image, axis=(0, 1))  # Sum of pixel normalized values
    image_s2 = np.sum(np.power(normed_image, 2), axis=(0, 1))
    class_freq = np.mean(data["gt_polygons_image"], axis=(0, 1)) / 255

    return image_s0, image_s1, image_s2, class_freq


def main():
    # Test using transforms from the frame_field_learning project:
    from frame_field_learning import data_transforms

    config = {
        "data_dir_candidates": [
                "/data/titane/user/nigirard/data",
                "./data",
                "/data"
        ],
        "dataset_params": {
            "small": True,
            "root_dirname": "autokart_dataset",
            "seed": 0,
            "train_fraction": 0.75
        },
        "num_workers": 8,
        "data_aug_params": {
            "enable": False,
            "vflip": True,
            "affine": True,
            "color_jitter": True,
            "device": "cuda"
        }
    }

    # Find data_dir
    data_dir = python_utils.choose_first_existing_path(config["data_dir_candidates"])
    if data_dir is None:
        print_utils.print_error("ERROR: Data directory not found!")
        exit()
    else:
        print_utils.print_info("Using data from {}".format(data_dir))
    root_dir = os.path.join(data_dir, config["dataset_params"]["root_dirname"])

    # --- Transforms: --- #
    # --- pre-processing transform (done once then saved on disk):
    # --- Online transform done on the host (CPU):
    train_online_cpu_transform = data_transforms.get_online_cpu_transform(config,
                                                                          augmentations=config["data_aug_params"][
                                                                              "enable"])
    test_online_cpu_transform = data_transforms.get_eval_online_cpu_transform()

    train_online_cuda_transform = data_transforms.get_online_cuda_transform(config,
                                                                            augmentations=config["data_aug_params"][
                                                                                "enable"])
    # --- --- #

    dataset = Autokart(root_dir,
                       transform=test_online_cpu_transform,
                       pre_transform=data_transforms.get_offline_transform_patch(),
                       fold="val",
                       small=config["dataset_params"]["small"],
                       pool_size=config["num_workers"])


    print("# --- Sample 0 --- #")
    sample = dataset[0]
    print(sample.keys())

    for key, item in sample.items():
        print("{}: {}".format(key, type(item)))

    print(sample["image"].shape)
    #print(len(sample["gt_polygons_image"]))
    print("# --- Samples --- #")
    # for data in tqdm(dataset):
    #     pass
    """
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=config["num_workers"])
    print("# --- Batches --- #")
    for batch in tqdm(data_loader):
        print("Images:")
        print(batch["image_relative_filepath"])
        print(batch["image"].shape)
        print(batch["gt_polygons_image"].shape)

        print("Apply online tranform:")
        batch = utils.batch_to_cuda(batch)
        batch = train_online_cuda_transform(batch)
        batch = utils.batch_to_cpu(batch)

        print(batch["image"].shape)
        print(batch["gt_polygons_image"].shape)

        # Save output to visualize
        seg = np.array(batch["gt_polygons_image"][0])
        seg = np.moveaxis(seg, 0, -1)
        seg_display = utils.get_seg_display(seg)
        seg_display = (seg_display * 255).astype(np.uint8)
        skimage.io.imsave("gt_seg.png", seg_display)
        skimage.io.imsave("gt_seg_edge.png", seg[:, :, 1])

        im = np.array(batch["image"][0])
        im = np.moveaxis(im, 0, -1)
        skimage.io.imsave('im.png', im)

        gt_crossfield_angle = np.array(batch["gt_crossfield_angle"][0])
        gt_crossfield_angle = np.moveaxis(gt_crossfield_angle, 0, -1)
        skimage.io.imsave('gt_crossfield_angle.png', gt_crossfield_angle)

        distances = np.array(batch["distances"][0])
        distances = np.moveaxis(distances, 0, -1)
        skimage.io.imsave('distances.png', distances)

        sizes = np.array(batch["sizes"][0])
        sizes = np.moveaxis(sizes, 0, -1)
        skimage.io.imsave('sizes.png', sizes)

        # valid_mask = np.array(batch["valid_mask"][0])
        # valid_mask = np.moveaxis(valid_mask, 0, -1)
        # skimage.io.imsave('valid_mask.png', valid_mask)

        input("Press enter to continue...")



"""

"""
class Autokart(LydornDataset):

    def __init__(self, root, small=False, transform=None, pre_transform=None, fold="train", pool_size=1):
        assert fold in ["train", "val", "test"], "Input fold={} should be in [\"train\", \"val\", \"test\"]".format(fold)

        self.root = root            # root directory
        self.fold = fold
        self.pool_size = pool_size  # enda litt usikker pa hva denne er for
        self.transform = transform
        #self.pre_transform = pre_transform
        #self.pre_filter = None

        # Make directory for processed images

        os.makedirs(os.path.expanduser(os.path.normpath(self.processed_dir)), exist_ok=True)


        self.coco = None
        self.image_id_list = self.load_image_ids()

        # Litt usikker pa hva denne seksjonen her skal holde, og om den er brukbar i det hele tatt
        ###############################################################################
        self.stats_filepath = os.path.join(self.processed_dir, "stats.pt")
        self.stats = None
        if os.path.exists(self.stats_filepath):
            self.stats = torch.load(self.stats_filepath)
        self.processed_flag_filepath = os.path.join(self.processed_dir,"processed-flag")
        #################################################################################



    def load_image_ids(self):
        image_id_list_filepath = os.path.join(self.processed_dir, "image_id_list.json")

        if os.path.exists(image_id_list_filepath):
            image_id_list = python_utils.load_json(image_id_list_filepath)
        else:
            coco = self.get_coco()
            image_id_list = coco.getImgIds()
        # Save for later so that the whole coco object doesn't have to be instantiated
        # when just reading processed samples with multiple workers:
        python_utils.save_json(image_id_list_filepath, image_id_list)
        return image_id_list



    def get_coco(self):
        if self.coco is None:
            annotation_filename =  "annotation.json"
            annotations_filepath = os.path.join(self.root,"raw", self.fold, annotation_filename)
            self.coco = COCO(annotations_filepath)
        return self.coco


    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed', self.fold)

    @property
    def processed_file_names(self):
        l = []
        for image_id in self.image_id_list:
            l.append(os.path.join("data_{:012d}.pt".format(image_id)))
        return l


    def __len__(self):
        return len(self.image_id_list)

    def __getitem__(self, idx):
        image_id = self.image_id_list[idx]
        data = torch.load(os.path.join(self.processed_dir, "data_{:012d}.pt".format(image_id)))
        data["image_mean"] = self.stats["image_mean"]
        data["image_std"] = self.stats["image_std"]
        return data


    def _download(self):
        pass

    def download(self):
        pass


    def _process(self):
        f = os.path.join(self.processed_dir, 'pre_transform.pt')
        if os.path.exists(f) and torch.load(f) != __repr__(self.pre_transform):
            warnings.warn(
                'The `pre_transform` argument differs from the one used in '
                'the pre-processed version of this dataset. If you really '
                'want to make use of another pre-processing technique, make '
                'sure to delete `{}` first.'.format(self.processed_dir))
        f = os.path.join(self.processed_dir, 'pre_filter.pt')
        if os.path.exists(f) and torch.load(f) != __repr__(self.pre_filter):
            warnings.warn(
                'The `pre_filter` argument differs from the one used in the '
                'pre-processed version of this dataset. If you really want to '
                'make use of another pre-fitering technique, make sure to '
                'delete `{}` first.'.format(self.processed_dir))

        if os.path.exists(self.processed_flag_filepath):
            return

        print('Processing...')

        self.process()

        path = os.path.join(self.processed_dir, 'pre_transform.pt')
        torch.save(__repr__(self.pre_transform), path)
        path = os.path.join(self.processed_dir, 'pre_filter.pt')
        torch.save(__repr__(self.pre_filter), path)

        print('Done!')


    def process(self):
        images_relative_dirpath = os.path.join("raw", self.fold, "images")

        image_info_list = []
        coco = self.get_coco()
        for image_id in self.image_id_list:
            filename = coco.loadImgs(image_id)[0]["file name"]
            annotation_ids = coco.getAnnIds(imgIds=image_id)
            annotation_list = coco.loadAnns(annotation_ids)
            image_info = {
                "image_id": image_id,
                "image_filepath": os.path.join(self.root, images_relative_dirpath, filename),
                "image_relative_filepath": os.path.join(images_relative_dirpath, filename),
                "annotation_list": annotation_list
            }
            image_info_list.append(image_info)


        partial_preprocess_one = partial(preprocess_one, pre_filter=self.pre_filter, pre_transform=self.pre_transform, processed_dir=self.processed_dir)

        with Pool(self.pool_size) as p:
            sample_stats_list = list(tqdm(p.imap(partial_preprocess_one, image_info_list), total=len(image_info_list)))

        # Aggregate sample_stats_list
        image_s0_list, image_s1_list, image_s2_list = zip(*sample_stats_list)
        image_s0_array = np.stack(image_s0_list, axis=0)
        image_s1_array = np.stack(image_s1_list, axis=0)
        image_s2_array = np.stack(image_s2_list, axis=0)
        #class_freq_array = np.stack(class_freq_list, axis=0)

        image_s0_total = np.sum(image_s0_array, axis=0)
        image_s1_total = np.sum(image_s1_array, axis=0)
        image_s2_total = np.sum(image_s2_array, axis=0)

        image_mean = image_s1_total / image_s0_total
        image_std = np.sqrt(image_s2_total/image_s0_total - np.power(image_mean, 2))
       # class_freq = np.sum(class_freq_array*image_s0_array[:, None], axis=0) / image_s0_total

        # Save aggregated stats
        self.stats = {
            "image_mean": image_mean,
            "image_std": image_std,
            #"class_freq": class_freq,
        }
        torch.save(self.stats, self.stats_filepath)


        # Indicates that processing has been performed:
        pathlib.Path(self.processed_flag_filepath).touch()






def preprocess_one(image_info, pre_filter, pre_transform, processed_dir):
    out_filepath = os.path.join(processed_dir, "data_{:012d}.pt".format(image_info["image_id"]))
    data = None
    if os.path.exists(out_filepath):
        # Load already-processed sample
        try:
            data = torch.load(out_filepath)
        except EOFError:
            pass
    if data is None:
        # Process sample:
        image = tifffile.imread(image_info["image_filepath"])  # For different file formats use skimage.io.imread
        gt_polygons = []
        for annotation in image_info["annotation_list"]:
            flattened_segmentation_list = annotation["segmentation"]
            if len(flattened_segmentation_list) != 1:
                print("WHAT!?!, len(flattened_segmentation_list = {}".format(len(flattened_segmentation_list)))
                raise NotImplementedError
            flattened_arrays = np.array(flattened_segmentation_list)
            coords = np.reshape(flattened_arrays, (-1, 2))
            polygon = shapely.geometry.Polygon(coords)

            # Filter out degenerate polygons (area is lower than 2.0)
            if 2.0 < polygon.area:
                gt_polygons.append(polygon)

        data = {
            "image": image,
            "gt_polygons": gt_polygons,
            "image_relative_filepath": image_info["image_relative_filepath"],
            "name": os.path.splitext(os.path.basename(image_info["image_relative_filepath"]))[0],
            "image_id": image_info["image_id"]
        }

        if pre_filter is not None and not pre_filter(data):
            return

        if pre_transform is not None:
            data = pre_transform(data)

        # masked_angles = data["gt_crossfield_angle"].astype(np.float) * data["gt_polygons_image"][:, :, 1].astype(np.float)
        # skimage.io.imsave("gt_crossfield_angle.png", data["gt_crossfield_angle"])
        # skimage.io.imsave("masked_angles.png", masked_angles)
        # exit()

        torch.save(data, out_filepath)

    # Compute stats for later aggregation for the whole dataset
    normed_image = data["image"] / 255
    image_s0 = data["image"].shape[0] * data["image"].shape[1]  # Number of pixels
    image_s1 = np.sum(normed_image, axis=(0, 1))  # Sum of pixel normalized values
    image_s2 = np.sum(np.power(normed_image, 2), axis=(0, 1))
    #class_freq = np.mean(data["gt_polygons"], axis=(0, 1)) / 255

    return image_s0, image_s1, image_s2 #, class_freq


def main():
    # Test using transforms from the frame_field_learning project:
    from frame_field_learning import data_transforms

    config = {
        "data_dir_candidates": ["./data"],
        "dataset_params":{
            "small": False,
            "root_dirname": "autokart_dataset",
            "seed": 0,
            "train_fraction": 0.75
        },
        "num_workers": 8,
        "data_aug_params": {
            "enable": False,
            "scaling": 0.75,
            "vflip": True,
            "affine": True,
            "color_jitter": True,
            "device": "cuda"
        }
    }


    # Find data_dir
    data_dir = python_utils.choose_first_existing_path(config["data_dir_candidates"])
    if data_dir is None:
        print_utils.print_error("ERROR: Data directory not found!")
        exit()
    else:
        print_utils.print_info("Using data from {}".format(data_dir))
    root_dir = os.path.join(data_dir, config["dataset_params"]["root_dirname"])

    # --- Transforms: --- #
    # --- pre-processing transform (done once then saved on disk):
    # --- Online transform done on the host (CPU):
    train_online_cpu_transform = data_transforms.get_online_cpu_transform(config,augmentations=config["data_aug_params"]["enable"])
    test_online_cpu_transform = data_transforms.get_eval_online_cpu_transform()

    train_online_cuda_transform = data_transforms.get_online_cuda_transform(config,augmentations=config["data_aug_params"]["enable"])

    # --- --- #
    pdb.set_trace()
    dataset = Autokart(root_dir,
                       transform=test_online_cpu_transform,
                       pre_transform=data_transforms.get_offline_transform_patch(),
                       fold="train",
                       small=config["dataset_params"]["small"],
                       pool_size=config["num_workers"])
    pdb.set_trace()#dataset.process()
    print("# --- Sample 0 --- #")
    sample = dataset[0]
    print(sample.keys())

    for key, item in sample.items():
        print("{}: {}".format(key, type(item)))

    print(sample["image"].shape)
    print("# --- Samples --- #")
    # for data in tqdm(dataset):
    #     pass

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=config["num_workers"])
    print("# --- Batches --- #")
    for batch in tqdm(data_loader):
        print("Images:")
        print(batch["image_relative_filepath"])
        print(batch["image"].shape)
        print(batch["gt_polygons_image"].shape)

        print("Apply online tranform:")
        batch = utils.batch_to_cuda(batch)
        batch = train_online_cuda_transform(batch)
        batch = utils.batch_to_cpu(batch)

        print(batch["image"].shape)
        print(batch["gt_polygons_image"].shape)

        # Save output to visualize
        seg = np.array(batch["gt_polygons_image"][0])
        seg = np.moveaxis(seg, 0, -1)
        seg_display = utils.get_seg_display(seg)
        seg_display = (seg_display * 255).astype(np.uint8)
        skimage.io.imsave("gt_seg.png", seg_display)
        skimage.io.imsave("gt_seg_edge.png", seg[:, :, 1])

        im = np.array(batch["image"][0])
        im = np.moveaxis(im, 0, -1)
        skimage.io.imsave('im.png', im)

        gt_crossfield_angle = np.array(batch["gt_crossfield_angle"][0])
        gt_crossfield_angle = np.moveaxis(gt_crossfield_angle, 0, -1)
        skimage.io.imsave('gt_crossfield_angle.png', gt_crossfield_angle)

        distances = np.array(batch["distances"][0])
        distances = np.moveaxis(distances, 0, -1)
        skimage.io.imsave('distances.png', distances)

        sizes = np.array(batch["sizes"][0])
        sizes = np.moveaxis(sizes, 0, -1)
        skimage.io.imsave('sizes.png', sizes)

        # valid_mask = np.array(batch["valid_mask"][0])
        # valid_mask = np.moveaxis(valid_mask, 0, -1)
        # skimage.io.imsave('valid_mask.png', valid_mask)

        input("Press enter to continue...")

        #"""




if __name__ == '__main__':
    main()
