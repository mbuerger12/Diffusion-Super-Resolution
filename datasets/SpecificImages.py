import os
import tifffile
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from osgeo import gdal
import torch
from .geo_tifffile import read_geotiff3D, write_geotiff3D

class SpecificImages(Dataset):
    """
    Dataset class for Sentinel-2 and SPOT-6 images

    Args:
        images (list): List of paths to Sentinel-2 images
        labels (list): List of paths to SPOT-6 images
        transform (torchvision.transforms.Compose): List of transforms to apply to the images
        norm_params (dict): Dictionary with normalization parameters for each location
    """

    def __init__(self, images, labels, bathymetry=False, transform=None, target_transform = None, norm_params = None):

        self.images = images
        self.labels = labels
        self.transform = transform
        self.target_trans = target_transform
        self.norm_params = norm_params
        self.target_trans = target_transform
        self.bathymetry = bathymetry


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.labels[idx]
        
        img = tifffile.imread(img_path).astype(np.float32)
        label = tifffile.imread(label_path).astype(np.float32)

        # Normalization
        if "agia_napa" in img_path:
            norm_param_s2 = self.norm_params["s2_an"]
            norm_param_spot6 = self.norm_params["spot6_an"]
            norm_param_depth = -30.443
        elif "puck_lagoon" in img_path:
            norm_param_s2 = self.norm_params["s2_pl"]
            norm_param_spot6 = self.norm_params["spot6_pl"]
            norm_param_depth = -11.0

        if self.bathymetry:
            img = img/norm_param_depth
            label = label/norm_param_depth
        else:
            img = (img - norm_param_s2[0]) / (norm_param_s2[1] - norm_param_s2[0])
            label = (label - norm_param_spot6[0]) / \
                (norm_param_spot6[1] - norm_param_spot6[0])
        
        img = np.clip(img, 0, 1)
        label = np.clip(label, 0, 1)

        # Swap from BGR to RGB
        if not self.bathymetry:
            img = img[..., [2, 1, 0]]

        if self.transform:
            img = self.transform(img)

        if self.target_trans:
            label = self.target_trans(label)
        else:
            label = self.transform(label)

        return img_path, label_path, img.to(torch.float32), label.to(torch.float32)
        
    def denormalize(self, img, path, is_label=False, bathy=False):
        """
        Denormalizes the input image based on the given path and normalization parameters.

        Args:
            img (numpy.ndarray): The input image to be denormalized. Values between 0-1. Shape H x W x C.
            path (str): The original path of the image.
            is_label (bool, optional): Specifies whether the image is a label. Defaults to False.

        Returns:
            numpy.ndarray: The denormalized image.

        """

        # Swap from RGB to BGR
        if not bathy:
            img = img[..., [2, 1, 0]]

        if "agia_napa" in path:
            norm_param_s2 = self.norm_params["s2_an"]
            norm_param_spot6 = self.norm_params["spot6_an"]
            norm_param_depth = -30.443
        elif "puck_lagoon" in path:
            norm_param_s2 = self.norm_params["s2_pl"]
            norm_param_spot6 = self.norm_params["spot6_pl"]
            norm_param_depth = -11.0

        if not bathy:
            if is_label:
                img = img * (norm_param_spot6[1] - norm_param_spot6[0]) + norm_param_spot6[0]
            else:
                img = img * (norm_param_s2[1] - norm_param_s2[0]) + norm_param_s2[0]
        else:
            img *= norm_param_depth

        return img
    
    def save_as_tiff(self, img, original_path, save_path):
        """
        Saves the input image as a TIFF file. Denormalizes the image before saving.

        Args:
            img (numpy.ndarray): The input image to be saved. Values between 0-1. Shape H x W x C. RGB
            original_path (str): The original path of the image.
            save_path (str): The path to save the image.

        """
        img = self.denormalize(img, original_path, bathy=self.bathymetry)#.round().astype(np.uint16)
        _, metadata = read_geotiff3D(original_path, self.bathymetry)
        write_geotiff3D(save_path, img, metadata, self.bathymetry)



class SpecificImagesDataLoader:
    """
    Data loader class for Sentinel-2 and SPOT-6 images

    Args:
        root_dir (str): Root directory of the dataset
        transform (torchvision.transforms.Compose): List of transforms to apply to the images
        batch_size (int): Batch size
        num_workers (int): Number of workers for the data loader
        test_size (float): Percentage of the dataset to use for testing
        val_size (float): Percentage of the dataset to use for validation
    """

    def __init__(self, root_dir="data/", transform=None,target_transform=None, batch_size=1, num_workers=0, test_size=0.2, val_size=0.1, bathymetry=False):

        self.root_dir = root_dir
        self.transform = transform
        self.target_trans = target_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_size = test_size
        self.val_size = val_size / (1 - test_size)

        self.bathymetry = bathymetry

        # Load normalization parameters
        # if bathymetry:
        #     self.norm_params = {
        #         "s2_an": np.load("configs/norm_params/norm_param_bathy.npy"),
        #         "s2_pl": np.load("configs/norm_params/norm_param_bathy.npy"),
        #         "spot6_an": np.load("configs/norm_params/norm_param_bathy.npy"),
        #         "spot6_pl": np.load("configs/norm_params/norm_param_bathy.npy")
        #     }
        # else:
        self.norm_params = {
            "s2_an": np.load("configs/norm_params/norm_param_s2_an.npy"),
            "s2_pl": np.load("configs/norm_params/norm_param_s2_pl.npy"),
            "spot6_an": np.load("configs/norm_params/norm_param_spot6_an.npy"),
            "spot6_pl": np.load("configs/norm_params/norm_param_spot6_pl.npy")
        }

        self.datasets = self._prepare_datasets()


    def _load_samples2(self):
        samples = {"images": [], "labels": []}
        locations = ["agia_napa", "puck_lagoon"]
        for location in locations:
            img_dir_s2 = os.path.join(self.root_dir, location, "img", "s2")
            img_dir_spot6 = os.path.join(
                self.root_dir, location, "img", "spot6")
            samples["images"].extend(sorted([os.path.join(
                img_dir_s2, f) for f in os.listdir(img_dir_s2) if f.endswith(".tif")]))
            samples["labels"].extend(sorted([os.path.join(
                img_dir_spot6, f) for f in os.listdir(img_dir_spot6) if f.endswith(".tif")]))
        return samples

    def _load_samples(self):
        samples = {"images": [], "labels": []}
        
        if not self.bathymetry:
            img_dir_s2_an = os.path.join(self.root_dir, "agia_napa", "img", "s2")
            img_dir_spot6_an = os.path.join(
                self.root_dir, "agia_napa", "img", "spot6")
            img_410_path_s2 = os.path.join(img_dir_s2_an, "img_410.tif")
            img_410_path_spot6 = os.path.join(img_dir_spot6_an, "img_410.tif")

            img_dir_s2_pl = os.path.join(self.root_dir, "puck_lagoon", "img", "s2")
            img_dir_spot6_pl = os.path.join(
                self.root_dir, "puck_lagoon", "img", "spot6")
            img_2987_path_s2 = os.path.join(img_dir_s2_pl, "img_2987.tif")
            img_2987_path_spot6 = os.path.join(img_dir_spot6_pl, "img_2987.tif")
        else:
            img_dir_s2_an = os.path.join(self.root_dir, "agia_napa", "depth", "s2")
            img_dir_spot6_an = os.path.join(
                self.root_dir, "agia_napa", "depth", "spot6")
            img_410_path_s2 = os.path.join(img_dir_s2_an, "img_410.tif")
            img_410_path_spot6 = os.path.join(img_dir_spot6_an, "img_410.tif")

            img_dir_s2_pl = os.path.join(self.root_dir, "puck_lagoon", "depth", "s2")
            img_dir_spot6_pl = os.path.join(
                self.root_dir, "puck_lagoon", "depth", "spot6")
            img_2987_path_s2 = os.path.join(img_dir_s2_pl, "img_2987.tif")
            img_2987_path_spot6 = os.path.join(img_dir_spot6_pl, "img_2987.tif")

        if os.path.exists(img_410_path_s2) and os.path.exists(img_410_path_spot6):
            samples["images"].append(img_410_path_s2)
            samples["labels"].append(img_410_path_spot6)

        if os.path.exists(img_2987_path_s2) and os.path.exists(img_2987_path_spot6):
            samples["images"].append(img_2987_path_s2)
            samples["labels"].append(img_2987_path_spot6)

        return samples

    def _prepare_datasets(self):
        samples = self._load_samples()
        images, labels = samples["images"], samples["labels"]

        datasets = {
            "train": SpecificImages(images, labels, self.bathymetry, self.transform, self.target_trans,self.norm_params),
            "val": SpecificImages(images, labels, self.bathymetry, self.transform,self.target_trans, self.norm_params),
            "test": SpecificImages(images, labels, self.bathymetry, self.transform,self.target_trans, self.norm_params)
        }
        return datasets

    def get_dataloader(self, set_type="train"):
        return DataLoader(
            self.datasets[set_type],
            batch_size=self.batch_size,
            shuffle=True if set_type == "train" else False,
            num_workers=self.num_workers
        )
