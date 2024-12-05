import os
import tifffile
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch
import time
from .geo_tifffile import read_geotiff3D, write_geotiff3D
from data.utils import bicubic_with_mask


class MagicBathyNet(Dataset):
    """
    Dataset class for Sentinel-2 and SPOT-6 images

    Args:
        images (list): List of paths to Sentinel-2 images
        labels (list): List of paths to SPOT-6 images
        transform (torchvision.transforms.Compose): List of transforms to apply to the images
        norm_params (dict): Dictionary with normalization parameters for each location
    """

    def __init__(self, images, labels, bathymetry_images, bathymetry=False, transform=None, target_transform = None, norm_params = None):

        self.images = images
        self.labels = labels
        self.transform = transform
        self.target_trans = target_transform
        self.norm_params = norm_params
        self.target_trans = target_transform
        self.bathymetry = bathymetry
        self.bathymetry_images = bathymetry_images
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.labels[idx]
        bath_path = self.bathymetry_images[idx]

        img = tifffile.imread(img_path).astype(np.float32)
        label = tifffile.imread(label_path).astype(np.float32)
        bath = tifffile.imread(bath_path).astype(np.float32)
        #bath = torch.tensor(bath)
        if "agia_napa" in img_path:
            norm_param_s2 = self.norm_params["s2_an"]
            norm_param_aerial = self.norm_params["aerial_an"]
            norm_param_depth = -30.443
        elif "puck_lagoon" in img_path:
            norm_param_s2 = self.norm_params["s2_pl"]
            norm_param_spot6 = self.norm_params["aerial_pl"]
            norm_param_depth = -11.0

        img = (img - norm_param_s2[0]) / (norm_param_s2[1] - norm_param_s2[0])
        label = (label - norm_param_aerial[0]) / (norm_param_aerial[1] - norm_param_aerial[0])
        bath = bath / norm_param_depth


        img = img[..., [2, 1, 0]] if "agia_napa" in img_path else img
        img = img.transpose(2, 0, 1)
        img = torch.clamp(torch.tensor(img), 0, 1)
        bath = torch.clamp(torch.tensor(bath), 0, 1)
        print(f"imgmin {img.min()} imgmax {img.max()} label {label.min()} label {label.max()} bath {bath.min()} bath {bath.max()}")

        # Swap from BGR to RGB


        if self.transform:
            img = self.transform(img)

        if self.target_trans:
            label = self.target_trans(label)
        else:
            print("No target transform")
            #label = self.transform(label)
        label = label.transpose(2, 0, 1)
        y = torch.tensor(label).to(torch.float32).unsqueeze(0)
        mask_hr = (y != 0).all(dim=0, keepdim=True).float()
        mask_lr = (img != 0).all(dim=0, keepdim=True).float().unsqueeze(0)
        source = torch.tensor(img).to(torch.float32).unsqueeze(0)
        guide = torch.tensor(bath).to(torch.float32).unsqueeze(0).unsqueeze(0)
        y_bicubic = torch.nn.functional.interpolate(torch.tensor(img).to(torch.float32).unsqueeze(0), size=(512, 512), mode='bicubic', align_corners=True).squeeze()
        y_bicubic = y_bicubic * mask_hr
        return {
            'guide': guide,
            'source': source,
            'y': y,
            'mask_lr': mask_lr,
            'y_bicubic': y_bicubic,
            'mask_hr': mask_hr
        }

        #return img_path, label_path, img.to(torch.float32), label.to(torch.float32)

    def denormalize(self, img, path, is_label=True, bathy=False):
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
            # if img.shape[0] <= 3:
            #     img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))
            # else:
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
            if is_label or "spot6" in path:
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
        


class MagicBathyNetDataLoader:
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

    def __init__(self, root_dir="data/", transform=None, target_transform = None, batch_size=16, num_workers=0, test_size=0.15, val_size=0.15, locations=["agia_napa", "puck_lagoon"], bathymetry=False):

        self.root_dir = root_dir
        self.transform = transform
        self.target_trans = target_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_size = test_size
        self.val_size = val_size / (1 - test_size)

        self.locations = locations
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
            "s2_an": np.load(os.path.join('.','datafolder','resized','agia_napa',"norm_param_s2_an.npy")),
            "aerial_an": np.load(os.path.join('.','datafolder','resized','agia_napa',"norm_param_aerial_an.npy")),
            "s2_pl": np.load(os.path.join('.','datafolder','MagicBathyNet_CV4RS_WiSe_2425','puck_lagoon',"norm_param_s2_pl.npy")),
            "aerial_pl": np.load(os.path.join('.','datafolder','MagicBathyNet_CV4RS_WiSe_2425','puck_lagoon',"norm_param_aerial.npy")),
        }


        self.datasets = self._prepare_datasets()

    def _load_samples(self):

        """
        return {
            'guide': torch.tensor(bath).to(torch.float32).unsqueeze(0).unsqueeze(0),
            'source': torch.tensor(img).to(torch.float32).unsqueeze(0),
            'y': y,
            'mask_lr': 0,
            'y_bicubic': torch.nn.functional.interpolate(torch.tensor(img).to(torch.float32).unsqueeze(0), size=(512, 512), mode='bicubic', align_corners=True),
            'mask_hr': mask_hr
        }
        :return:
        """
        samples = {"source": [], "y": [], "guide": []}
        for location in self.locations:
            if self.bathymetry:
                img_dir_bath= os.path.join(
                    self.root_dir, location, "depth", "aerial")
                img_dir_s2 = os.path.join(self.root_dir, location, "img", "s2")
                img_dir_spot6 = os.path.join(
                    self.root_dir, location, "img", "aerial")
            else:
                img_dir_s2 = os.path.join(self.root_dir, location, "img", "s2")
                img_dir_spot6 = os.path.join(
                    self.root_dir, location, "img", "aerial")
            samples["source"].extend(sorted([os.path.join(
                img_dir_s2, f) for f in os.listdir(img_dir_s2) if f.endswith(".tif")]))
            samples["y"].extend(sorted([os.path.join(
                img_dir_spot6, f) for f in os.listdir(img_dir_spot6) if f.endswith(".tif")]))
            samples["guide"].extend(sorted([os.path.join(img_dir_bath, f) for f in os.listdir(img_dir_bath) if f.endswith(".tif")]))
        return samples


    def _prepare_datasets(self):
        samples = self._load_samples()

        agia_napa_test_ids = {'411', '387', '410', '398', '370', '369', '397'}
        puck_lagoon_test_ids = {'2987', '2707', '840', '293', '1510', '2667', '1720', '877', '1878', '3156', '358', '1829', '1121', '1044',
                                '1274', '1107', '2814', '760', '2938', '253', '2647', '1760', '1848', '2065', '1983', '1176', '1408',
                                '1618', '2632', '1264', '365', '876', '3267', '1985', '869', '2866', '676', '273', '185', '2120', '2846',
                                '460', '2125', '3196', '345', '535', '1718', '2041', '2160', '1401', '2876', '952', '2577', '765', '3069',
                                '2752', '1431', '2325', '496', '231', '2558', '12', '2951', '2106', '236', '2008', '837', '1182', '61',
                                '475', '1287', '1662', '168', '1970', '341', '2480', '2678', '3194', '783', '2575', '868', '1446', '874',
                                '1318', '1746', '1144', '2414', '2757', '2875', '3296', '1091', '281', '1596', '2042', '2163', '1427', '96',
                                '543', '2723', '2610', '2728', '1045', '1313', '2885', '142', '1503', '867', '2677', '1252', '280', '2107',
                                '1417', '1664', '3040', '1032', '2197', '780', '1193', '1148', '1635', '331', '999', '200', '147', '1137',
                                '2362', '2488', '885', '318', '3176', '2935', '325', '1950', '2270', '2561', '1346', '1798', '59', '544',
                                '1083', '2252', '3186', '248', '3151', '945', '2050', '552', '3142', '525', '1301', '2214', '3085', '274',
                                '2347', '993', '213', '831', '2910', '2983', '416', '245', '302', '1066', '1677', '2758', '3088', '3160',
                                '851', '2738', '1466', '1183', '2019', '2966', '1213', '3155', '1851', '1321', '583', '2335', '2349', '107',
                                '204', '2171', '2971', '183', '3080', '1307', '793', '1722', '1934', '2500', '234', '2508', '2547', '1734',
                                '933', '38', '1989', '2308', '2751', '380', '2989', '1868', '2552', '2211', '1564', '2165', '1875', '2026',
                                '2427', '564', '899', '673', '3171', '2717', '617', '1640', '1203', '1418', '3038', '1660', '1351', '1591',
                                '858', '2084', '1471', '3232', '3238', '1674', '1748', '298', '86', '2518', '610', '3126', '2596', '2699',
                                '2605', '433', '1324', '1608', '2196', '378', '2562', '1527', '2732', '516', '3089', '659', '3165', '907',
                                '826', '1940', '1781', '958', '2493', '2340', '685', '2712', '1319', '1244', '3234', '3269', '3002', '2379',
                                '169', '2952', '1068', '2968', '2249', '3220', '3231', '3071', '967', '2284', '492', '264', '929', '471',
                                '964', '2774', '518', '1716', '313', '276', '2034', '1663', '98', '2590', '2705', '1332', '2934', '1667',
                                '2530', '2726', '212', '1204', '357', '111', '584', '474', '1732', '984', '767', '1539', '681', '2565',
                                '1671', '560', '1842', '932', '1613', '2403', '1173', '575', '2481', '297', '1412', '1447', '2974', '708',
                                '3003', '812', '225', '2775', '2332', '1209', '1410', '2331', '1972', '3256', '292', '508', '875', '3286',
                                '1867', '694', '2172', '2254', '2062', '256', '3214', '312', '1111', '2680', '156', '2893', '1592', '3282',
                                '579', '330', '1653', '2486', '1576', '241', '1764', '569', '1461', '2780', '1343', '2888', '2943', '152',
                                '3309', '864', '781', '2571', '2770', '304', '1688', '546', '1011', '1077', '2279', '1978', '1326', '988',
                                '735', '1816', '2727', '488', '2515', '926', '1526', '965', '1905', '1818', '1040', '1491', '1536', '482',
                                '626', '2992', '4', '3198', '2471', '2215', '1065', '3127', '3091', '1757', '383', '2117', '2539', '184',
                                '766', '1714', '909', '526', '320', '1385', '1406', '443', '60', '2624', '1706', '3235', '1407', '2014',
                                '2957', '2475', '894', '116', '432', '1747', '2132', '259', '2567', '1015', '2978', '1443', '3041', '2950',
                                '3219', '714', '2573', '3135', '3266', '1262', '3287', '1279', '1881', '821', '1726', '1376', '370', '2320',
                                '419', '884', '1799', '2118', '319', '1721', '186', '2255', '740', '1733', '1540', '2669', '2710', '941',
                                '219', '2426', '1009', '789', '2445', '2697', '1743', '2724', '88', '596', '2750', '1248', '1017', '1846',
                                '2081', '2097', '1769', '1832', '368', '509', '1285', '1549', '1665', '1115', '737', '1877', '2212', '2829',
                                '604', '2381', '797', '3103', '481', '521', '2057', '2538', '1809', '1986', '3132', '582', '1342', '938',
                                '2450', '3116', '836', '2067', '2673', '1023', '2193', '1254', '3100', '1652', '2315', '155', '2430',
                                '3130', '1741', '968', '996', '2282', '491', '3166', '2994', '2570', '898', '923', '823', '2868', '3122',
                                '1489', '1310', '1373', '2400', '3150', '201', '2022', '2046', '3049', '1146', '2348', '470', '774', '2122',
                                '3207', '2612', '2783', '2192', '600', '950', '1690', '857', '101', '3187', '381', '435', '162', '1368',
                                '2058', '1046', '251', '249', '1075', '2642', '1292', '1167', '1975', '2778', '1079', '634', '1215', '1356'}

        """
        for s2_path, aerial_path, depth_path in zip(samples["s2"], samples["aerial"], samples["depth"]):
            img_name = os.path.basename(aerial_path)
            img_id = img_name.split('_')[1].split('.')[0]

            if "agia_napa" in aerial_path and img_id in agia_napa_test_ids:
                test_aerial.append(aerial_path)
                test_s2.append(s2_path)
                test_depth.append(depth_path)
            elif "puck_lagoon" in aerial_path and img_id in puck_lagoon_test_ids:
                test_aerial.append(aerial_path)
                test_s2.append(s2_path)
                test_depth.append(depth_path)
            else:
                train_aerial.append(aerial_path)
                train_s2.append(s2_path)
                train_depth.append(depth_path)
        """
        # Create dictionaries to index files by their identifiers
        aerial_dict = {os.path.basename(path).split('_')[1].split('.')[0]: path for path in samples["y"]}
        s2_dict = {os.path.basename(path).split('_')[1].split('.')[0]: path for path in samples["source"]}
        depth_dict = {os.path.basename(path).split('_')[2].split('.')[0]: path for path in samples["guide"]}  # TODO: .split('_')[2] if agia_napa

        # Find common identifiers
        common_ids = set(aerial_dict.keys()) & set(s2_dict.keys()) & set(depth_dict.keys())

        # Initialize train/test splits
        train_aerial, train_s2, train_depth = [], [], []
        test_aerial, test_s2, test_depth = [], [], []

        for img_id in common_ids:
            aerial_path = aerial_dict[img_id]
            s2_path = s2_dict[img_id]
            depth_path = depth_dict[img_id]

            # Split based on dataset and test IDs
            if "agia_napa" in aerial_path and img_id in agia_napa_test_ids:
                test_aerial.append(aerial_path)
                test_s2.append(s2_path)
                test_depth.append(depth_path)
            elif "puck_lagoon" in aerial_path and img_id in puck_lagoon_test_ids:
                test_aerial.append(aerial_path)
                test_s2.append(s2_path)
                test_depth.append(depth_path)
            else:
                train_aerial.append(aerial_path)
                train_s2.append(s2_path)
                train_depth.append(depth_path)

        test_aerial = sorted(test_aerial)
        test_s2 = sorted(test_s2)
        test_depth = sorted(test_depth)
        train_aerial = sorted(train_aerial)
        train_s2 = sorted(train_s2)
        train_depth = sorted(train_depth)

        images_s2_train, images_s2_val, images_aerial_train, images_aerial_val, depth_train, depth_val = train_test_split(train_s2, train_aerial, train_depth, test_size=self.val_size, random_state=65)

        datasets = {
            "train": MagicBathyNet(images_s2_train, images_aerial_train, depth_train, self.bathymetry, self.transform, self.target_trans, self.norm_params),
            "val": MagicBathyNet(images_s2_val, images_aerial_val, depth_val, self.bathymetry, self.transform, self.target_trans, self.norm_params),
            "test": MagicBathyNet(test_s2, test_aerial, test_depth, self.bathymetry, self.transform, self.target_trans, self.norm_params)
        }
        return datasets


    def get_dataloader(self, set_type="train"):
        return DataLoader(
            self.datasets[set_type],
            batch_size=self.batch_size,
            shuffle=True if set_type == "train" else False,
            num_workers=self.num_workers
        )
