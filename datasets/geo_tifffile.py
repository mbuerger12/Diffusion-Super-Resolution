from osgeo import gdal
import numpy as np

"""
2 functions to read and write geotiff files. Copied from the original code of the MagicBathyNet notebook.

Reference:
Agrafiotis, P., Janowski, L., Skarlatos, D. & Demir, B. (2024) MagicBathyNet: A Multimodal Remote Sensing Dataset for Bathymetry Prediction and Pixel-based Classification in Shallow Waters, arXiv preprint arXiv:2405.15477, 2024.
@article{agrafiotis2024magicbathynet,
      title={MagicBathyNet: A Multimodal Remote Sensing Dataset for Bathymetry Prediction and Pixel-based Classification in Shallow Waters}, 
      author={Panagiotis Agrafiotis and Łukasz Janowski and Dimitrios Skarlatos and Begüm Demir},
      year={2024},
      journal={arXiv preprint arXiv:2405.15477}
}

Original code from GitHub repository: https://github.com/pagraf/MagicBathyNet.git
"""

def read_geotiff3D(filename, bathymetry=False):
    """
    Read a geotiff file and return the image as a numpy array

    Args:
    filename: str, path to the geotiff file
    bathymetry: bool, if True, read only the first band
    """

    ds = gdal.Open(filename)

    # Dimensions
    # print("X-Size: ", ds.RasterXSize)
    # print("Y-Size: ", ds.RasterYSize)

    # Number of bands
    # print("Bands: ", ds.RasterCount)

    # Metadata for the raster dataset
    # print("Meta Data: ", ds.GetMetadata())

    
    band1 = ds.GetRasterBand(1)
    if not bathymetry:
        band2 = ds.GetRasterBand(2)
        band3 = ds.GetRasterBand(3)
        bands = [band1.ReadAsArray(), band2.ReadAsArray(), band3.ReadAsArray()]
    else:
        bands = [band1.ReadAsArray()]
    img = np.stack(bands, axis=0)
    img = np.transpose(img, (1, 2, 0)).astype(np.float32)
    
    return img, ds


def write_geotiff3D(filename, img, in_ds, bathymetry=False):
    """
    Write a numpy array to a geotiff file

    Args:
    filename: str, path to the geotiff file
    img: numpy array, image to write
    in_ds: metadata from the original geotiff file
    bathymetry: bool, if True, write only the first band
    """

    if img.dtype == np.float32:
        arr_type = gdal.GDT_Float32
    else:
        arr_type = gdal.GDT_Int32
    driver = gdal.GetDriverByName("GTiff")

    if bathymetry:
        out_ds = driver.Create(filename, img.shape[1], img.shape[0], 1, arr_type)
    else:
        img = np.transpose(img, (2, 0, 1)) # H x W x C -> C x H x W
        out_ds = driver.Create(filename, img[0].shape[1], img[0].shape[0], 3, arr_type)
    
    out_ds.SetProjection(in_ds.GetProjection())
    out_ds.SetGeoTransform(in_ds.GetGeoTransform())

    if not bathymetry:
        band1 = out_ds.GetRasterBand(1)
        band2 = out_ds.GetRasterBand(2)
        band3 = out_ds.GetRasterBand(3)
        band1.WriteArray(img[0])
        band1.FlushCache()
        band1.ComputeStatistics(False)
        band2.WriteArray(img[1])
        band2.FlushCache()
        band2.ComputeStatistics(False)
        band3.WriteArray(img[2])
        band3.FlushCache()
        band3.ComputeStatistics(False)
    else:
        band1 = out_ds.GetRasterBand(1)
        band1.WriteArray(img)
        band1.FlushCache()
        band1.ComputeStatistics(False)
