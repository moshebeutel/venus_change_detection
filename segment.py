import os
from osgeo import gdal
import leafmap
from samgeo import SamGeo, show_image, download_file, overlay_images, tms_to_geotiff
import torch
import segmentation_models_pytorch as smp
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window
from rasterio import Affine as A
from rasterio.warp import calculate_default_transform, reproject, Resampling

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = 'cpu'
class_dict = pd.read_csv(f"{Path.home()}/datasets/massachusetts-buildings-dataset/label_class_dict.csv")
class_rgb_values = class_dict[['r', 'g', 'b']].values.tolist()
class_names = class_dict['name'].tolist()
select_classes = ['background', 'building']
select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
select_class_rgb_values = np.array(class_rgb_values)[select_class_indices]
ENCODER = 'resnet152'
# ENCODER = 'resnet101'
# ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = class_names
ACTIVATION = 'sigmoid'
metula_lat_lon = [33.277232, 35.578235]
image = "satellite.tif"
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
model = torch.load('/home/user1/saved_models/unet_resnet152_on_arerial/aerial_blur_1_5.pth', map_location=DEVICE)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def colour_code_segmentation(im, label_values):
    colour_codes = np.array(label_values)
    x = colour_codes[im.astype(int)]
    return x


def reverse_one_hot(im):
    x = np.argmax(im, axis=-1)
    return x


def read_satellite_image():
    col_ofset = 1200
    width = 1824
    row_ofset = 2000
    height = 3200
    window = Window(col_ofset, row_ofset, width, height)
    window_transform = None
    window_crs = None
    with rasterio.open('resources/satellite.tif') as src:
        print(src.width, src.height)
        img = np.concatenate([src.read(i, window=window)[:, :, np.newaxis] for i in range(1, src.count + 1)], axis=-1)
        # print(src.bounds, src.transform, src.crs, img.shape, src.count)
        src_transform = src.transform
        window_transform = rasterio.windows.transform(window, src.transform)
        window_crs = src.crs
    return img, window_transform, window_crs


def infer_buildings_in_image(img: np.ndarray):
    img = img.astype(float)
    img = preprocessing_fn(img)
    x_tensor = torch.from_numpy(to_tensor(img)).to(DEVICE).unsqueeze(0)
    with torch.no_grad():
        pred_mask = model(x_tensor)
        # pred_mask1 = model1(x_tensor)
    pred_mask = pred_mask.detach().squeeze().cpu().numpy()
    pred_mask = np.transpose(pred_mask, (1, 2, 0))
    pred_building_heatmap = pred_mask[:, :, select_classes.index('building')]
    pred_background_heatmap = pred_mask[:, :, select_classes.index('background')]
    pred_mask = colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values)
    pred_mask = pred_mask.astype(np.uint8)
    pred_building_heatmap *= 255.0
    pred_building_heatmap = pred_building_heatmap.astype(np.uint8)
    return pred_building_heatmap, pred_mask


def create_mask_geotiff(pred_mask: np.ndarray, pred_building_heatmap, window_transform, window_crs):
    with rasterio.open(
            'pred_mask.tif',

            'w',

            driver='GTiff',

            height=pred_mask.shape[0],

            width=pred_mask.shape[1],

            count=pred_mask.shape[2] + 1,

            dtype=pred_mask.dtype,

            crs=window_crs,

            transform=window_transform
    ) as new_dataset:
        for band in range(pred_mask.shape[2]):
            new_dataset.write(pred_mask[:, :, band], band + 1)
        new_dataset.write(pred_building_heatmap, pred_mask.shape[2] + 1)
