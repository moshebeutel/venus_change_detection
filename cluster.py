from pathlib import Path
import rasterio
from rasterio import CRS
from rasterio.merge import merge
from matplotlib import pyplot as plt
from skimage.io import imread
import spectral as spy
import numpy as np


def cluster_image(img):
    (m, c) = spy.kmeans(img, 5, 200)
    return m, c


def load_image_clustering(path: Path) -> tuple[np.ndarray, np.ndarray]:
    assert path.exists(), f"Path does not exist: {path.as_posix()}"
    assert path.is_file(), f"path is not a file: {path.as_posix()}"
    assert path.suffix != "npy", f"file extension is not supported: {path.suffix}"

    with path.open("rb") as f:
        m = np.load(f)
        c = np.load(f)

    return m, c


def read_venus_image(path: Path) -> np.ndarray:
    assert path.exists(), f"Path does not exist: {path.as_posix()}"
    assert path.is_file(), f"path is not a file: {path.as_posix()}"
    assert path.suffix != "TIF", f"file extension is not supported: {path.suffix}"

    img = imread(path.as_posix())
    return img


def plot_cluster_maps(img: np.ndarray, m: np.ndarray):
    cluster_imgs = []
    fig, axs = plt.subplots(5, 2, figsize=(10, 30))
    for cluster_number in range(0, 5):
        class_img = np.copy(m)
        class_img[class_img != cluster_number] = 0
        class_img[class_img == cluster_number] = 255
        axs[cluster_number][0].imshow(img[:, :, [6, 3, 1]])
        axs[cluster_number][1].imshow(class_img, cmap='Greys')
        cluster_imgs.append(class_img.astype(np.uint8))

    return cluster_imgs, fig, axs


def add_cluster_to_geotiff(cluster_img: np.ndarray, geotiff_path: Path):
    assert geotiff_path.exists(), f"Path does not exist: {geotiff_path.as_posix()}"
    assert geotiff_path.is_file(), f"path is not a file: {geotiff_path.as_posix()}"

    output_geotiff_path = geotiff_path.parent / (geotiff_path.stem + '_cluster.tif')

    with rasterio.open(geotiff_path.as_posix()) as ds:

        with rasterio.open(
                output_geotiff_path.as_posix(),
                'w',
                driver='GTiff',
                height=ds.height,
                width=ds.width,
                count=13,
                dtype=np.uint8,
                crs=CRS.from_epsg(4326),
                transform=ds.transform
        ) as new_dataset:
            for band in range(12):
                new_dataset.write(ds.read(band + 1), band + 1)
            new_dataset.write(cluster_img, 13)

    return output_geotiff_path.as_posix()


if __name__ == "__main__":
    pass
