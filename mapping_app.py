from pathlib import Path

import streamlit as st
import leafmap.foliumap as leafmap

from cluster import read_venus_image, cluster_image, load_image_clustering, plot_cluster_maps, add_cluster_to_geotiff
from config import VENUS_PATH, CLUSTERING_PRETRAINED_PATH, VENUS_SRE_PATH, VENUS_SRE_SAMPLE_PATH, VENUS_SAMPLE_GEOTIFF
from segment import read_satellite_image, infer_buildings_in_image, create_mask_geotiff

# Set up the Streamlit app title
st.title("VENuS")

import streamlit as st

tab1, tab2, tab3 = st.tabs(["Clustering", "Semantic Segmentation", "Change Detection"])
default_location = [32.06252332017934, 34.85233940396549]

# venus_images_gen = Path(VENUS_SRE_PATH).glob("*.*")
# image_path = next(iter(venus_images_gen))
image_path = Path(VENUS_SRE_SAMPLE_PATH)
assert image_path.exists()
venus_image = read_venus_image(image_path)
# m,c = cluster_image(venus_image)

venus_clustering_pretrained_gen = Path(CLUSTERING_PRETRAINED_PATH).glob("*.npy")
m, c = load_image_clustering(next(iter(venus_clustering_pretrained_gen)))
cluster_imgs, fig, axs = plot_cluster_maps(img=venus_image, m=m)
assert Path(VENUS_SAMPLE_GEOTIFF).exists()
geotiff_with_cluster = add_cluster_to_geotiff(cluster_img=cluster_imgs[-1], geotiff_path=Path(VENUS_SAMPLE_GEOTIFF))

img, window_transform, window_crs =read_satellite_image()
print(img.shape)
# plt.imshow(img)
pred_building_heatmap, pred_mask = infer_buildings_in_image(img)
# plt.imshow(pred_building_heatmap)
# plt.imshow(pred_mask)
create_mask_geotiff(pred_mask, pred_building_heatmap, window_transform, window_crs)
with tab1:
    # Main content
    st.header("Cluster All Bands")
    # default_location = [37.7749, -122.4194]
    user_location = default_location
    # Create a LeafMap

    map = leafmap.Map(center=user_location, zoom=12)
    map.add_raster(VENUS_SAMPLE_GEOTIFF, bands=[7,4,2], layer_name='venus_raster')
    map.add_raster(geotiff_with_cluster,bands=[13], layer_name='clustering')

    # st.plotly_chart(axs, use_container_width=True)
    st.pyplot(fig)

    # Render the map in Streamlit
    st.components.v1.html(map.to_html(), height=600, width=800, scrolling=True)

with tab2:
    st.header("Semantic Segmentation")
    # default_location = [37.7749, -122.4194]
    user_location = default_location
    # Create a LeafMap
    map1 = leafmap.Map(center=user_location, zoom=12)
    # m = leafmap.Map(center=metula_lat_lon, height="800px")
    map1.add_raster('resources/satellite.tif', layer_name="Image")
    map1.add_raster('pred_mask.tif', bands=[1, 2, 3], layer_name="pred mask")
    map1.add_raster('pred_mask.tif', bands=[4], layer_name="building heatmap")


    # Render the map in Streamlit
    st.components.v1.html(map1.to_html(), height=600, width=800, scrolling=True)

with tab3:
    st.header("Change Detection")
    # default_location = [37.7749, -122.4194]
    user_location = default_location
    # Create a LeafMap
    m = leafmap.Map(center=user_location, zoom=12)

    # Add a marker at the user's location
    m.add_marker(location=user_location)

    # Render the map in Streamlit
    st.components.v1.html(m.to_html(), height=600, width=800, scrolling=True)



# Sidebar for user input
st.sidebar.header("Map Settings")

# user_location = st.sidebar.map_input("Enter a location:", default_location, key="user_location")


# st.pydeck_chart(m, use_container_width=True)