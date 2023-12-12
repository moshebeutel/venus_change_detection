import streamlit as st
import leafmap.foliumap as leafmap
# Set up the Streamlit app title
st.title("VENuS")

import streamlit as st

tab1, tab2, tab3 = st.tabs(["Clustering", "Semantic Segmentation", "Change Detection"])
default_location = [32.06252332017934, 34.85233940396549]
with tab1:
    # Main content
    st.header("Cluster All Bands")
    # default_location = [37.7749, -122.4194]
    user_location = default_location
    # Create a LeafMap
    m = leafmap.Map(center=user_location, zoom=12)

    # Add a marker at the user's location
    m.add_marker(location=user_location)

    # Render the map in Streamlit
    st.components.v1.html(m.to_html(), height=600, width=800, scrolling=True)

with tab2:
    st.header("Semantic Segmentation")
    # default_location = [37.7749, -122.4194]
    user_location = default_location
    # Create a LeafMap
    m = leafmap.Map(center=user_location, zoom=12)

    # Add a marker at the user's location
    m.add_marker(location=user_location)

    # Render the map in Streamlit
    st.components.v1.html(m.to_html(), height=600, width=800, scrolling=True)

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