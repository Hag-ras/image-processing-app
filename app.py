import streamlit as st
from ui.helpers import load_image, to_gray
from ui import threshold_tab, segmentation_tab


def main():
    st.set_page_config(page_title="Image Processing", layout="wide")
    st.title("Image Thresholding & Segmentation")

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
    if not uploaded:
        st.info("Upload an image to begin.")
        return

    image = load_image(uploaded)
    gray = to_gray(image)

    st.image(image, caption="Original", width=300)
    st.divider()

    tab_thresh, tab_seg = st.tabs(["Thresholding (Grayscale)", "Segmentation (Color)"])

    with tab_thresh:
        threshold_tab.render(gray)

    with tab_seg:
        segmentation_tab.render(image)


if __name__ == "__main__":
    main()
