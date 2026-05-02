import streamlit as st
import numpy as np
import cv2
from PIL import Image
from typing import Optional
from base import ProcessedResult


def load_image(uploaded) -> Optional[np.ndarray]:
    return np.array(Image.open(uploaded).convert("RGB"))


def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def show_results(results: list[ProcessedResult], cols: int = 2):
    for i in range(0, len(results), cols):
        columns = st.columns(cols)
        for j, col in enumerate(columns):
            if i + j < len(results):
                r = results[i + j]
                col.image(r.image, caption=r.title, use_container_width=True, clamp=True)
