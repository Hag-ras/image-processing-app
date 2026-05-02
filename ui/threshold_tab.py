import streamlit as st
import numpy as np
from thresholding import OptimalThreshold, OtsuThreshold, SpectralThreshold, LocalThreshold
from ui.helpers import show_results


def render(gray: np.ndarray):
    st.subheader("Thresholding Methods")

    n_classes = st.slider("Spectral classes", 3, 6, 3)
    block_size = st.slider("Local threshold block size (odd)", 11, 101, 51, step=2)

    methods = [
        OptimalThreshold(),
        OtsuThreshold(),
        SpectralThreshold(n_classes),
        LocalThreshold(block_size),
    ]

    if st.button("Run Thresholding"):
        results = [m.apply(gray) for m in methods]
        show_results(results, cols=2)
