import streamlit as st
import numpy as np
from segmentation import KMeansSegmentation, RegionGrowingSegmentation, AgglomerativeSegmentation, MeanShiftSegmentation
from ui.helpers import show_results


def render(image: np.ndarray):
    st.subheader("Segmentation Methods")

    k     = st.slider("K-Means clusters", 2, 8, 4)
    n_agg = st.slider("Agglomerative clusters", 2, 10, 5)
    sp    = st.slider("Mean Shift spatial window", 5, 50, 20)
    sr    = st.slider("Mean Shift color window", 10, 100, 40)

    methods = [
        KMeansSegmentation(k),
        RegionGrowingSegmentation(),
        AgglomerativeSegmentation(n_agg),
        MeanShiftSegmentation(sp, sr),
    ]

    if st.button("Run Segmentation"):
        with st.spinner("Processing…"):
            results = [m.apply(image) for m in methods]
        show_results(results, cols=2)
