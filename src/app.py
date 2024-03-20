import numpy as np
import streamlit as st

from PIL import Image
from metrics import calculate_metrics
from models import get_available_model_names, run_inference


st.set_page_config(layout="wide")
st.title("Generated Image Comparison")

model_col1, model_col2 = st.columns(2)
with model_col1:
    available_models = get_available_model_names()
    selected_model = st.selectbox("Select Model to Run Inference", available_models)

with model_col2:
    num_images = st.selectbox("Number of Images to Generate", [10, 20, 30, 50])

upload_col1, upload_col2 = st.columns(2)  
with upload_col1:
    st.header("Real Images")
    real_images = st.file_uploader("Upload Real Images", accept_multiple_files=True, disabled=not selected_model)
    
    if real_images:
        real_image_list = [np.array(Image.open(img).resize((1024, 1024), Image.Resampling.BILINEAR), dtype=np.uint8) for img in real_images]
        st.image(real_image_list, width=150, caption=["Real" for _ in real_image_list])

with upload_col2:
    st.header("Generated Images")
    if st.button("Generate Images", type="primary", disabled=not selected_model):
        with st.spinner(f"Running inference using {selected_model} to generate {num_images} images..."):
            generated_images = run_inference("test_model.safetensors", num_images)

        generated_image_list = [np.array(Image.open(img).resize((1024, 1024), Image.Resampling.BILINEAR), dtype=np.uint8) for img in generated_images]  
        with st.spinner("Calculating Metrics..."):
            metrics_1 = calculate_metrics(real_image_list, generated_image_list)
        st.image(generated_image_list, width=150, caption=["Generated" for _ in generated_image_list])
        st.metric("FID", round(metrics_1['FID'], 4))
        st.metric("MIFID", round(metrics_1['MIFID'], 4))
        st.metric("IS Mean", round(metrics_1['IS'][0], 4)) 
        st.metric("IS Std Dev", round(metrics_1['IS'][1], 4)) 
        st.metric("SSIM", round(metrics_1['SSIM'], 4))
        st.metric("MS SSIM", round(metrics_1['MS_SSIM'], 4))

st.header("Metrics")

if real_images and generated_images:
    img1, img2 = st.columns(2)
    for real_img, gen_img in zip(real_image_list, generated_image_list):
        img1.image(real_img, caption='Real', use_column_width=True)  
        img2.image(gen_img, caption='Generated', use_column_width=True)