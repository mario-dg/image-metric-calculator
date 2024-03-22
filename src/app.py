import numpy as np
import streamlit as st

from PIL import Image
from metrics import calculate_metrics
from models import get_available_model_names, run_inference
from data import load_dataset_from_hub, get_train_data, PROMPTS


st.set_page_config(layout="wide")
st.title("Generated Image Comparison")

with st.spinner("Downloading dataset (mario-dg/dreambooth-cell-images)..."):
    load_dataset_from_hub()

model_col1, model_col2, model_col3 = st.columns([2, 2, 3])
with model_col1:
    available_models = get_available_model_names()
    selected_model = st.selectbox("Select Model to Run Inference", available_models)

with model_col2:
    num_images = st.selectbox("Number of Images to Generate", [10, 20, 30, 50])

with model_col3:
    selected_prompt = st.selectbox("Prompt", PROMPTS)

upload_col1, upload_col2 = st.columns(2)  
with upload_col1:
    st.header("Real Images")
    if not selected_model:
        with st.spinner("Loading Real Images..."):
            real_images, num_images = get_train_data(selected_prompt, num_images)
            real_image_list = [np.array(img.resize((1024, 1024), Image.Resampling.BILINEAR), dtype=np.uint8) for img in real_images]
            st.image(real_image_list, width=150)

with upload_col2:
    st.header("Generated Images")
    if st.button("Generate Images", type="primary"):
        
        progress_bar = st.progress(0, text="Generating Images...")
        generated_images = run_inference(selected_model, num_images, selected_prompt, progress_bar)
        progress_bar.empty()

        generated_image_list = [np.array(Image.open(img).resize((1024, 1024), Image.Resampling.BILINEAR), dtype=np.uint8) for img in generated_images]  
        with st.spinner("Calculating Metrics..."):
            metrics_1 = calculate_metrics(real_image_list, generated_image_list)
        st.image(generated_image_list, width=150)
        st.metric("FID", round(metrics_1['FID'], 4))
        st.metric("IS Mean", round(metrics_1['IS'][0], 4)) 
        st.metric("IS Std Dev", round(metrics_1['IS'][1], 4)) 
        st.metric("SSIM", round(metrics_1['SSIM'], 4))
