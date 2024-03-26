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
    if selected_model:
        with st.spinner("Loading Real Images..."):
            real_image_list, num_images = get_train_data(selected_prompt, num_images)
            st.image(real_image_list, width=256)

with upload_col2:
    st.header("Generated Images")
    if st.button("Generate Images", type="primary"):
        generated_images = run_inference(selected_model, num_images, selected_prompt, None)
        generated_image_list = [np.array(Image.open(img), dtype=np.uint8) for img in generated_images]  

        with st.spinner("Calculating Metrics..."):
            metrics = calculate_metrics(real_image_list, generated_image_list)
            print(metrics)

        st.metric("FID", round(metrics['FID'], 4))
        st.metric("SSIM", round(metrics['SSIM'], 4))
        st.image(generated_image_list, width=256)
