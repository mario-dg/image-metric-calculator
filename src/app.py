from metrics import calculate_fid, calculate_is, calculate_ssim, calculate_ms_ssim, calculate_mifid

import streamlit as st
import numpy as np
from PIL import Image

def calculate_metrics(real_images, generated_images):
    # Calculate FID, IS, SSIM here using the code from your original script
    # Return a dict with the metrics
    return {'FID': calculate_fid(real_images, generated_images),
            'MIFID': calculate_mifid(real_images, generated_images),
            'IS': calculate_is(generated_images), 
            'SSIM': calculate_ssim(real_images, generated_images),
            'MS_SSIM': calculate_ms_ssim(real_images, generated_images)}

st.set_page_config(layout="wide")
st.title("Generated Image Comparison")

col1, col2 = st.columns(2)  

with col1:
    st.header("Real Images")
    real_images = st.file_uploader("Upload Real Images", accept_multiple_files=True)
    
    if real_images:
        real_image_list = [np.array(Image.open(img).resize((1024, 1024), Image.Resampling.BILINEAR), dtype=np.uint8) for img in real_images]
        st.image(real_image_list, width=150, caption=["Real" for _ in real_image_list])

with col2:
    st.header("Generated Images")
    generated_images = st.file_uploader("Upload Generated Images", accept_multiple_files=True) 

    if generated_images:
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