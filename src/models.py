import os
import time

import streamlit as st


MODEL_PATH = "models/"

def get_available_model_names() -> list[str]:
    files = os.listdir(MODEL_PATH)
    return [f for f in files if f.endswith(".safetensors")]

def run_inference(model_file: str, num_images: int, prompt: str, progress_bar: st.progress) -> list[str]:
    for i in range(num_images):
        progress_bar.progress((i + 1) / num_images, text=f"Generating image {i + 1}/{num_images}...")
        time.sleep(1)
    return [""]