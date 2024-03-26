import os
import torch
import streamlit as st

from glob import glob
from pathlib import Path
from diffusers import StableDiffusionPipeline

from utils import timeit


MODEL_PATH = ".\\models"
SAVE_PATH = ".\\data"

def get_available_model_names() -> list[str]:
    files = glob(f"{MODEL_PATH}\\*.safetensors")
    return [Path(f).name for f in files]

@timeit
def run_inference(model_file: str, num_images: int, prompt: str, progress_bar: st.progress) -> list[str]:
    progress_bar = st.progress(0, text="Generating Images...")
    model_file_short = f"{model_file.rsplit('_', 1)[0]}"
    model_data_dir = Path(f"{SAVE_PATH}\\{model_file_short}")
    model_data_dir.mkdir(parents=True, exist_ok=True)
    pipe = StableDiffusionPipeline.from_single_file(f"{MODEL_PATH}\\{model_file}", torch_dtype=torch.float16, use_safetensors=True)
    pipe = pipe.to("cuda")

    for i in range(num_images):
        image = pipe(prompt=prompt, width=512, height=512, num_inference_steps=40, guidance_scale=11.0).images[0]
        img_dir = f"{model_data_dir}\\{prompt}_{i:02d}.png"
        image.save(img_dir)
        progress_bar.progress((i + 1) / num_images, text=f"Generated image {i + 1}/{num_images} -> {img_dir}")
    
    progress_bar.empty()
    return [os.path.join(model_data_dir, f) for f in os.listdir(model_data_dir) if prompt in f]