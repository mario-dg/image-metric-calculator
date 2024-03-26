import numpy as np
import streamlit as st

from datasets import load_dataset
from PIL import Image


PROMPTS = [
    "a grayscale microscopy image of a ohwx cell",
    "a grayscale microscopy image of a adherent kker cell",
    "a grayscale microscopy image of a wiwx cell on the well edge",
    "a grayscale microscopy image of a kywx cell with debris",
]

PROMPT_TO_CLASS = {
    "a grayscale microscopy image of a ohwx cell": "cell",
    "a grayscale microscopy image of a adherent kker cell": "cell_rug",
    "a grayscale microscopy image of a wiwx cell on the well edge": "well_edge",
    "a grayscale microscopy image of a kywx cell with debris": "debris",

}

def _get_class_name_from_prompt(prompt: str) -> str:
    return f"real_{PROMPT_TO_CLASS[prompt]}"

def load_dataset_from_hub():
    if "dataset" not in st.session_state:
        temp = load_dataset("mario-dg/dreambooth-cell-images", split="train")

        st.session_state.dataset = [img for img in temp if img["class_name"].startswith("real_")]

def _get_images_by_class_name(class_name: str, num_images: int) -> tuple[list[Image.Image], int]:
    class_images = [np.array(img["image"], dtype=np.uint8)[:, :, :3] for img in st.session_state.dataset if img["class_name"] == class_name]
    if class_name == "real_debris":
        num_images = 10
    return class_images[:num_images], num_images

def get_train_data(prompt: str, num_images: int) -> tuple[list[Image.Image], int]:
    class_name = _get_class_name_from_prompt(prompt)
    return _get_images_by_class_name(class_name, num_images)
