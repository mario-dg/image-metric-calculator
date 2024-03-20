import os
import time


MODEL_PATH = "src/models/"

def get_available_model_names() -> list[str]:
    files = os.listdir(MODEL_PATH)
    return [f for f in files if f.endswith(".safetensors")]

def run_inference(model_file: str, num_images: int) -> list[str]:
    print(f"Running inference using {model_file} to generate {num_images} images...")
    time.sleep(5)
    return [""]