from datasets import load_dataset
from random import shuffle
from PIL import Image


dataset = None
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
    global dataset
    if not dataset:
        dataset = load_dataset("mario-dg/dreambooth-cell-images", split="train")


def _get_images_by_class_name(class_name: str, num_images: int) -> tuple[list[Image], int]:
    class_images = [img["image"] for img in dataset if img["class_name"] == class_name]
    shuffle(class_images)

    if class_name == "real_debris":
        num_images = 10

    return class_images[:num_images], num_images

def get_train_data(prompt: str, num_images: int) -> tuple[list[Image], int]:
    class_name = _get_class_name_from_prompt(prompt)
    print(f"Retrieving images for class: {class_name}")
    return _get_images_by_class_name(class_name, num_images)
