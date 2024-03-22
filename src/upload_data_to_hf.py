from datasets import load_dataset


dataset = load_dataset("imagefolder", data_dir="data/cell images", split="train")
dataset.push_to_hub("mario-dg/dreambooth-cell-images", commit_message="Add class_names to metadata")
