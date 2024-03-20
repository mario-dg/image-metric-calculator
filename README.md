# Image Metric Calculator
## Overview
This little Streamlit app serves as a helper for my most recent paper "Enhanced Generation of Synthetic Brightfield Microscopy Images Using Dreambooth for Stable Diffusion", where I try to generate realistic looking brightfield microscopy images, while addressing the shortcomings of my previous paper.
For this I need to calculate several metrics for each concept (4) and model (24) trained.

## Installation
The first step, as always, clone this repo and move into it.
```cmd
git clone https://github.com/mario-dg/image-metric-calculator.git
cd image-metric-calculator
```

Make sure that poetry is installed on your local machine, to conveniently install all dependencies.
```cmd
poetry install
```

Otherwise check the `pyproject.toml` file for all dependencies needed and install them using any environment and dependeny manager of your choice, like `conda`.

Start the Streamlit app.
```cmd
poetry run streamlit run src/app.py
```

## Usage
All the models were trained on either 10, 20, 30 or 50 images of each concept, hence only these amounts can and should be generated for the metric calculations. Some metrics require the same amount of `true` and `fake` images.
For this paper 3 different Stable Diffusion architectures were trained:
- Stable Diffusion 1.5
- Stable Diffusion 2.1
- Stable Diffusion XL 1.0
So make sure that these are available in the `models/` directory and are saved as `.safetensors`.

The specific prompts and unique identifiers for each concept that these models were trained on are hardcoded and can/should not be changed, otherwise the results are unpredictable.
The dataset is available on huggingface and will be downloaded on startup of the application.

450 subject class images were generated for each model architecture and concept. These are not necessary for the metric calculations, but are part of the training data that I published on huggingface.