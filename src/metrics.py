import torch
import numpy as np

from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure


_ = torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_fid(real_images, generated_images):
    fid = FrechetInceptionDistance().to(device)
    
    for real, fake in zip(real_images, generated_images):
        imgs_dist1 = torch.tensor(real.transpose((2, 0, 1)), device=device).unsqueeze(dim=0)
        imgs_dist2 = torch.tensor(fake.transpose((2, 0, 1)), device=device).unsqueeze(dim=0)
        fid.update(imgs_dist1, real=True)
        fid.update(imgs_dist2, real=False)
    
    return fid.compute().item()


def calculate_is(generated_images):
    inception = InceptionScore().to(device)
    for gen_img in generated_images:
        img = torch.tensor(gen_img.transpose((2, 0, 1)), device=device, dtype=torch.uint8).unsqueeze(dim=0)
        inception.update(img)
    
    return [v.item() for v in inception.compute()]


def calculate_ssim(real_images, generated_images):
    ssim = StructuralSimilarityIndexMeasure().to(device)
    num_gen_images = len(generated_images)
    preds = torch.tensor(np.transpose(generated_images, (0, 3, 1, 2)), dtype=torch.float64, device=device)
    target = torch.tensor(np.transpose(real_images[:num_gen_images], (0, 3, 1, 2)), dtype=torch.float64, device=device)

    return ssim(preds, target).item()


def calculate_metrics(real_images, generated_images):
    return {'FID': calculate_fid(real_images, generated_images),
            'SSIM': calculate_ssim(real_images, generated_images),
    }

