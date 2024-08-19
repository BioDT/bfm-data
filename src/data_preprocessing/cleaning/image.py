# src/data_preprocessing/cleaning/image.py

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def resize_image(image: Image.Image, size: tuple) -> Image.Image:
    """
    Resizes an image to the specified dimensions.

    Args:
        image (Image.Image): The input image.
        size (tuple): The desired dimensions (width, height).

    Returns:
        Image.Image: The resized image.
    """
    return image.resize(size)


def normalize_image(
    image: torch.Tensor, mean: list = [0.5, 0.5, 0.5], std: list = [0.5, 0.5, 0.5]
) -> torch.Tensor:
    """
    Normalizes an image tensor to have a standard range of values.

    Args:
    image (torch.Tensor): The input image tensor.
    mean (list): The mean values for each channel.
    std (list): The standard deviation values for each channel.

    Returns:
    torch.Tensor: The normalized image tensor.
    """
    normalize = transforms.Normalize(mean=mean, std=std)
    return normalize(image)


def denoise_image(image: Image.Image) -> Image.Image:
    """
    Removes noise from an image using a denoising algorithm.

    Args:
        image (Image.Image): The input image.

    Returns:
        Image.Image: The denoised image.
    """
    image_np = np.array(image)
    denoised_image_np = cv2.fastNlMeansDenoisingColored(image_np, None, 10, 10, 7, 21)
    denoised_image = Image.fromarray(denoised_image_np)
    return denoised_image
