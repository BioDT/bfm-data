"""Copyright (C) 2025 TNO, The Netherlands. Licensed under the MIT license."""

import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image


def resize_crop_image(
    image: Image.Image, size: tuple, crop: bool = False
) -> Image.Image:
    """
    Resizes or crops an image to a specified size.

    Args:
        image (Image.Image): The input image.
        size (tuple): The target size (width, height).
        crop (bool): If True, crops the image; otherwise, resizes.

    Returns:
        Image.Image: The resized or cropped image.
    """
    if crop:
        transform = transforms.CenterCrop(size)
    else:
        image = image.resize(size, Image.Resampling.LANCZOS)
        return image

    return transform(image)


def denoise_image(image: Image.Image) -> Image.Image:
    """
    Reduces noise from an image using a denoising algorithm.

    Args:
        image (Image.Image): The input image.

    Returns:
        Image.Image: The denoised image.
    """
    image_np = np.array(image)
    denoised_image_np = cv2.fastNlMeansDenoisingColored(image_np, None, 6, 6, 7, 21)
    denoised_image = Image.fromarray(denoised_image_np)
    return denoised_image


def blur_image(image: Image.Image, kernel_size: int = 5) -> Image.Image:
    """
    Applies Gaussian blurring to an image.

    Args:
        image (Image.Image): The input image.
        kernel_size (int): The size of the Gaussian kernel (must be odd).

    Returns:
        Image.Image: The blurred image.
    """
    image_np = np.array(image)
    blurred_image_np = cv2.GaussianBlur(image_np, (kernel_size, kernel_size), 0)
    blurred_image = Image.fromarray(blurred_image_np)
    return blurred_image
