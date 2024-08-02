# src/utils/preprocessing/image.py

# 2. Image Files
# Resizing: Standardize the dimensions of images for uniformity.
# Normalization: Adjust pixel values to a standard range (e.g., 0-1 or -1 to 1) to help with model training.
# Data Augmentation: Techniques like rotation, flipping, or cropping can increase the diversity of the dataset and improve model robustness.
# Color Adjustment: Convert to grayscale if color information is not needed, or normalize color channels.
# Denoising: Remove any noise present in the images to enhance clarity.

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def process_image(
    image_path: str,
    output_path: str,
    size: tuple = (224, 224),
    mean: list = [0.5, 0.5, 0.5],
    std: list = [0.5, 0.5, 0.5],
    to_grayscale: bool = False,
) -> None:
    """
    Processes an image by resizing, normalizing, augmenting, optionally converting to grayscale, normalizing color channels, and denoising.

    Args:
        image_path (str): The path to the input image file.
        output_path (str): The path to save the processed image file.
        size (tuple): The desired dimensions (width, height) for resizing. Default is (224, 224).
        mean (list): The mean values for each channel for normalization. Default is [0.5, 0.5, 0.5].
        std (list): The standard deviation values for each channel for normalization. Default is [0.5, 0.5, 0.5].
        to_grayscale (bool): Whether to convert the image to grayscale. Default is False.
    """
    image = Image.open(image_path)

    image = resize_image(image, size)
    image = augment_image(image)

    if to_grayscale:
        image = convert_to_grayscale(image)

    transform_to_tensor = transforms.ToTensor()
    image_tensor = transform_to_tensor(image)

    image_tensor = normalize_color_channels(image_tensor, mean, std)

    transform_to_pil = transforms.ToPILImage()
    image = transform_to_pil(image_tensor)

    image = denoise_image(image)
    image.save(output_path)


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


def augment_image(image: Image.Image) -> Image.Image:
    """
    Applies data augmentation techniques like rotation, flipping, and cropping to an image.

    Args:
        image (Image.Image): The input image.

    Returns:
        Image.Image: The augmented image.
    """
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(size=image.size, scale=(0.8, 1.0)),
        ]
    )
    return transform(image)


def convert_to_grayscale(image: Image.Image) -> Image.Image:
    """
    Converts an image to grayscale.

    Args:
        image (Image.Image): The input image.

    Returns:
        Image.Image: The grayscale image.
    """
    return image.convert("L")


def normalize_color_channels(
    image: torch.Tensor, mean: list = [0.5, 0.5, 0.5], std: list = [0.5, 0.5, 0.5]
) -> torch.Tensor:
    """
    Normalizes the color channels of an image tensor.

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
