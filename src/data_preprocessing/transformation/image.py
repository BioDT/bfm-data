"""Copyright (C) 2025 TNO, The Netherlands. Licensed under the MIT license."""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image


def augment_image(image: Image.Image) -> Image.Image:
    """
    Applies data augmentation techniques like rotation, flipping, zooming, and shearing to an image.

    Args:
        image (Image.Image): The input image.

    Returns:
        Image.Image: The augmented image.
    """
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop(size=image.size, scale=(0.8, 1.0)),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.RandomAffine(degrees=0, shear=10),
        ]
    )
    return transform(image)


def convert_color_space(image: Image.Image, target_space: str = "gray") -> Image.Image:
    """
    Converts an image to a specified color space.

    Args:
        image (Image.Image): The input image.
        target_space (str): The target color space ('gray', 'hsv', 'lab').

    Returns:
        Image.Image: The converted image.
    """
    image_np = np.array(image)
    if target_space == "gray":
        converted_image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        return Image.fromarray(converted_image_np)
    elif target_space == "hsv":
        converted_image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    elif target_space == "lab":
        converted_image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    else:
        raise ValueError("Target color space must be 'gray', 'hsv', or 'lab'.")

    converted_image = Image.fromarray(converted_image_np)
    return converted_image


def normalise_image(
    image: torch.Tensor, mean: list = [0.5, 0.5, 0.5], std: list = [0.5, 0.5, 0.5]
) -> torch.Tensor:
    """
    Normalises an image tensor to have a standard range of values.

    Args:
        image (torch.Tensor): The input image tensor.
        mean (list): The mean values for each channel.
        std (list): The standard deviation values for each channel.

    Returns:
        torch.Tensor: The normalised image tensor.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )
    return transform(image)


def denormalise_tensor(
    tensor: torch.Tensor, mean: list = [0.5, 0.5, 0.5], std: list = [0.5, 0.5, 0.5]
) -> torch.Tensor:
    """
    Denormalises a tensor by reversing the normalisation.

    Args:
        tensor (torch.Tensor): The normalised tensor.
        mean (list): Mean used in the normalisation.
        std (list): Standard deviation used in the normalisation.

    Returns:
        torch.Tensor: The denormalised tensor.
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    denormalised_tensor = tensor * std + mean
    return denormalised_tensor


def equalize_histogram(image: Image.Image) -> Image.Image:
    """
    Adjusts the contrast of an image using histogram equalization.

    Args:
        image (Image.Image): The input image.

    Returns:
        Image.Image: The contrast-enhanced image.
    """
    image_np = np.array(image)

    if len(image_np.shape) == 2:
        equalized_image_np = cv2.equalizeHist(image_np)
    else:
        img_yuv = cv2.cvtColor(image_np, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        equalized_image_np = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    equalized_image = Image.fromarray(equalized_image_np)
    return equalized_image


def resize_image_tensor(
    image_tensor: torch.Tensor, target_shape: tuple
) -> torch.Tensor:
    """
    Resizes an image tensor to the target shape using torchvision transforms.

    Args:
        image_tensor (torch.Tensor): The input image tensor with shape (3, H, W).
        target_shape (tuple): The target shape for the image (e.g., (3, 64, 64)).

    Returns:
        torch.Tensor: The resized image tensor.
    """
    resize = transforms.Resize((target_shape[1], target_shape[2]))
    return resize(image_tensor.unsqueeze(0)).squeeze(0)
