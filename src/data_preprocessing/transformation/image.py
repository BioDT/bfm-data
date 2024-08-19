# src/data_preprocessing/transformation/image.py

import torchvision.transforms as transforms
from PIL import Image


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
