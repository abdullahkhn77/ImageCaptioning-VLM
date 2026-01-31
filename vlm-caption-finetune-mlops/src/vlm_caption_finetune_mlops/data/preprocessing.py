"""
Image preprocessing for VLM inputs (resize, normalize, format).
"""

from pathlib import Path
from typing import List, Union

import torch
from PIL import Image


def preprocess_image(
    image: Union[Image.Image, str, Path],
    size: Union[int, List[int]] = 224,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Preprocess a single image for VLM input.

    Args:
        image: PIL Image, or path to image file.
        size: Target size (int for square, or [H, W]).
        normalize: Whether to normalize to [0, 1] or ImageNet stats.

    Returns:
        Tensor of shape (C, H, W) in range expected by the model.
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    if isinstance(size, int):
        size = (size, size)
    # Resize (simple; processor usually does more)
    image = image.resize((size[1], size[0]), Image.BILINEAR)
    # To tensor [C, H, W]
    tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    if normalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
    return tensor
