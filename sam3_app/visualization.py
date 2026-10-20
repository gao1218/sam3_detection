"""Utilities for blending masks and drawing segmentation results."""
from PIL import Image, ImageDraw
import numpy as np


def overlay(image: Image.Image, masks: list[np.ndarray], boxes: list[list[float]],
            scores: list[float], alpha: float = 0.48) -> Image.Image:
    base = np.asarray(image.convert("RGB")).copy()
    colors = [(38, 190, 255), (255, 122, 65), (127, 220, 90), (204, 112, 255)]
    for index, mask in enumerate(masks):
        color = np.asarray(colors[index % len(colors)], dtype=np.uint8)
        base[mask] = (base[mask] * (1 - alpha) + color * alpha).astype(np.uint8)
    result = Image.fromarray(base)
    draw = ImageDraw.Draw(result)
    for i, box in enumerate(boxes):
        draw.rectangle(tuple(map(float, box)), outline=(255, 235, 50), width=3)
        if i < len(scores):
            draw.text((box[0] + 3, box[1] + 3), f"{scores[i]:.2f}", fill="white", stroke_width=1, stroke_fill="black")
    return result

