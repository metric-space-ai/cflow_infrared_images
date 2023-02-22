import math
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np
from skimage import morphology


class ThresholdMethod(str, Enum):
    """Threshold method to apply post-processing to the output predictions."""

    ADAPTIVE = "adaptive"
    MANUAL = "manual"


def add_label(
    image: np.ndarray,
    label_name: str,
    color: Tuple[int, int, int],
    confidence: Optional[float] = None,
    font_scale: float = 5e-3,
    thickness_scale=1e-3,
):
    image = image.copy()
    img_height, img_width, _ = image.shape

    font = cv2.FONT_HERSHEY_PLAIN
    text = label_name if confidence is None else f"{label_name} ({confidence*100:.0f}%)"

    # get font sizing
    font_scale = min(img_width, img_height) * font_scale
    thickness = math.ceil(min(img_width, img_height) * thickness_scale)
    (width, height), baseline = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)

    # create label
    label_patch = np.zeros((height + baseline, width + baseline, 3), dtype=np.uint8)
    label_patch[:, :] = color
    cv2.putText(
        label_patch,
        text,
        (0, baseline // 2 + height),
        font,
        fontScale=font_scale,
        thickness=thickness,
        color=0,
        lineType=cv2.LINE_AA,
    )

    # add label to image
    image[: baseline + height, : baseline + width] = label_patch
    return image


def add_normal_label(image: np.ndarray, confidence: Optional[float] = None):
    """Adds the normal label to the image."""
    return add_label(image, "normal", (225, 252, 134), confidence)


def add_anomalous_label(image: np.ndarray, confidence: Optional[float] = None):
    """Adds the anomalous label to the image."""
    return add_label(image, "anomalous", (255, 100, 100), confidence)


def anomaly_map_to_color_map(anomaly_map: np.ndarray, normalize: bool = True) -> np.ndarray:
    if normalize:
        anomaly_map = (anomaly_map - anomaly_map.min()) / np.ptp(anomaly_map)
    anomaly_map = anomaly_map * 255
    anomaly_map = anomaly_map.astype(np.uint8)

    anomaly_map = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
    anomaly_map = cv2.cvtColor(anomaly_map, cv2.COLOR_BGR2RGB)
    return anomaly_map


def superimpose_anomaly_map(
    anomaly_map: np.ndarray, image: np.ndarray, alpha: float = 0.4, gamma: int = 0, normalize: bool = False
) -> np.ndarray:

    anomaly_map = anomaly_map_to_color_map(anomaly_map.squeeze(), normalize=normalize)
    superimposed_map = cv2.addWeighted(anomaly_map, alpha, image, (1 - alpha), gamma)
    return superimposed_map


def compute_mask(anomaly_map: np.ndarray, threshold: float, kernel_size: int = 4) -> np.ndarray:

    anomaly_map = anomaly_map.squeeze()
    mask: np.ndarray = np.zeros_like(anomaly_map).astype(np.uint8)
    mask[anomaly_map > threshold] = 1

    kernel = morphology.disk(kernel_size)
    mask = morphology.opening(mask, kernel)

    mask *= 255

    return mask
