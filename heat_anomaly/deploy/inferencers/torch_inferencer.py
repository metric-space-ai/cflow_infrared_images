"""This module contains Torch inference implementations."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch import Tensor

from heat_anomaly.config import get_configurable_parameters
from heat_anomaly.deploy.export import get_model_metadata
from heat_anomaly.models import get_model
from heat_anomaly.models.components import AnomalyModule
from heat_anomaly.pre_processing import PreProcessor

from .base_inferencer import Inferencer


class TorchInferencer(Inferencer):
    def __init__(
        self,
        config: Union[str, Path, DictConfig, ListConfig],
        model_source: Union[str, Path, AnomalyModule],
        meta_data_path: Union[str, Path] = None,
    ):

        # Check and load the configuration
        if isinstance(config, (str, Path)):
            self.config = get_configurable_parameters(config_path=config)
        elif isinstance(config, (DictConfig, ListConfig)):
            self.config = config
        else:
            raise ValueError(f"Unknown config type {type(config)}")

        # Check and load the model weights.
        if isinstance(model_source, AnomalyModule):
            self.model = model_source
        else:
            self.model = self.load_model(model_source)

        self.meta_data = self._load_meta_data(meta_data_path)

    def _load_meta_data(self, path: Optional[Union[str, Path]] = None) -> Union[Dict, DictConfig]:
        meta_data: Union[DictConfig, Dict[str, Union[float, Tensor, np.ndarray]]]
        if path is None:
            meta_data = get_model_metadata(self.model)
        else:
            meta_data = super()._load_meta_data(path)
        return meta_data

    def load_model(self, path: Union[str, Path]) -> AnomalyModule:
        model = get_model(self.config)
        model.load_state_dict(torch.load(path)["state_dict"])
        model.to('cuda')
        model.eval()
        return model

    def pre_process(self, image: np.ndarray) -> Tensor:
        transform_config = (
            self.config.dataset.transform_config.val if "transform_config" in self.config.dataset.keys() else None
        )
        image_size = tuple(self.config.dataset.image_size)
        pre_processor = PreProcessor(transform_config, image_size)
        processed_image = pre_processor(image=image)["image"]

        if len(processed_image) == 3:
            processed_image = processed_image.unsqueeze(0)

        return processed_image.to('cuda')

    def forward(self, image: Tensor) -> Tensor:
        return self.model(image)

    def post_process(self, predictions: Tensor, meta_data: Optional[Union[Dict, DictConfig]] = None) -> Dict[str, Any]:
        
        if meta_data is None:
            meta_data = self.meta_data

        if isinstance(predictions, Tensor):
            anomaly_map = predictions.detach().cpu().numpy()
            pred_score = anomaly_map.reshape(-1).max()
        else:
            if isinstance(predictions[1], (Tensor)):
                anomaly_map, pred_score = predictions
                anomaly_map = anomaly_map.detach().cpu().numpy()
                pred_score = pred_score.detach().cpu().numpy()
            else:
                anomaly_map, pred_score = predictions
                pred_score = pred_score.detach()

        pred_label: Optional[str] = None
        if "image_threshold" in meta_data:
            pred_idx = pred_score >= meta_data["image_threshold"]
            pred_label = "Anomalous" if pred_idx else "Normal"

        pred_mask: Optional[np.ndarray] = None
        if "pixel_threshold" in meta_data:
            pred_mask = (anomaly_map >= meta_data["pixel_threshold"]).squeeze().astype(np.uint8)

        anomaly_map = anomaly_map.squeeze()
        anomaly_map, pred_score = self._normalize(anomaly_map, pred_score, meta_data)

        if isinstance(anomaly_map, Tensor):
            anomaly_map = anomaly_map.detach().cpu().numpy()

        if "image_shape" in meta_data and anomaly_map.shape != meta_data["image_shape"]:
            image_height = meta_data["image_shape"][0]
            image_width = meta_data["image_shape"][1]
            anomaly_map = cv2.resize(anomaly_map, (image_width, image_height))

            if pred_mask is not None:
                pred_mask = cv2.resize(pred_mask, (image_width, image_height))

        return {
            "anomaly_map": anomaly_map,
            "pred_label": pred_label,
            "pred_score": pred_score,
            "pred_mask": pred_mask,
        }
