from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_functional
from PIL import Image, ImageEnhance, ImageFilter
from ultralytics import YOLOWorld


def apply_corruption(
    image: Image.Image,
    corruption_type: str,
    value: float,
) -> Image.Image:
    if corruption_type == "identity":
        return image.copy()
    if corruption_type == "brightness":
        return ImageEnhance.Brightness(image).enhance(float(value))
    if corruption_type == "gaussian_blur":
        return image.filter(ImageFilter.GaussianBlur(radius=float(value)))
    raise ValueError(f"Unsupported corruption type: {corruption_type}")


def apply_gamma_correction(image: Image.Image, gamma: float) -> Image.Image:
    """Brighten a low-light RGB image with a fixed power-law transform."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    lookup = [
        int(round(255.0 * ((value / 255.0) ** float(gamma))))
        for value in range(256)
    ]
    return image.convert("RGB").point(lookup * 3)


def apply_unsharp_mask(
    image: Image.Image,
    radius: float,
    amount: float,
) -> Image.Image:
    """Sharpen an RGB image without changing its spatial resolution."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    if amount < 0:
        raise ValueError("amount must be non-negative")
    original = np.asarray(image.convert("RGB"), dtype=np.float32)
    blurred = np.asarray(
        image.convert("RGB").filter(
            ImageFilter.GaussianBlur(radius=float(radius))
        ),
        dtype=np.float32,
    )
    sharpened = original + float(amount) * (original - blurred)
    return Image.fromarray(
        np.clip(np.rint(sharpened), 0, 255).astype(np.uint8),
        mode="RGB",
    )


def apply_wiener_deconvolution(
    image: Image.Image,
    blur_sigma: float,
    regularization: float,
    *,
    blend: float = 1.0,
) -> Image.Image:
    """Approximately invert Gaussian blur with a regularized Wiener filter.

    The operation is deterministic, uses no learned weights, and preserves the
    input resolution. Reflect padding reduces FFT boundary ringing.
    """
    if blur_sigma <= 0:
        raise ValueError("blur_sigma must be positive")
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    if not 0.0 <= blend <= 1.0:
        raise ValueError("blend must be in [0, 1]")

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    radius = max(1, int(np.ceil(3.0 * float(blur_sigma))))
    coordinates = np.arange(-radius, radius + 1, dtype=np.float32)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    kernel = np.exp(
        -(xx**2 + yy**2) / (2.0 * float(blur_sigma) ** 2)
    )
    kernel /= float(kernel.sum())

    pad = max(16, radius * 2)
    padded = np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    kernel_padded = np.zeros(padded.shape[:2], dtype=np.float32)
    center_y = kernel_padded.shape[0] // 2
    center_x = kernel_padded.shape[1] // 2
    kernel_padded[
        center_y - radius : center_y + radius + 1,
        center_x - radius : center_x + radius + 1,
    ] = kernel
    kernel_frequency = np.fft.fft2(
        np.fft.ifftshift(kernel_padded)
    )
    inverse = np.conj(kernel_frequency) / (
        np.abs(kernel_frequency) ** 2 + float(regularization)
    )

    restored_channels = []
    for channel_index in range(3):
        observed = np.fft.fft2(padded[:, :, channel_index])
        restored_channels.append(
            np.fft.ifft2(observed * inverse).real.astype(np.float32)
        )
    restored = np.stack(restored_channels, axis=-1)
    restored = restored[pad:-pad, pad:-pad]
    restored = float(blend) * restored + (1.0 - float(blend)) * rgb
    return Image.fromarray(
        np.clip(np.rint(restored * 255.0), 0, 255).astype(np.uint8),
        mode="RGB",
    )


def weighted_spherical_prompt_mean(
    embeddings: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Return a normalized weighted mean of unit text embeddings."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [prompt, dimension]")
    if weights.ndim != 1 or weights.shape[0] != embeddings.shape[0]:
        raise ValueError("weights must have one value per prompt")
    if bool((weights < 0).any()) or float(weights.sum()) <= 0:
        raise ValueError("weights must be non-negative with a positive sum")
    normalized_weights = weights.to(
        device=embeddings.device,
        dtype=embeddings.dtype,
    )
    normalized_weights = normalized_weights / normalized_weights.sum()
    mean = (embeddings * normalized_weights[:, None]).sum(dim=0)
    return torch_functional.normalize(mean, p=2, dim=0)


class YoloWorldPredictor:
    def __init__(self, config: dict[str, Any], category_names: list[str]) -> None:
        model_config = config["model"]
        torch.manual_seed(int(config["seed"]))
        self.model = YOLOWorld(model_config["name"])
        self.device = model_config["device"]
        self.image_size = int(model_config["image_size"])
        self.raw_confidence = float(model_config["raw_confidence"])
        self.internal_nms_iou = float(model_config["internal_nms_iou"])
        self.max_detections = int(model_config["max_detections"])
        self.category_names = category_names
        self.current_prompts: list[str] | None = None

    def set_prompts(self, prompts: list[str]) -> None:
        if prompts != self.current_prompts:
            # Ultralytics caches the CLIP wrapper. After the first prediction,
            # YOLO moves the registered CLIP module to CUDA, but the wrapper's
            # plain ``device`` attribute can still say "cpu". Keep that
            # attribute aligned before tokenizing a new prompt set.
            clip_model = getattr(self.model.model, "clip_model", None)
            if clip_model is not None:
                clip_model.device = next(clip_model.parameters()).device
            self.model.set_classes(prompts)
            self.current_prompts = prompts.copy()

    def set_prompt_prototypes(
        self,
        prompt_groups: list[list[str]],
        prompt_weights: list[list[float]],
    ) -> None:
        """Set one normalized CLIP prototype per canonical output class."""
        if len(prompt_groups) != len(self.category_names):
            raise ValueError("one prompt group is required per category")
        if len(prompt_weights) != len(prompt_groups):
            raise ValueError("prompt groups and weight groups must align")
        if any(not group for group in prompt_groups):
            raise ValueError("prompt groups must not be empty")
        if any(
            len(group) != len(weights)
            for group, weights in zip(prompt_groups, prompt_weights)
        ):
            raise ValueError("each prompt must have one weight")

        world_model = self.model.model
        clip_model = getattr(world_model, "clip_model", None)
        if clip_model is not None:
            clip_model.device = next(clip_model.parameters()).device

        flattened = [
            prompt for group in prompt_groups for prompt in group
        ]
        text_embeddings = world_model.get_text_pe(flattened)[0]
        prototypes: list[torch.Tensor] = []
        offset = 0
        for group, weights in zip(prompt_groups, prompt_weights):
            group_embeddings = text_embeddings[offset : offset + len(group)]
            prototypes.append(
                weighted_spherical_prompt_mean(
                    group_embeddings,
                    torch.as_tensor(
                        weights,
                        device=group_embeddings.device,
                        dtype=group_embeddings.dtype,
                    ),
                )
            )
            offset += len(group)

        world_model.txt_feats = torch.stack(prototypes, dim=0).unsqueeze(0)
        world_model.model[-1].nc = len(self.category_names)
        world_model.names = list(self.category_names)
        if self.model.predictor:
            self.model.predictor.model.names = list(self.category_names)
        self.current_prompts = [
            " + ".join(group) for group in prompt_groups
        ]

    def predict(
        self,
        image: Image.Image,
        *,
        image_size: int | None = None,
    ) -> list[dict[str, Any]]:
        result = self.model.predict(
            source=np.asarray(image.convert("RGB")),
            conf=self.raw_confidence,
            iou=self.internal_nms_iou,
            imgsz=(self.image_size if image_size is None else int(image_size)),
            device=self.device,
            max_det=self.max_detections,
            verbose=False,
        )[0]
        predictions: list[dict[str, Any]] = []
        if result.boxes is None:
            return predictions
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        class_indices = result.boxes.cls.detach().cpu().numpy().astype(int)
        for box, confidence, class_index in zip(boxes, confidences, class_indices):
            if class_index < 0 or class_index >= len(self.category_names):
                continue
            predictions.append(
                {
                    "category": self.category_names[class_index],
                    "prompt": self.current_prompts[class_index],
                    "confidence": float(confidence),
                    "bbox_xyxy": [float(value) for value in box],
                }
            )
        return predictions
