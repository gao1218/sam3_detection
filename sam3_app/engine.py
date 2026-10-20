"""Adapters around the local Meta SAM3 image processor and point predictor."""
from dataclasses import dataclass
from contextlib import nullcontext
import sys
from typing import Any

import numpy as np
from PIL import Image

from .config import settings


@dataclass
class Segmentation:
    masks: list[np.ndarray]
    boxes: list[list[float]]
    scores: list[float]


class Sam3Engine:
    def __init__(self) -> None:
        self.device = settings.device
        self._native = None

    def _inference_context(self):
        import torch
        if self._native and self._native[1].startswith("cuda"):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.autocast(device_type="cuda", dtype=dtype)
        return nullcontext()

    def _load_native(self) -> None:
        if self._native is not None:
            return
        if not settings.source_dir.is_dir():
            raise RuntimeError(f"找不到 SAM3 源码目录：{settings.source_dir}")
        if not settings.checkpoint.is_file():
            raise RuntimeError(f"找不到 SAM3 权重文件：{settings.checkpoint}")
        import psutil
        import torch
        device = self.device if self.device == "cpu" or torch.cuda.is_available() else "cpu"
        available_memory = psutil.virtual_memory().available
        if device.startswith("cuda"):
            # The checkpoint is memory-mapped and loaded directly onto CUDA.
            # Windows can page the temporary CPU-side model construction, so
            # assess pagefile and VRAM as well instead of using a fixed 5 GB RAM gate.
            pagefile_free = psutil.swap_memory().free
            vram_free, _ = torch.cuda.mem_get_info()
            min_ram, min_pagefile, min_vram = 1.25, 6.0, 4.0
            if (available_memory < min_ram * 1024**3
                    or pagefile_free < min_pagefile * 1024**3
                    or vram_free < min_vram * 1024**3):
                raise RuntimeError(
                    "当前资源不足以尝试 CUDA 加载 SAM3："
                    f"系统可用内存 {available_memory / 1024**3:.1f} GB（至少 {min_ram:.1f} GB），"
                    f"页面文件可用 {pagefile_free / 1024**3:.1f} GB（至少 {min_pagefile:.1f} GB），"
                    f"显存可用 {vram_free / 1024**3:.1f} GB（至少 {min_vram:.1f} GB）。"
                    "请释放显存或增加 Windows 页面文件后重试。"
                )
        else:
            required_memory = max(8 * 1024**3, int(settings.checkpoint.stat().st_size * 1.5))
            if available_memory < required_memory:
                raise RuntimeError(
                    "可用内存不足以安全加载 SAM3："
                    f"当前约 {available_memory / 1024**3:.1f} GB，建议至少留出 "
                    f"{required_memory / 1024**3:.1f} GB。请关闭占内存程序后重试。"
                )
        source_dir = str(settings.source_dir.resolve())
        if source_dir not in sys.path:
            sys.path.insert(0, source_dir)
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as exc:
            raise RuntimeError(f"SAM3 导入失败：{exc}。请使用 python_embeded 环境安装 sam3 目录的项目依赖。") from exc
        model = build_sam3_image_model(
            checkpoint_path=str(settings.checkpoint.resolve()),
            load_from_HF=False,
            enable_inst_interactivity=True,
            device=device,
        )
        self._native = (Sam3Processor(model, device=device, confidence_threshold=settings.confidence_threshold), device)

    @staticmethod
    def _to_result(output: dict[str, Any]) -> Segmentation:
        # CUDA autocast can return bfloat16 tensors, which NumPy cannot convert.
        masks = output["masks"].detach().float().cpu().numpy()
        boxes = output["boxes"].detach().float().cpu().numpy().tolist()
        scores = output["scores"].detach().float().cpu().numpy().tolist()
        masks = np.squeeze(masks, axis=1) if masks.ndim == 4 and masks.shape[1] == 1 else masks
        return Segmentation([m.astype(bool) for m in masks], boxes, scores)

    def text(self, image: Image.Image, prompt: str) -> Segmentation:
        self._load_native()
        processor, _ = self._native
        with self._inference_context():
            state = processor.set_image(image.convert("RGB"))
            output = processor.set_text_prompt(state=state, prompt=prompt)
        return self._to_result(output)

    def box(self, image: Image.Image, box_xyxy: tuple[int, int, int, int]) -> Segmentation:
        return self.boxes(image, [(box_xyxy, True)])

    def boxes(
        self,
        image: Image.Image,
        prompts: list[tuple[tuple[int, int, int, int], bool]],
    ) -> Segmentation:
        """Apply multiple positive/negative box prompts in one image state."""
        if not any(is_positive for _, is_positive in prompts):
            raise ValueError("框提示至少需要一个包含框；屏蔽框用于排除区域。")
        self._load_native()
        processor, _ = self._native
        w, h = image.size
        with self._inference_context():
            state = processor.set_image(image.convert("RGB"))
            output = None
            for (x1, y1, x2, y2), is_positive in prompts:
                cx, cy = (x1 + x2) / (2 * w), (y1 + y2) / (2 * h)
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                output = processor.add_geometric_prompt(
                    box=[cx, cy, bw, bh], label=is_positive, state=state
                )
        if output is None:
            raise ValueError("请先绘制包含框或屏蔽框。")
        return self._to_result(output)

    def points(
        self,
        image: Image.Image,
        points: list[tuple[int, int, bool]],
        point_target_ids: list[int | None] | None = None,
    ) -> Segmentation:
        """Segment point groups; a missing target id makes each positive click its own target."""
        self._load_native()
        processor, _ = self._native
        if point_target_ids is not None and len(point_target_ids) != len(points):
            raise ValueError("点提示目标分组数量与点数量不一致。")
        target_groups: dict[object, list[tuple[int, int]]] = {}
        negative_points = [(x, y) for x, y, positive in points if not positive]
        for index, (x, y, positive) in enumerate(points):
            if not positive:
                continue
            target_id = point_target_ids[index] if point_target_ids is not None else None
            # None deliberately gives this click a unique group.
            group_key: object = target_id if target_id is not None else ("point", index)
            target_groups.setdefault(group_key, []).append((x, y))
        if not target_groups:
            raise ValueError("点提示至少需要一个前景点。")

        masks_out: list[np.ndarray] = []
        boxes: list[list[float]] = []
        scores_out: list[float] = []
        with self._inference_context():
            state = processor.set_image(image.convert("RGB"))
            # Treat each foreground click as an independent target. Keeping
            # negatives in every query lets the user exclude shared background.
            for positive_points in target_groups.values():
                query_points = [*positive_points, *negative_points]
                point_coords = np.asarray(query_points, dtype=np.float32)
                point_labels = np.asarray([*([1] * len(positive_points)), *([0] * len(negative_points))], dtype=np.int32)
                masks, scores, _ = processor.model.predict_inst(
                    state,
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                    normalize_coords=False,
                )
                masks = masks.detach().float().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
                scores = scores.detach().float().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores)
                masks = np.squeeze(masks)
                scores = np.asarray(scores).reshape(-1)
                best = int(np.argmax(scores)) if scores.size else 0
                mask = masks[best].astype(bool) if masks.ndim == 3 else masks.astype(bool)
                masks_out.append(mask)
                scores_out.append(float(scores[best]) if scores.size else 0.0)
                ys, xs = np.where(mask)
                boxes.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                             if len(xs) else [0, 0, 0, 0])
        return Segmentation(masks_out, boxes, scores_out)

