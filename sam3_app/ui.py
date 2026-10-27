"""Streamlit UI for text, point, and box image prompts."""
import io
import hashlib
import re
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

import streamlit.elements.image as streamlit_image

if not hasattr(streamlit_image, "image_to_url"):
    # streamlit-drawable-canvas 0.9.x expects an older Streamlit helper.
    # Adapt that call to the current image_utils API while keeping the existing
    # canvas widget and its background-image behavior.
    from streamlit.elements.lib.image_utils import image_to_url
    from streamlit.elements.lib.layout_utils import create_layout_config

    def _canvas_image_to_url(image, width, clamp, channels, output_format, image_id):
        layout = create_layout_config(width=width)
        return image_to_url(image, layout, clamp, channels, output_format, image_id)

    streamlit_image.image_to_url = _canvas_image_to_url

from streamlit_drawable_canvas import st_canvas

from .engine import Sam3Engine
from .config import PROJECT_ROOT
from .visualization import overlay


TARGET_COLORS = [
    (22, 160, 133), (52, 152, 219), (155, 89, 182), (241, 196, 15),
    (230, 126, 34), (46, 204, 113), (52, 73, 94), (26, 188, 156),
]


def _point_color(rgb: tuple[int, int, int], alpha: float = 0.55) -> str:
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})"


def _save_mask_with_original_name(mask_image: Image.Image, path: Path) -> None:
    """Save a mask using the source filename's exact extension and spelling."""
    image_format = {
        ".png": "PNG",
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".webp": "WEBP",
    }.get(path.suffix.lower())
    if image_format is None:
        raise ValueError(f"不支持将 mask 保存为此文件扩展名：{path.suffix}")
    image = mask_image
    if image_format in {"JPEG", "WEBP"} and image.mode not in {"L", "RGB"}:
        image = image.convert("L")
    if image_format == "JPEG":
        image.save(path, format=image_format, quality=100, subsampling=0)
    elif image_format == "WEBP":
        image.save(path, format=image_format, lossless=True)
    else:
        image.save(path, format=image_format)


@st.cache_resource
def get_engine() -> Sam3Engine:
    return Sam3Engine()


def main() -> None:
    st.set_page_config(page_title="SAM 3 图像分割", layout="wide")
    st.title("SAM 3 图像分割")
    st.caption("上传图像后选择文字、点或框提示。文字/框使用 Meta 原生 SAM3 图像接口；点提示使用 SAM3 Tracker 图像接口。")
    uploaded = st.file_uploader("选择图像", type=["png", "jpg", "jpeg", "webp"])
    if not uploaded:
        st.info("请先上传一张图像。")
        return
    image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
    image_id = hashlib.sha256(uploaded.getvalue()).hexdigest()
    if st.session_state.get("segmentation_image_id") != image_id:
        st.session_state.pop("segmentation_result", None)
        st.session_state["segmentation_image_id"] = image_id
    mode = st.radio("提示方式", ["文字提示", "交互式点提示", "交互式框提示"], index=2, horizontal=True)
    max_width = min(900, image.width)
    scale = max_width / image.width
    display = image.resize((max_width, max(1, round(image.height * scale))))
    prompt_text = st.text_input("目标描述", placeholder="例如：红色汽车") if mode == "文字提示" else ""
    point_positive = True
    box_action = "绘制框"
    box_positive = True
    point_group_mode = "每个前景点单独一个目标"
    active_target = 0
    target_count = 1
    if mode == "文字提示":
        st.image(display, caption="输入图像", use_container_width=False)
    else:
        if mode == "交互式点提示":
            point_positive = st.radio("点标签", ["前景点", "背景点"], horizontal=True) == "前景点"
            point_group_mode = st.radio("前景点分组", ["每个前景点单独一个目标", "手动分组（同一目标可加多个点）"],
                                        horizontal=True, key=f"point-groups-{uploaded.name}")
            if point_group_mode == "手动分组（同一目标可加多个点）":
                target_count = st.number_input("目标数量", min_value=1, max_value=len(TARGET_COLORS),
                                               value=2, step=1, key=f"target-count-{uploaded.name}")
                active_target = st.selectbox("当前添加点的目标", range(int(target_count)),
                                             format_func=lambda i: f"目标 {i + 1}",
                                             key=f"active-target-{uploaded.name}")
        else:
            box_action = st.radio("框操作", ["绘制框", "移动/调整框"], horizontal=True,
                                  key=f"box-action-{uploaded.name}")
            box_positive = st.radio("框标签", ["包含框", "屏蔽框"], horizontal=True,
                                    key=f"box-label-{uploaded.name}") == "包含框"
        if mode == "交互式点提示":
            if point_group_mode.startswith("手动分组"):
                st.caption("选择目标后添加前景点，同一目标可添加多个点；背景点会用于约束所有目标。不同颜色表示不同目标。")
            else:
                st.caption("每个前景点单独分割一个目标；背景点会用于排除误分区域。")
        if mode == "交互式框提示":
            st.caption("包含框标记要分割的区域；屏蔽框标记需要排除的区域。绘制框模式下拖动新增当前标签的框，移动/调整模式下拖动已有框不会新增框。")
        target_rgb = TARGET_COLORS[active_target] if point_group_mode.startswith("手动分组") else TARGET_COLORS[0]
        point_color = ("#%02x%02x%02x" % target_rgb) if point_positive else "#e74c3c"
        point_fill = _point_color(target_rgb) if point_positive else "rgba(231,76,60,0.55)"
        canvas_stroke = point_color if mode == "交互式点提示" or box_positive else "#e74c3c"
        canvas_fill = point_fill if mode == "交互式点提示" or box_positive else "rgba(231,76,60,0.35)"
        canvas = st_canvas(background_image=display, height=display.height, width=display.width,
                           drawing_mode=("point" if mode == "交互式点提示" else
                                         ("rect" if box_action == "绘制框" else "transform")),
                           point_display_radius=6, stroke_width=3, stroke_color=canvas_stroke,
                           fill_color=canvas_fill, update_streamlit=True,
                           key=f"canvas-{uploaded.name}-{mode}")
    run = st.button("开始分割", type="primary")
    if run:
        try:
            engine = get_engine()
            if mode == "文字提示":
                if not prompt_text.strip():
                    st.warning("请输入目标描述。")
                    return
                result = engine.text(image, prompt_text.strip())
            elif mode == "交互式框提示":
                objects = (canvas.json_data or {}).get("objects", [])
                rects = [o for o in objects if o.get("type") == "rect" and o.get("width", 0) > 2 and o.get("height", 0) > 2]
                if not rects:
                    st.warning("请先绘制包含框；屏蔽框不能单独用于分割。")
                    return
                box_prompts = []
                for rect in rects:
                    x1 = round(rect["left"] / scale); y1 = round(rect["top"] / scale)
                    x2 = round((rect["left"] + rect["width"] * rect.get("scaleX", 1)) / scale)
                    y2 = round((rect["top"] + rect["height"] * rect.get("scaleY", 1)) / scale)
                    fill = str(rect.get("fill", "")).replace(" ", "").lower()
                    is_positive = "231,76,60" not in fill and "e74c3c" not in fill
                    coords = (max(0, x1), max(0, y1), min(image.width, x2), min(image.height, y2))
                    if coords[2] > coords[0] and coords[3] > coords[1]:
                        box_prompts.append((coords, is_positive))
                if not any(is_positive for _, is_positive in box_prompts):
                    st.warning("至少需要一个包含框；屏蔽框只负责排除区域。")
                    return
                result = engine.boxes(image, box_prompts)
            else:
                objects = (canvas.json_data or {}).get("objects", [])
                points = []
                point_target_ids = []
                for obj in objects:
                    if obj.get("type") == "circle":
                        radius = obj.get("radius", 0)
                        x = round((obj["left"] + radius * obj.get("scaleX", 1)) / scale)
                        y = round((obj["top"] + radius * obj.get("scaleY", 1)) / scale)
                        fill = obj.get("fill", "")
                        normalized_fill = fill.replace(" ", "").lower()
                        positive = "231,76,60" not in normalized_fill and "e74c3c" not in normalized_fill
                        points.append((x, y, positive))
                        if positive and point_group_mode.startswith("手动分组"):
                            matching_target = next((target_id for target_id, rgb in enumerate(TARGET_COLORS[:int(target_count)])
                                                    if normalized_fill == _point_color(rgb).lower()), active_target)
                            point_target_ids.append(matching_target)
                        else:
                            point_target_ids.append(None)
                if not any(positive for _, _, positive in points):
                    st.warning("请至少添加一个前景点；每个前景点代表一个目标，背景点会用于排除误分区域。")
                    return
                result = engine.points(
                    image, points,
                    point_target_ids=point_target_ids if point_group_mode.startswith("手动分组") else None,
                )
            st.session_state["segmentation_result"] = result
            st.session_state["segmentation_mode"] = mode
        except Exception as exc:
            st.error(str(exc))

    result = st.session_state.get("segmentation_result")
    if result is None:
        return
    st.success(f"完成：分割出 {len(result.masks)} 个目标/掩码。")
    left, right = st.columns(2)
    overlay_image = overlay(image, result.masks, result.boxes, result.scores)
    with left:
        st.image(overlay_image, caption="分割结果（原图叠加 mask）", use_container_width=True)
    with right:
        st.image(image, caption="原图", use_container_width=True)

    if result.masks:
        mask_output = st.radio("Mask 显示/保存方式", ["分别查看/保存", "组合为二值并集 mask", "组合为目标编号 mask"],
                               index=1, horizontal=True, key=f"mask-output-{image_id}")
        mask_bytes = []
        mask_images = []
        mask_image_to_save = None
        for mask in result.masks:
            buffer = io.BytesIO()
            mask_image = Image.fromarray(mask.astype("uint8") * 255, mode="L")
            mask_images.append(mask_image)
            mask_image.save(buffer, format="PNG")
            mask_bytes.append(buffer.getvalue())
        if mask_output == "分别查看/保存":
            selected = st.selectbox("查看单个目标的 mask", range(len(result.masks)),
                                    format_func=lambda i: f"目标 {i + 1}（置信度 {result.scores[i]:.3f}）")
            mask_col, download_col = st.columns([2, 1])
            with mask_col:
                st.image(mask_bytes[selected], caption=f"目标 {selected + 1} 的二值 mask（白色为前景）", use_container_width=True)
            mask_image_to_save = mask_images[selected]
            with download_col:
                st.download_button("保存此目标 mask（PNG）", data=mask_bytes[selected],
                                   file_name=f"sam3_mask_{selected + 1:02d}.png", mime="image/png")
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                for index, data in enumerate(mask_bytes, start=1):
                    archive.writestr(f"mask_{index:02d}.png", data)
            st.download_button("保存全部 masks（ZIP）", data=zip_buffer.getvalue(),
                               file_name="sam3_masks.zip", mime="application/zip")
        elif mask_output == "组合为二值并集 mask":
            union_mask = np.logical_or.reduce(result.masks)
            union_image = Image.fromarray(union_mask.astype("uint8") * 255, mode="L")
            mask_images = [union_image]
            mask_image_to_save = union_image
            union_buffer = io.BytesIO()
            union_image.save(union_buffer, format="PNG")
            st.image(union_image, caption="合并后的二值 mask（所有目标并集）", use_container_width=True)
            st.download_button("保存合并 mask（PNG）", data=union_buffer.getvalue(),
                               file_name="sam3_mask_union.png", mime="image/png")
        else:
            labels = np.zeros(result.masks[0].shape, dtype=np.uint16)
            for index, mask in enumerate(result.masks, start=1):
                labels[np.asarray(mask, dtype=bool) & (labels == 0)] = index
            label_image = Image.fromarray(labels, mode="I;16")
            mask_images = [label_image]
            mask_image_to_save = label_image
            label_buffer = io.BytesIO()
            label_image.save(label_buffer, format="PNG")
            palette = np.zeros((*labels.shape, 3), dtype=np.uint8)
            for index in range(1, len(result.masks) + 1):
                palette[labels == index] = TARGET_COLORS[(index - 1) % len(TARGET_COLORS)]
            st.image(Image.fromarray(palette), caption="目标编号预览（重叠区域归入编号较小的目标）", use_container_width=True)
            st.caption("下载的灰度 PNG 中，0 表示背景，1、2、3… 对应目标编号；重叠像素归入编号较小的目标。")
            st.download_button("保存目标编号 mask（PNG）", data=label_buffer.getvalue(),
                               file_name="sam3_mask_labels.png", mime="image/png")

        st.caption("两个按钮可分别保存到项目目录下的 masked 文件夹。")
        save_overlay_col, save_mask_col = st.columns(2)
        original_stem = Path(uploaded.name).stem
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", original_stem).strip("._") or "image"
        output_dir = PROJECT_ROOT / "masked"
        with save_overlay_col:
            if st.button("仅保存叠加图", type="primary"):
                try:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    overlay_path = output_dir / f"{safe_stem}_masked.png"
                    overlay_image.save(overlay_path, format="PNG")
                    st.success(f"叠加图已保存：{overlay_path.resolve()}")
                except Exception as exc:
                    st.error(f"叠加图保存失败：{exc}")
        with save_mask_col:
            if st.button("仅保存 mask"):
                try:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    mask_path = output_dir / uploaded.name
                    if mask_path.name != uploaded.name or mask_path.resolve().parent != output_dir.resolve():
                        raise ValueError("原图文件名包含路径部分，无法安全地按原名保存。")
                    if mask_image_to_save is None:
                        raise ValueError("当前没有可保存的 mask。")
                    _save_mask_with_original_name(mask_image_to_save, mask_path)
                    st.success(f"Mask 已按原图文件名保存：{mask_path.resolve()}")
                except Exception as exc:
                    st.error(f"Mask 保存失败：{exc}")


