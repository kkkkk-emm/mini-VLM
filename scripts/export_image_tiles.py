#!/usr/bin/env python3
"""Export mini-VLM image processor tiles as PNG files."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.processors import get_image_processor
from models.config import VLMConfig


def _tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().float()
    tensor = (tensor * 0.5 + 0.5).clamp(0.0, 1.0)
    tensor = (tensor * 255.0).round().to(torch.uint8)
    array = tensor.permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def _save_contact_sheet(
    *,
    output_dir: Path,
    image_files: list[tuple[str, str]],
    tile_size: int,
) -> str:
    columns = min(4, max(1, len(image_files)))
    rows = math.ceil(len(image_files) / columns)
    label_height = 22
    sheet = Image.new(
        "RGB",
        (columns * tile_size, rows * (tile_size + label_height)),
        color="white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, (label, filename) in enumerate(image_files):
        row = index // columns
        col = index % columns
        x = col * tile_size
        y = row * (tile_size + label_height)
        with Image.open(output_dir / filename) as image:
            sheet.paste(image.convert("RGB"), (x, y))
        draw.text((x + 4, y + tile_size + 4), label, fill="black")
    filename = "contact_sheet.png"
    sheet.save(output_dir / filename)
    return filename


def export_image_tiles(
    *,
    image_path: Path | str,
    output_dir: Path | str,
    max_image_size: int,
    split_size: int,
    resize_to_max_side_len: bool = False,
    make_contact_sheet: bool = True,
) -> dict[str, Any]:
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    if max_image_size <= 0:
        raise ValueError("max_image_size must be positive")
    if split_size <= 0:
        raise ValueError("split_size must be positive")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    original_size = list(image.size)
    processor = get_image_processor(
        max_image_size,
        split_size,
        resize_to_max_side_len,
    )
    processed_images, grid = processor(image)
    nh, nw = grid

    global_patch = None
    local_start = 0
    contact_items = []
    if grid != (1, 1):
        global_patch = "resized_or_global.png"
        _tensor_to_image(processed_images[0]).save(output_dir / global_patch)
        local_start = 1
        contact_items.append(("global", global_patch))

    local_tiles = []
    for local_index, tensor_index in enumerate(range(local_start, processed_images.size(0))):
        row = local_index // nw + 1
        col = local_index % nw + 1
        filename = f"tile_r{row}_c{col}.png"
        _tensor_to_image(processed_images[tensor_index]).save(output_dir / filename)
        local_tiles.append({"row": row, "col": col, "filename": filename})
        contact_items.append((f"r{row}c{col}", filename))

    contact_sheet = None
    if make_contact_sheet and contact_items:
        contact_sheet = _save_contact_sheet(
            output_dir=output_dir,
            image_files=contact_items,
            tile_size=split_size,
        )

    metadata = {
        "image": str(image_path),
        "output_dir": str(output_dir),
        "original_size": original_size,
        "max_image_size": max_image_size,
        "split_size": split_size,
        "resize_to_max_side_len": resize_to_max_side_len,
        "grid": [nh, nw],
        "num_local_tiles": len(local_tiles),
        "global_patch": global_patch,
        "local_tiles": local_tiles,
        "contact_sheet": contact_sheet,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def parse_args(argv=None):
    defaults = VLMConfig()
    default_image = Path("data/eval_images/image-01-golden-dog-balloons.jpg")
    parser = argparse.ArgumentParser(
        description="Export mini-VLM image processor tiles as PNG files",
    )
    parser.add_argument("--image", type=Path, default=default_image)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-image-size", type=int, default=defaults.max_img_size)
    parser.add_argument("--split-size", type=int, default=defaults.vit_img_size)
    parser.add_argument(
        "--resize-to-max-side-len",
        action=argparse.BooleanOptionalAction,
        default=defaults.resize_to_max_side_len,
    )
    parser.add_argument(
        "--no-contact-sheet",
        dest="make_contact_sheet",
        action="store_false",
        help="Do not write contact_sheet.png.",
    )
    parser.set_defaults(make_contact_sheet=True)
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = Path("results") / "image_tiles" / args.image.stem
    return args


def main(argv=None):
    args = parse_args(argv)
    metadata = export_image_tiles(
        image_path=args.image,
        output_dir=args.output_dir,
        max_image_size=args.max_image_size,
        split_size=args.split_size,
        resize_to_max_side_len=args.resize_to_max_side_len,
        make_contact_sheet=args.make_contact_sheet,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
