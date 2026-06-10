import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from scripts.export_image_tiles import export_image_tiles


def test_export_image_tiles_saves_global_tiles_contact_sheet_and_metadata(tmp_path):
    image_path = tmp_path / "wide.png"
    output_dir = tmp_path / "tiles"
    Image.new("RGB", (500, 300), color=(120, 80, 40)).save(image_path)

    metadata = export_image_tiles(
        image_path=image_path,
        output_dir=output_dir,
        max_image_size=256,
        split_size=128,
        resize_to_max_side_len=False,
        make_contact_sheet=True,
    )

    assert metadata["original_size"] == [500, 300]
    assert metadata["grid"] == [2, 2]
    assert metadata["num_local_tiles"] == 4
    assert metadata["global_patch"] == "resized_or_global.png"
    assert metadata["local_tiles"] == [
        {"row": 1, "col": 1, "filename": "tile_r1_c1.png"},
        {"row": 1, "col": 2, "filename": "tile_r1_c2.png"},
        {"row": 2, "col": 1, "filename": "tile_r2_c1.png"},
        {"row": 2, "col": 2, "filename": "tile_r2_c2.png"},
    ]
    assert (output_dir / "metadata.json").is_file()
    assert json.loads((output_dir / "metadata.json").read_text(encoding="utf-8")) == metadata
    assert (output_dir / "resized_or_global.png").is_file()
    assert (output_dir / "contact_sheet.png").is_file()

    for item in metadata["local_tiles"]:
        with Image.open(output_dir / item["filename"]) as tile:
            assert tile.size == (128, 128)
            assert tile.mode == "RGB"

    with Image.open(output_dir / "resized_or_global.png") as global_patch:
        assert global_patch.size == (128, 128)
        assert global_patch.mode == "RGB"


def test_export_image_tiles_single_grid_has_no_global_patch_or_contact_sheet(tmp_path):
    image_path = tmp_path / "small.png"
    output_dir = tmp_path / "tiles"
    Image.new("RGB", (100, 100), color=(20, 60, 100)).save(image_path)

    metadata = export_image_tiles(
        image_path=image_path,
        output_dir=output_dir,
        max_image_size=128,
        split_size=128,
        resize_to_max_side_len=False,
        make_contact_sheet=False,
    )

    assert metadata["grid"] == [1, 1]
    assert metadata["num_local_tiles"] == 1
    assert metadata["global_patch"] is None
    assert metadata["contact_sheet"] is None
    assert metadata["local_tiles"] == [{"row": 1, "col": 1, "filename": "tile_r1_c1.png"}]
    assert not (output_dir / "resized_or_global.png").exists()
    assert not (output_dir / "contact_sheet.png").exists()

    with Image.open(output_dir / "tile_r1_c1.png") as tile:
        assert tile.size == (128, 128)
        assert tile.mode == "RGB"


def test_export_image_tiles_cli_help_runs_from_repo_root():
    result = subprocess.run(
        [sys.executable, "scripts/export_image_tiles.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--image" in result.stdout
    assert "--output-dir" in result.stdout
