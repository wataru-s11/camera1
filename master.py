"""Standalone runner for the interactive face review pipeline."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

from processing_pipeline import (
    CAMERA_CROP_CONFIGS,
    DEFAULT_CROP,
    ReviewAborted,
    ReviewManager,
    load_yolo_model,
    process_folder,
)

# 既定の入出力パス
DEFAULT_INPUT_ROOT = Path(r"C:\Users\sakai\OneDrive\Desktop\Raspi5\pi-vital2")
DEFAULT_OUTPUT_FOLDER = Path(r"Z:\Raspi_face\cropped_face")


def _resolve_output_folder() -> Path:
    return Path(os.environ.get("REVIEW_OUTPUT_ROOT", str(DEFAULT_OUTPUT_FOLDER)))


def _resolve_input_folder() -> Path:
    env_input = os.environ.get("REVIEW_INPUT_FOLDER")
    if env_input:
        return Path(env_input)

    today_suffix = datetime.now().strftime("%Y%m%d_processed")
    today_candidate = DEFAULT_INPUT_ROOT / today_suffix
    if today_candidate.is_dir():
        return today_candidate

    processed_dirs = sorted(
        (path for path in DEFAULT_INPUT_ROOT.glob("*_processed") if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    if processed_dirs:
        logging.getLogger(__name__).info(
            "最新の日付フォルダを使用します: %s", processed_dirs[0]
        )
        return processed_dirs[0]

    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    input_folder = _resolve_input_folder()
    output_folder = _resolve_output_folder()

    camera_id = os.environ.get("CAMERA_ID")
    model, _ = load_yolo_model(camera_id)

    review_manager = ReviewManager(output_folder)
    review_aborted = False

    try:
        process_folder(
            input_folder,
            model=model,
            review_manager=review_manager,
            camera_crop_configs=CAMERA_CROP_CONFIGS,
            default_crop=DEFAULT_CROP,
        )
    except ReviewAborted:
        review_aborted = True
    finally:
        review_manager.close()

    if review_aborted:
        print("[INFO] オペレータがレビューを中断しました。")


if __name__ == "__main__":
    main()
