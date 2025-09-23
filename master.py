"""Standalone runner for the interactive face review pipeline."""

from __future__ import annotations

import logging
import os
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
INPUT_FOLDER = Path(r"C:\Users\sakai\OneDrive\Desktop\Raspi5\pi-vital2\20250728_processed")
OUTPUT_FOLDER = Path(r"Z:\Raspi_face\cropped_face")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    camera_id = os.environ.get("CAMERA_ID")
    model, _ = load_yolo_model(camera_id)

    review_manager = ReviewManager(OUTPUT_FOLDER)
    review_aborted = False

    try:
        process_folder(
            INPUT_FOLDER,
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
