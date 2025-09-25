"""End-to-end automation: capture images and run the face review pipeline."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path

from every1h_photo_lux import (
    CAMERA_CONFIGS,
    capture_and_transfer,
    ensure_save_directories,
)
from processing_pipeline import (
    CAMERA_CROP_CONFIGS,
    DEFAULT_CROP,
    ReviewAborted,
    ReviewManager,
    load_yolo_model,
    process_image,
)

DEFAULT_OUTPUT_ROOT = Path(r"Z:\Raspi_face\cropped_face")


def _env_interval(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default

    try:
        seconds = int(value)
    except ValueError:
        print(f"[WARN] Invalid {name} value: {value!r}. Using default {default} seconds.")
        return default

    if seconds <= 0:
        print(f"[WARN] {name} must be positive. Using default {default} seconds.")
        return default

    return seconds


CYCLE_INTERVAL_SECONDS = _env_interval("PIPELINE_INTERVAL_SECONDS", 1800)
OUTPUT_ROOT = Path(os.environ.get("PIPELINE_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT)))


def run_cycle(model: "YOLO", review_manager: ReviewManager) -> None:
    cycle_started = datetime.now()
    print(f"[INFO] Pipeline cycle started at {cycle_started:%Y-%m-%d %H:%M:%S}")

    ensure_save_directories(CAMERA_CONFIGS, cycle_started)

    for config in CAMERA_CONFIGS:
        name = config.label()
        try:
            result = capture_and_transfer(config, cycle_time=cycle_started)
        except Exception as exc:
            print(f"[ERROR] {name}: {exc}")
            continue

        image_path = Path(result.image_path)
        lux_path = Path(result.lux_path)
        try:
            process_image(
                image_path,
                model=model,
                review_manager=review_manager,
                lux_path=lux_path,
                camera_crop_configs=CAMERA_CROP_CONFIGS,
                default_crop=DEFAULT_CROP,
            )
        except ReviewAborted:
            raise
        except Exception as exc:
            print(f"[ERROR] {name}: 処理中にエラーが発生しました: {exc}")

    print(f"[INFO] Cycle finished. Sleeping for {CYCLE_INTERVAL_SECONDS} seconds.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    camera_id = os.environ.get("CAMERA_ID")
    model, _ = load_yolo_model(camera_id)
    review_manager = ReviewManager(OUTPUT_ROOT)

    try:
        while True:
            try:
                run_cycle(model, review_manager)
            except ReviewAborted:
                print("[INFO] オペレータがレビューを中断しました。")
                break

            time.sleep(CYCLE_INTERVAL_SECONDS)
    finally:
        review_manager.close()


if __name__ == "__main__":
    main()
