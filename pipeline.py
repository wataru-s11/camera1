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


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    print(f"[WARN] Invalid boolean for {name}: {raw!r}. Using default {default}.")
    return default


def run_cycle(
    model: "YOLO" | None,
    review_manager: ReviewManager | None,
    *,
    announce_sleep: bool = True,
    defer_review: bool = False,
) -> None:
    cycle_started = datetime.now()
    print(f"[INFO] Pipeline cycle started at {cycle_started:%Y-%m-%d %H:%M:%S}")

    ensure_save_directories(CAMERA_CONFIGS, cycle_started)

    if defer_review:
        print(
            "[INFO] レビューを後回しにしています。必要になったら master.py --mode review で分類してください。"
        )

    capture_results: list[tuple[str, Path, Path]] = []

    review_folders: set[Path] = set()

    for config in CAMERA_CONFIGS:
        name = config.label()
        try:
            result = capture_and_transfer(config, cycle_time=cycle_started)
        except Exception as exc:
            print(f"[ERROR] {name}: {exc}")
            continue

        image_path = Path(result.image_path)
        lux_path = Path(result.lux_path)
        capture_results.append((name, image_path, lux_path))

        review_folders.add(image_path.parent)

        if defer_review:
            print(
                "[INFO] レビュー保留: %s / 後で review モードで分類してください。" % image_path
            )
            print(f"[INFO] 参考: 照度ログ → {lux_path}")

    if defer_review:
        if review_folders:
            print("[INFO] 後からレビューするフォルダの候補:")
            for folder in sorted(review_folders):
                print(f"        - {folder}")
            print(
                "[INFO] 例: python master.py --mode review --input "
                "\"<上記のフォルダ>\""
            )
        return

    if model is None or review_manager is None:
        raise RuntimeError("defer_review=False の場合、model と review_manager が必要です。")

    for name, image_path, lux_path in capture_results:
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

    if announce_sleep:
        print(f"[INFO] Cycle finished. Sleeping for {CYCLE_INTERVAL_SECONDS} seconds.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    camera_id = os.environ.get("CAMERA_ID")
    defer_review = _env_flag("PIPELINE_DEFER_REVIEW", True)

    model = None
    if not defer_review:
        model, _ = load_yolo_model(camera_id)

    review_manager: ReviewManager | None = None
    if not defer_review:
        review_manager = ReviewManager(OUTPUT_ROOT)

    try:
        while True:
            try:
                run_cycle(
                    model,
                    review_manager,
                    defer_review=defer_review,
                )
            except ReviewAborted:
                print("[INFO] オペレータがレビューを中断しました。")
                break

            time.sleep(CYCLE_INTERVAL_SECONDS)
    finally:
        if review_manager is not None:
            review_manager.close()


if __name__ == "__main__":
    main()
