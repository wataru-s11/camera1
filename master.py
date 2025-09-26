"""Standalone runner for the interactive face review pipeline."""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime
from pathlib import Path

from pipeline import CYCLE_INTERVAL_SECONDS, run_cycle
from processing_pipeline import (
    CAMERA_CROP_CONFIGS,
    DEFAULT_CROP,
    ReviewAborted,
    ReviewManager,
    load_yolo_model,
    process_folder,
)

# 既定の入出力パス
_STATIC_INPUT_ROOT_CANDIDATES: tuple[Path, ...] = (
    Path(r"Z:\\Raspi_face\\pi-vital2"),
    Path(r"C:\\Users\\sakai\\OneDrive\\Desktop\\Raspi5\\pi-vital2"),
)
DEFAULT_OUTPUT_FOLDER = Path(r"Z:\Raspi_face\cropped_face")


def _default_input_root_candidates() -> tuple[Path, ...]:
    """Generate candidate folders to search for processed images."""

    dynamic_candidates = []
    repo_root = Path(__file__).resolve().parent

    for relative in ("pi-vital2", Path("data") / "pi-vital2"):
        candidate = repo_root / relative
        dynamic_candidates.append(candidate)

    # Remove duplicates while preserving order
    seen: set[Path] = set()
    ordered_candidates: list[Path] = []
    for path in (*_STATIC_INPUT_ROOT_CANDIDATES, *dynamic_candidates):
        if path in seen:
            continue
        seen.add(path)
        ordered_candidates.append(path)

    return tuple(ordered_candidates)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default

    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    logging.getLogger(__name__).warning(
        "Invalid boolean for %s: %r. Using default %s.", name, raw, default
    )
    return default


def _resolve_output_folder(override: Path | None = None) -> Path:
    if override is not None:
        return Path(override)
    return Path(os.environ.get("REVIEW_OUTPUT_ROOT", str(DEFAULT_OUTPUT_FOLDER)))


def _resolve_input_folder(
    override: Path | None = None,
    override_root: Path | None = None,
) -> Path:
    if override is not None:
        override_path = Path(override)
        if not override_path.exists():
            raise FileNotFoundError(f"指定された入力フォルダが見つかりません: {override_path}")
        return override_path

    env_input = os.environ.get("REVIEW_INPUT_FOLDER")
    if env_input:
        return Path(env_input)

    logger = logging.getLogger(__name__)

    roots: tuple[Path, ...]
    if override_root is not None:
        roots = (Path(override_root),)
    else:
        env_root = os.environ.get("REVIEW_INPUT_ROOT")
        if env_root:
            roots = (Path(env_root),)
        else:
            roots = _default_input_root_candidates()

    default_roots = roots
    for default_root in default_roots:
        if not default_root.exists():
            continue

        today_suffix = datetime.now().strftime("%Y%m%d_processed")
        today_candidate = default_root / today_suffix
        if today_candidate.is_dir():
            return today_candidate

        processed_dirs = sorted(
            (path for path in default_root.glob("*_processed") if path.is_dir()),
            key=lambda path: path.name,
            reverse=True,
        )
        if processed_dirs:
            logger.info("最新の日付フォルダを使用します: %s", processed_dirs[0])
            return processed_dirs[0]

        dated_dirs = sorted(
            (
                path
                for path in default_root.iterdir()
                if path.is_dir() and path.name[:8].isdigit()
            ),
            key=lambda path: path.name,
            reverse=True,
        )
        if dated_dirs:
            logger.warning(
                "*_processed フォルダが見つからないため %s を使用します", dated_dirs[0]
            )
            return dated_dirs[0]

    candidates = "\n".join(str(path) for path in default_roots)
    raise FileNotFoundError(
        "入力フォルダが見つかりません。REVIEW_INPUT_FOLDER を設定するか、以下の候補のいずれかに日付フォルダ (例: 20250101_processed) を用意してください:\n"
        f"{candidates}"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("pipeline", "review"),
        default="pipeline",
        help=(
            "pipeline: 全自動で撮影からレビューまで実行します / "
            "review: 既存の処理済みフォルダのみレビューします"
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="レビュー対象となる処理済み画像フォルダのパス。",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        help=(
            "処理済み画像フォルダが格納されたルートパス。"
            "最新の日付フォルダを自動的に選択します。"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="レビュー結果を書き出すフォルダのパス。",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="pipeline モードで 1 サイクルのみ実行します。",
    )
    parser.add_argument(
        "--defer-review",
        action="store_true",
        help=(
            "pipeline モードでレビューを後回しにします。撮影とファイル転送のみを行い、"
            "分類は review モードで実施してください。"
        ),
    )
    args = parser.parse_args()

    output_folder = _resolve_output_folder(args.output)

    camera_id = os.environ.get("CAMERA_ID")
    defer_review = args.defer_review
    if args.mode == "pipeline" and not defer_review:
        defer_review = _env_flag("PIPELINE_DEFER_REVIEW", False)
        if defer_review:
            logging.getLogger(__name__).info(
                "PIPELINE_DEFER_REVIEW=1 → レビューを後回しにします"
            )

    requires_model = not (args.mode == "pipeline" and defer_review)
    model = None
    if requires_model:
        model, _ = load_yolo_model(camera_id)

    review_manager: ReviewManager | None = None
    review_aborted = False

    if args.mode == "pipeline":

        if not defer_review:
            review_manager = ReviewManager(output_folder)

        try:
            while True:
                try:
                    run_cycle(
                        model,
                        review_manager,
                        announce_sleep=not args.once,
                        defer_review=defer_review,
                    )
                except ReviewAborted:
                    review_aborted = True
                    break

                if args.once:
                    break

                time.sleep(CYCLE_INTERVAL_SECONDS)
        finally:
            if review_manager is not None:
                review_manager.close()

        if review_aborted:
            print("[INFO] オペレータがレビューを中断しました。")
        return

    review_manager = ReviewManager(output_folder)

    input_folder = _resolve_input_folder(args.input, args.input_root)

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
