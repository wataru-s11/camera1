"""Shared utilities for camera capture post-processing and review."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# ========= レビュー設定 =========
REVIEW_WINDOW_NAME = "Face Review"
REVIEW_CATEGORIES: tuple[str, ...] = ("normal", "cyanosis", "controversial", "delete")
REVIEW_KEY_BINDINGS: Mapping[str, str] = {
    "n": "normal",
    "c": "cyanosis",
    "v": "controversial",
    "d": "delete",
}
REVIEW_QUIT_KEYS: set[str] = {"q"}

DERIVED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
INPUT_DERIVED_KEYWORDS = ("full", "stage1", "firststage", "crop", "gamma")
OUTPUT_DERIVED_KEYWORDS = ("full", "stage1", "firststage", "gamma")


class ReviewAborted(Exception):
    """Raised when an operator chooses to abort the interactive review."""


def _default_display_max_dimension() -> int | None:
    raw_value = os.environ.get("FACE_REVIEW_MAX_DIM")
    if raw_value is None:
        return 800

    env_value = raw_value.strip().lower()
    if env_value in {"", "none", "auto"}:
        return None

    try:
        parsed = int(env_value)
    except ValueError:
        logger.warning(
            "Invalid FACE_REVIEW_MAX_DIM '%s'. Falling back to 800.", raw_value
        )
        return 800

    if parsed <= 0:
        logger.warning(
            "FACE_REVIEW_MAX_DIM should be positive. Display scaling disabled."
        )
        return None

    return parsed


@dataclass(slots=True)
class ReviewManager:
    """Stateful helper that manages review destinations and UI interactions."""

    output_root: Path
    categories: Sequence[str] = REVIEW_CATEGORIES
    key_bindings: Mapping[str, str] = field(
        default_factory=lambda: dict(REVIEW_KEY_BINDINGS)
    )
    quit_keys: Sequence[str] = tuple(REVIEW_QUIT_KEYS)
    window_name: str = REVIEW_WINDOW_NAME
    display_max_dimension: int | None = field(
        default_factory=_default_display_max_dimension
    )
      
    _destinations: dict[str, Path] = field(init=False, repr=False)
    _key_bindings: dict[str, str] = field(init=False, repr=False)
    _quit_keys: set[str] = field(init=False, repr=False)
    _window_initialized: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.output_root = Path(self.output_root)
        self._destinations: dict[str, Path] = {
            category: self.output_root / category for category in self.categories
        }
        for destination in self._destinations.values():
            destination.mkdir(parents=True, exist_ok=True)

        # Normalize keys for comparison
        self._key_bindings = {key.lower(): value for key, value in self.key_bindings.items()}
        self._quit_keys = {key.lower() for key in self.quit_keys}

    # -- convenience -----------------------------------------------------
    def bindings_description(self) -> str:
        return " / ".join(f"{key}:{label}" for key, label in self._key_bindings.items())

    def close(self) -> None:
        """Release any UI resources used for the review window."""

        if self._window_initialized:
            cv2.destroyWindow(self.window_name)
            cv2.waitKey(1)
            self._window_initialized = False

    # -- internal helpers -------------------------------------------------
    def _ensure_window(self) -> None:
        if not self._window_initialized:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self._window_initialized = True

    def _instruction_text(self) -> str:
        return "  ".join(f"[{key}] {label}" for key, label in self._key_bindings.items()) + "  [q] quit"

    def _prepare_display_image(self, face: np.ndarray) -> tuple[np.ndarray, float]:
        display_img = face.copy()
        scale = 1.0

        if self.display_max_dimension is not None:
            height, width = display_img.shape[:2]
            max_dim = max(height, width)
            if max_dim > self.display_max_dimension:
                scale = self.display_max_dimension / max_dim
                new_size = (
                    max(1, int(width * scale)),
                    max(1, int(height * scale)),
                )
                interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                display_img = cv2.resize(display_img, new_size, interpolation=interpolation)

        return display_img, scale

    # -- public API -------------------------------------------------------
    def prompt(self, face: np.ndarray, base_name: str, idx: int, conf: float | None) -> str:
        """Show a face crop and wait for operator classification."""

        self._ensure_window()
        display_img, scale = self._prepare_display_image(face)
        while True:
            scaled_for_text = max(scale, 0.5)
            font_scale = max(0.5, 0.6 * scale)
            thickness = max(1, int(2 * scale))
            line_height = int(25 * scaled_for_text)
            y = int(30 * scaled_for_text)
            text_overlay = display_img.copy()
            overlay_lines = [
                f"{base_name} face#{idx}",
                f"confidence: {conf:.2f}" if conf is not None else "confidence: n/a",
                self._instruction_text(),
            ]
            for line in overlay_lines:
                cv2.putText(
                    text_overlay,
                    line,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (0, 255, 0),
                    thickness,
                    cv2.LINE_AA,
                )
                y += line_height

            cv2.imshow(self.window_name, text_overlay)
            cv2.resizeWindow(self.window_name, text_overlay.shape[1], text_overlay.shape[0])
            key = cv2.waitKey(0)
            if key < 0:
                continue

            key = key & 0xFF
            if key == 27:  # ESC
                raise ReviewAborted

            key_char = chr(key).lower()
            if key_char in self._quit_keys:
                raise ReviewAborted
            if key_char in self._key_bindings:
                return self._key_bindings[key_char]

            print(
                f"[INFO] 未対応のキー: '{key_char}'. 指示に従ってください → {self._instruction_text()}"
            )

    def destination_for(self, category: str) -> Path:
        return self._destinations[category]


# ========= ユーティリティ =========

def cleanup_artifacts(paths: Iterable[Path | str]) -> None:
    """Delete generated files while logging the outcome."""

    seen: set[Path] = set()
    for path in paths:
        if path is None:
            continue
        file_path = Path(path)
        if file_path in seen:
            continue
        seen.add(file_path)
        if file_path.exists():
            try:
                file_path.unlink()
                logger.info("Removed file: %s", file_path)
            except Exception as exc:  # pragma: no cover - best effort cleanup
                logger.warning("Failed to remove %s: %s", file_path, exc)


def find_existing_derived_artifacts(
    base_name: str,
    img_path: Path,
    *,
    output_root: Path,
) -> set[Path]:
    """Locate known derivative files for removal when detection fails."""

    derived: set[Path] = set()
    prefixes = {img_path.stem}
    if base_name:
        prefixes.add(base_name)

    search_specs = (
        (img_path.parent, INPUT_DERIVED_KEYWORDS),
        (output_root, OUTPUT_DERIVED_KEYWORDS),
    )

    # Search save folder first, then output root (which may be shared across runs)
    for directory, keywords in search_specs:
        directory_path = Path(directory)
        if not directory_path.exists():
            continue
        for prefix in prefixes:
            for candidate in directory_path.glob(f"{prefix}*"):
                if not candidate.is_file():
                    continue
                if candidate.suffix.lower() not in DERIVED_IMAGE_EXTENSIONS:
                    continue
                stem_lower = candidate.stem.lower()
                if any(keyword in stem_lower for keyword in keywords):
                    derived.add(candidate)

    return derived


def gamma_correct(img_bgr: np.ndarray, gamma: float) -> np.ndarray:
    if gamma == 1.0:
        return img_bgr
    inv = 1.0 / gamma
    table = np.array([((k / 255.0) ** inv) * 255 for k in range(256)], dtype="uint8")
    return cv2.LUT(img_bgr, table)


def safe_crop(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray | None:
    H, W = img.shape[:2]
    x1 = max(0, min(W, x1))
    x2 = max(0, min(W, x2))
    y1 = max(0, min(H, y1))
    y2 = max(0, min(H, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def determine_gamma_from_lux(lux_path: Path, base_name: str) -> float:
    gamma = 1.0
    if lux_path.exists():
        try:
            lux_value = float(lux_path.read_text(encoding="utf-8").strip())
            if lux_value < 30:
                gamma = 1.5
            elif lux_value < 80:
                gamma = 1.2
            elif lux_value > 120:
                gamma = 0.9
        except Exception as exc:
            print(f"[WARN] 照度補正エラー({base_name}): {exc}")
    else:
        print(f"[INFO] {base_name}: lux file not found → 補正なし")
    return gamma


def derive_base_name(img_path: Path) -> str:
    base_name = img_path.name
    base_name = base_name.replace("image_", "")
    base_name = base_name.replace(".jpg", "")
    base_name = base_name.replace(".JPG", "")
    return base_name


# ========= YOLOモデル初期化 =========

def _default_preferred_weight_paths() -> list[Path]:
    return [
        Path(r"Z:\Raspi_face\face_detector\runs\train_face_gpu\weights\best.pt"),
    ]


def load_yolo_model(
    camera_id: str | None = None,
    *,
    preferred_weight_paths: Sequence[Path] | None = None,
) -> tuple[YOLO, str]:
    """Load YOLO model with the same search strategy used in ``master.py``."""

    if preferred_weight_paths is None:
        preferred_weight_paths = _default_preferred_weight_paths()
    else:
        preferred_weight_paths = list(preferred_weight_paths)

    search_paths: list[Path] = list(preferred_weight_paths)
    if camera_id:
        camera_specific_paths = [
            Path(fr"Z:\Raspi_face\face_detector\runs\{camera_id}\weights\best.pt"),
            Path(fr"Z:\Raspi_face\face_detector\runs\train_{camera_id}\weights\best.pt"),
        ]
        for path in camera_specific_paths:
            if path not in search_paths:
                search_paths.append(path)

    model_path: str | None = None
    for weight_path in search_paths:
        if weight_path.exists():
            model_path = str(weight_path)
            print(f"[INFO] Using model from Z: share: {model_path}")
            break

    if model_path is None:
        od = os.environ.get("OneDrive")
        if od:
            detect_root = Path(od) / "Desktop" / "Raspi5" / "face_annotation" / "runs" / "detect"
        else:
            detect_root = (
                Path.home()
                / "OneDrive"
                / "Desktop"
                / "Raspi5"
                / "face_annotation"
                / "runs"
                / "detect"
            )

        preferred_locations_msg = "\n".join(f"    - {p}" for p in search_paths)
        print(
            "[INFO] Z:ドライブ上で学習済みが見つからなかったため、OneDriveフォルダを探索します:\n"
            f"    OneDrive detect root: {detect_root}"
        )

        if not detect_root.exists():
            raise FileNotFoundError(
                "YOLO学習済み重みが見つかりません。\n"
                f"Z: ドライブで確認した場所:\n{preferred_locations_msg}\n"
                f"OneDriveフォールバックも存在しません: {detect_root}\n"
                "新しい best.pt を Z: ドライブ(推奨) または OneDrive の runs/detect 配下に配置してください。"
            )

        cands = sorted(
            detect_root.glob("**/weights/best.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not cands:
            raise FileNotFoundError(
                "YOLO学習済み重みが見つかりません。\n"
                f"Z: ドライブで確認した場所:\n{preferred_locations_msg}\n"
                f"OneDrive配下でも best.pt が見つかりません: {detect_root}\n"
                "新しい best.pt を Z: ドライブ(推奨) または OneDrive の runs/detect 配下に配置してください。"
            )

        model_path = str(cands[0])
        print(f"[INFO] Using fallback model from OneDrive: {model_path}")

    print(f"[INFO] Using model: {model_path}")
    model = YOLO(model_path)
    return model, model_path


# ========= メイン処理 =========

def process_image(
    img_path: Path,
    *,
    model: YOLO,
    review_manager: ReviewManager,
    lux_path: Path | None = None,
    cleanup_on_failure: bool = True,
) -> None:
    """Process a single captured image through gamma correction and YOLO review."""

    img_path = Path(img_path)
    if lux_path is None:
        base_name_guess = derive_base_name(img_path)
        lux_path = img_path.parent / f"lux_{base_name_guess}.txt"
    else:
        lux_path = Path(lux_path)

    base_name = derive_base_name(img_path)
    artifacts_to_cleanup: set[Path] = {img_path}

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] 読み込み失敗: {img_path}")
        return

    full = img

    gamma = determine_gamma_from_lux(lux_path, base_name)
    full = gamma_correct(full, gamma)

    pil_img: Image.Image | None = None
    try:
        pil_img = Image.fromarray(cv2.cvtColor(full, cv2.COLOR_BGR2RGB))
        results = model(pil_img)
    finally:
        if pil_img is not None:
            pil_img.close()

    if not results:
        print(f"[INFO] 推論結果なし: {img_path.name}")
        if cleanup_on_failure:
            artifacts_to_cleanup.update(
                find_existing_derived_artifacts(
                    base_name,
                    img_path,
                    output_root=review_manager.output_root,
                )
            )
            cleanup_artifacts(artifacts_to_cleanup)
        return

    r = results[0]
    if getattr(r, "boxes", None) is None or len(r.boxes) == 0:
        print(f"[INFO] 検出なし: {img_path.name}")
        if cleanup_on_failure:
            artifacts_to_cleanup.update(
                find_existing_derived_artifacts(
                    base_name,
                    img_path,
                    output_root=review_manager.output_root,
                )
            )
            cleanup_artifacts(artifacts_to_cleanup)
        return

    boxes_xyxy = r.boxes.xyxy.cpu().numpy()
    confs = (
        r.boxes.conf.cpu().numpy()
        if getattr(r.boxes, "conf", None) is not None
        else [None] * len(boxes_xyxy)
    )

    for idx, (box, conf) in enumerate(zip(boxes_xyxy, confs)):
        x1, y1, x2, y2 = map(int, box)
        face = safe_crop(full, x1, y1, x2, y2)
        if face is None:
            continue

        conf_value = float(conf) if conf is not None else None
        category = review_manager.prompt(face, base_name, idx, conf_value)
        conf_tag = f"_{conf_value:.2f}" if conf_value is not None else ""
        dest_dir = review_manager.destination_for(category)
        save_path = dest_dir / f"{base_name}_face{idx}{conf_tag}.jpg"
        cv2.imwrite(str(save_path), face)
        print(f"[REVIEW] {category}: {save_path}")


def process_folder(
    input_folder: Path,
    *,
    model: YOLO,
    review_manager: ReviewManager,
    cleanup_on_failure: bool = True,
) -> None:
    """Process all JPG images in ``input_folder`` sequentially."""

    input_folder = Path(input_folder)
    if not input_folder.exists():
        raise FileNotFoundError(
            f"入力フォルダが見つかりません: {input_folder}\nZ: ドライブ割当やNAS接続を確認してください。"
        )

    image_files = sorted(
        p for p in input_folder.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"
    )
    if not image_files:
        print(f"[WARN] JPGが見つかりません: {input_folder}")
        return

    instruction_line = review_manager.bindings_description()
    print(f"[INFO] 分類キー → {instruction_line} / q:quit")

    for img_path in image_files:
        try:
            process_image(
                img_path,
                model=model,
                review_manager=review_manager,
                cleanup_on_failure=cleanup_on_failure,
            )
        except ReviewAborted:
            print("[INFO] オペレータがレビューを中断しました。")
            raise
    else:
        print("=== 全処理終了 ===")

