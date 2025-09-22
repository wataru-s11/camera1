import os
import logging
from pathlib import Path
from collections.abc import Iterable
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Z:ドライブ優先で学習済みを探し、無ければ従来どおりOneDrive配下をglob探索
preferred_weight_paths = [
    Path(r"Z:\Raspi_face\face_detector\runs\train_face_gpu\weights\best.pt"),
]

camera_id = os.environ.get("CAMERA_ID")
if camera_id:
    # カメラ名ごとの専用フォルダも任意で探索 (train_◯◯ や ◯◯ フォルダを想定)
    camera_specific_paths = [
        Path(fr"Z:\Raspi_face\face_detector\runs\{camera_id}\weights\best.pt"),
        Path(fr"Z:\Raspi_face\face_detector\runs\train_{camera_id}\weights\best.pt"),
    ]
    for p in camera_specific_paths:
        if p not in preferred_weight_paths:
            preferred_weight_paths.append(p)

model_path: str | None = None
for weight_path in preferred_weight_paths:
    if weight_path.exists():
        model_path = str(weight_path)
        print(f"[INFO] Using model from Z: share: {model_path}")
        break

if model_path is None:
    # OneDrive環境変数からユーザごとのパスを自動取得
    od = os.environ.get("OneDrive")
    if od:
        detect_root = Path(od) / "Desktop" / "Raspi5" / "face_annotation" / "runs" / "detect"
    else:
        # fallback: sakai ユーザ直書き
        detect_root = Path.home() / "OneDrive" / "Desktop" / "Raspi5" / "face_annotation" / "runs" / "detect"

    preferred_locations_msg = "\n".join(f"    - {p}" for p in preferred_weight_paths)
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

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

print(f"[INFO] Using model: {model_path}")
model = YOLO(model_path)

# ========= 入出力 =========
input_folder = Path(r"C:\Users\sakai\OneDrive\Desktop\Raspi5\pi-vital2\20250728_processed")
output_folder = Path(r"Z:\Raspi_face\cropped_face")
output_folder.mkdir(parents=True, exist_ok=True)

# Review destinations
REVIEW_CATEGORIES = ("normal", "cyanosis", "controversial", "delete")
REVIEW_WINDOW_NAME = "Face Review"
REVIEW_KEY_BINDINGS = {
    "n": "normal",
    "c": "cyanosis",
    "v": "controversial",
    "d": "delete",
}
REVIEW_QUIT_KEYS = {"q"}
review_destinations = {category: output_folder / category for category in REVIEW_CATEGORIES}
for destination in review_destinations.values():
    destination.mkdir(parents=True, exist_ok=True)
_review_window_initialized = False

if not input_folder.exists():
    raise FileNotFoundError(f"入力フォルダが見つかりません: {input_folder}\nZ: ドライブ割当やNAS接続を確認してください。")

# 全体クロップ範囲（x, y, w, h）
DEFAULT_CROP = (1098, 50, 1843, 1789)

# ここにカメラごとのクロップ設定を追加することで、
# 複数カメラでも処理を共通化できる
CAMERA_CROP_CONFIGS: dict[str, tuple[int, int, int, int]] = {
    # pi-vital2/pi2
    "pi-vital2": (1098, 50, 1843, 1789),
    "pi2": (1098, 50, 1843, 1789),
    # pi-vital3/pi3
    "pi-vital3": (748, 1008, 1533, 1454),
    "pi3": (748, 1008, 1533, 1454),
    # pi-vital4/pi4
    "pi-vital4": (1918, 716, 1800, 1641),
    "pi4": (1918, 716, 1800, 1641),
    # pi-vital5/pi5
    "pi-vital5": (475, 43, 2728, 2383),
    "pi5": (475, 43, 2728, 2383),
}

# ========= ユーティリティ =========
DERIVED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
INPUT_DERIVED_KEYWORDS = ("full", "stage1", "firststage", "crop", "gamma")
OUTPUT_DERIVED_KEYWORDS = ("full", "stage1", "firststage", "gamma")


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


def find_existing_derived_artifacts(base_name: str, img_path: Path) -> set[Path]:
    """Locate known derivative files for removal when detection fails."""

    derived: set[Path] = set()
    prefixes = {img_path.stem}
    if base_name:
        prefixes.add(base_name)

    search_specs = (
        (img_path.parent, INPUT_DERIVED_KEYWORDS),
        (output_folder, OUTPUT_DERIVED_KEYWORDS),
    )

    for directory, keywords in search_specs:
        for prefix in prefixes:
            for candidate in directory.glob(f"{prefix}*"):
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


def resolve_crop_rect(path: Path) -> tuple[int, int, int, int]:
    path_lower = str(path).lower()
    for key, rect in CAMERA_CROP_CONFIGS.items():
        if key in path_lower:
            return rect
    return DEFAULT_CROP


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
        except Exception as e:
            print(f"[WARN] 照度補正エラー({base_name}): {e}")
    else:
        print(f"[INFO] {base_name}: lux file not found → 補正なし")
    return gamma


class ReviewAborted(Exception):
    """Raised when an operator chooses to abort the interactive review."""


def _ensure_review_window() -> None:
    global _review_window_initialized
    if not _review_window_initialized:
        cv2.namedWindow(REVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
        _review_window_initialized = True


def _close_review_window() -> None:
    global _review_window_initialized
    if _review_window_initialized:
        cv2.destroyWindow(REVIEW_WINDOW_NAME)
        cv2.waitKey(1)
        _review_window_initialized = False


def prompt_face_review(
    face: np.ndarray,
    base_name: str,
    idx: int,
    conf: float | None,
) -> str:
    """Show a face crop and wait for operator classification."""

    _ensure_review_window()

    instruction_text = "  ".join(
        f"[{key}] {label}" for key, label in REVIEW_KEY_BINDINGS.items()
    ) + "  [q] quit"

    while True:
        display_img = face.copy()
        overlay_lines = [
            f"{base_name} face#{idx}",
            f"confidence: {conf:.2f}" if conf is not None else "confidence: n/a",
            instruction_text,
        ]
        y = 30
        for line in overlay_lines:
            cv2.putText(
                display_img,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            y += 25

        cv2.imshow(REVIEW_WINDOW_NAME, display_img)
        key = cv2.waitKey(0)
        if key < 0:
            continue

        key = key & 0xFF
        if key == 27:  # ESC
            raise ReviewAborted

        key_char = chr(key).lower()
        if key_char in REVIEW_QUIT_KEYS:
            raise ReviewAborted
        if key_char in REVIEW_KEY_BINDINGS:
            return REVIEW_KEY_BINDINGS[key_char]

        print(
            f"[INFO] 未対応のキー: '{key_char}'. 指示に従ってください → {instruction_text}"
        )

# ========= 画像一覧 =========
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".jpg")]
if not image_files:
    print(f"[WARN] JPGが見つかりません: {input_folder}")

# ========= メイン処理 =========
if image_files:
    instruction_line = " / ".join(
        f"{key}:{label}" for key, label in REVIEW_KEY_BINDINGS.items()
    )
    print(f"[INFO] 分類キー → {instruction_line} / q:quit")

review_aborted = False

try:
    for fname in sorted(image_files):
        img_path = input_folder / fname
        base_name = (
            fname.replace("image_", "")
            .replace(".jpg", "")
            .replace(".JPG", "")
        )
        lux_path = input_folder / f"lux_{base_name}.txt"
        artifacts_to_cleanup: set[Path] = {img_path}

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] 読み込み失敗: {img_path}")
            continue

        crop_x, crop_y, crop_w, crop_h = resolve_crop_rect(img_path)

        # 全体クロップ（はみ出しガード）
        full = safe_crop(img, crop_x, crop_y, crop_x + crop_w, crop_y + crop_h)
        if full is None:
            print(f"[WARN] 全体クロップ範囲が不正: {fname}")
            continue

        # 照度ファイル→ガンマ決定し、フレーム全体に一括適用
        gamma = determine_gamma_from_lux(lux_path, base_name)
        full = gamma_correct(full, gamma)

        # YOLO推論はRGBが安定
        pil_img: Image.Image | None = None
        try:
            pil_img = Image.fromarray(cv2.cvtColor(full, cv2.COLOR_BGR2RGB))
            results = model(pil_img)
        finally:
            if pil_img is not None:
                pil_img.close()

        if not results:
            print(f"[INFO] 推論結果なし: {fname}")
            artifacts_to_cleanup.update(find_existing_derived_artifacts(base_name, img_path))
            cleanup_artifacts(artifacts_to_cleanup)
            continue

        r = results[0]
        if getattr(r, "boxes", None) is None or len(r.boxes) == 0:
            print(f"[INFO] 検出なし: {fname}")
            artifacts_to_cleanup.update(find_existing_derived_artifacts(base_name, img_path))
            cleanup_artifacts(artifacts_to_cleanup)
            continue

        # 各検出ボックスでクロップ
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

            try:
                category = prompt_face_review(face, base_name, idx, conf)
            except ReviewAborted:
                review_aborted = True
                break

            conf_tag = f"_{conf:.2f}" if conf is not None else ""
            dest_dir = review_destinations[category]
            save_path = dest_dir / f"{base_name}_face{idx}{conf_tag}.jpg"
            cv2.imwrite(str(save_path), face)
            print(f"[REVIEW] {category}: {save_path}")

        if review_aborted:
            break
    else:
        print("=== 全処理終了 ===")
finally:
    _close_review_window()

if review_aborted:
    print("[INFO] オペレータがレビューを中断しました。")
