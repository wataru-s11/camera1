import os
from pathlib import Path
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

print(f"[INFO] Using model: {model_path}")
model = YOLO(model_path)

# ========= 入出力 =========
input_folder = Path(r"C:\Users\sakai\OneDrive\Desktop\Raspi5\pi-vital2\20250728_processed")
output_folder = Path(r"Z:\Raspi_face\cropped_face")
output_folder.mkdir(parents=True, exist_ok=True)

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

# ========= 画像一覧 =========
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".jpg")]
if not image_files:
    print(f"[WARN] JPGが見つかりません: {input_folder}")

# ========= メイン処理 =========
for fname in sorted(image_files):
    img_path = input_folder / fname
    base_name = fname.replace("image_", "").replace(".jpg", "").replace(".JPG", "")
    lux_path = input_folder / f"lux_{base_name}.txt"

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
    pil_img = Image.fromarray(cv2.cvtColor(full, cv2.COLOR_BGR2RGB))
    results = model(pil_img)
    if not results:
        print(f"[INFO] 推論結果なし: {fname}")
        continue

    r = results[0]
    if getattr(r, "boxes", None) is None or len(r.boxes) == 0:
        print(f"[INFO] 検出なし: {fname}")
        continue

    # 各検出ボックスでクロップ
    boxes_xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy() if getattr(r.boxes, "conf", None) is not None else [None]*len(boxes_xyxy)

    for idx, (box, conf) in enumerate(zip(boxes_xyxy, confs)):
        x1, y1, x2, y2 = map(int, box)
        face = safe_crop(full, x1, y1, x2, y2)
        if face is None:
            continue

        # 保存（信頼度も付与）
        conf_tag = f"_{conf:.2f}" if conf is not None else ""
        save_path = output_folder / f"{base_name}_face{idx}{conf_tag}.jpg"
        cv2.imwrite(str(save_path), face)
        print(f"[SAVE] {save_path}")

print("=== 全処理終了 ===")
