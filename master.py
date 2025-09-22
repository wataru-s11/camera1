import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# OneDrive環境変数からユーザごとのパスを自動取得
od = os.environ.get("OneDrive")
if od:
    detect_root = Path(od) / "Desktop" / "Raspi5" / "face_annotation" / "runs" / "detect"
else:
    # fallback: sakai ユーザ直書き
    detect_root = Path.home() / "OneDrive" / "Desktop" / "Raspi5" / "face_annotation" / "runs" / "detect"

if not detect_root.exists():
    raise FileNotFoundError(
        f"detect ルートが見つかりません: {detect_root}\n"
        "OneDriveが同期中/オフライン(☁)なら右クリック→『このデバイスに常に保持する』でローカル化してください。"
    )

cands = sorted(
    detect_root.glob("**/weights/best.pt"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
if not cands:
    raise FileNotFoundError(
        f"best.pt が見つかりません: {detect_root}\n"
        "学習済みが別PC/別フォルダの可能性。場所を確認するか、学習を実行してください。"
    )

model_path = str(cands[0])
print(f"[INFO] Using model: {model_path}")
model = YOLO(model_path)

# ========= 入出力 =========
input_folder = Path(r"C:\Users\sakai\OneDrive\Desktop\Raspi5\pi-vital2\20250728_processed")
output_folder = Path(r"Z:\Raspi_face\cropped_face")
output_folder.mkdir(parents=True, exist_ok=True)

if not input_folder.exists():
    raise FileNotFoundError(f"入力フォルダが見つかりません: {input_folder}\nZ: ドライブ割当やNAS接続を確認してください。")

# 全体クロップ範囲（x, y, w, h）
x, y, w, h = 1098, 50, 1843, 1789

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

    # 全体クロップ（はみ出しガード）
    full = safe_crop(img, x, y, x + w, y + h)
    if full is None:
        print(f"[WARN] 全体クロップ範囲が不正: {fname}")
        continue

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

    # 照度ファイル→ガンマ決定
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

    # 各検出ボックスでクロップ
    boxes_xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy() if getattr(r.boxes, "conf", None) is not None else [None]*len(boxes_xyxy)

    for idx, (box, conf) in enumerate(zip(boxes_xyxy, confs)):
        x1, y1, x2, y2 = map(int, box)
        face = safe_crop(full, x1, y1, x2, y2)
        if face is None:
            continue

        # 照度に応じたガンマ補正
        face = gamma_correct(face, gamma)

        # 保存（信頼度も付与）
        conf_tag = f"_{conf:.2f}" if conf is not None else ""
        save_path = output_folder / f"{base_name}_face{idx}{conf_tag}.jpg"
        cv2.imwrite(str(save_path), face)
        print(f"[SAVE] {save_path}")

print("=== 全処理終了 ===")
