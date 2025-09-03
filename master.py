import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# モデル
model = YOLO(r"C:\Users\sakai\OneDrive\Desktop\Raspi5\face_annotation\runs\detect\train\weights\best.pt")

input_folder = r"C:\Users\sakai\OneDrive\Desktop\Raspi5\pi-vital5\20250831"
output_folder = r"C:\Users\sakai\OneDrive\Desktop\Raspi5\cropped_face_corrected"
os.makedirs(output_folder, exist_ok=True)

x, y, w, h = 475, 43, 2728, 2383  # 全体クロップ範囲

image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]

for fname in image_files:
    img_path = os.path.join(input_folder, fname)
    base_name = fname.replace("image_", "").replace(".jpg", "")
    lux_path = os.path.join(input_folder, f"lux_{base_name}.txt")

    img = cv2.imread(img_path)
    if img is None:
        print(f"{img_path} 読み込み失敗")
        continue

    cropped = img[y:y+h, x:x+w]
    # 顔検出はPIL形式推奨（ultralytics YOLO推論用）
    pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    results = model(pil_img)

    for i, r in enumerate(results):
        for j, box in enumerate(r.boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            # cv2 crop用にxyxy→
            face = cropped[y1:y2, x1:x2]
            # 照度補正
            if os.path.exists(lux_path):
                with open(lux_path, "r") as f:
                    lux = f.read().strip()
                try:
                    lux_value = float(lux)
                    if lux_value < 30:
                        gamma = 1.5
                    elif lux_value < 80:
                        gamma = 1.2
                    elif lux_value > 120:
                        gamma = 0.9
                    else:
                        gamma = 1.0
                    invGamma = 1.0 / gamma
                    table = np.array([((k / 255.0) ** invGamma) * 255 for k in range(256)], dtype="uint8")
                    face = cv2.LUT(face, table)
                except Exception as e:
                    print(f"照度補正エラー: {e}")
            else:
                print(f"{base_name}: lux file not found, no correction.")

            save_path = os.path.join(output_folder, f"{base_name}_face{i}_{j}.jpg")
            cv2.imwrite(save_path, face)
            print(f"保存完了: {save_path}")

print("全処理終了")
