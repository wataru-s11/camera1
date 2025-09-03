import cv2
import os
import glob
import numpy as np

input_folder = "C:/Users/sakai/OneDrive/Desktop/Raspi5/pi-vital5/20250831"
output_folder = "C:/Users/sakai/OneDrive/Desktop/Raspi5/pi-vital5/20250831_processed"
os.makedirs(output_folder, exist_ok=True)

x, y, w, h = 475, 43, 2728, 2383

image_files = glob.glob(os.path.join(input_folder, "*.jpg"))

for img_path in image_files:
    base_name = os.path.basename(img_path).replace("image_", "").replace(".jpg", "")
    lux_path = os.path.join(input_folder, f"lux_{base_name}.txt")

    img = cv2.imread(img_path)
    if img is None:
        print(f"{img_path} 読み込み失敗")
        continue

    cropped = img[y:y+h, x:x+w]

    if os.path.exists(lux_path):
        with open(lux_path, "r") as f:
            lux = f.read().strip()
        try:
            lux_value = float(lux)
            print(f"{base_name}: lux={lux_value}")

            # 現場感覚の段階的補正
            if lux_value < 30:
                gamma = 1.5
            elif lux_value < 80:
                gamma = 1.2
            elif lux_value > 120:
                gamma = 0.9
            else:
                gamma = 1.0

            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)], dtype="uint8")
            cropped = cv2.LUT(cropped, table)

        except Exception as e:
            print(f"照度補正エラー: {e}")
    else:
        print(f"{base_name}: lux file not found, no correction.")

    save_path = os.path.join(output_folder, f"processed_{base_name}.jpg")
    cv2.imwrite(save_path, cropped)
    print(f"保存完了: {save_path}")

print("全処理終了")
