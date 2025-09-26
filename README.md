# camera1

PICUのカメラを開発

## master.py の使い方

### 入力フォルダの指定

`master.py` はレビュー対象の画像を格納したフォルダを自動的に探します。標準では以下の順で探索します。

1. `--input` で直接指定したフォルダ
2. 環境変数 `REVIEW_INPUT_FOLDER`
3. `--input-root` または環境変数 `REVIEW_INPUT_ROOT` で指定したルート配下にある最新の日付フォルダ
4. リポジトリに同梱されている既定候補 (`pi-vital2/` など)

`--input-root`/`REVIEW_INPUT_ROOT` を利用すると、毎日生成される `20250101_processed` のようなフォルダ群が配置されているルートパスを一度指定するだけで、最新のフォルダが自動的に選択されます。

### 実行例

```powershell
# フォルダを直接指定する場合
python master.py --input "Z:\Raspi_face\pi-vital2\20250101_processed"

# ルートを指定して最新のフォルダを自動選択する場合
python master.py --input-root "Z:\Raspi_face\pi-vital2"

# 環境変数を利用する場合 (PowerShell)
$env:REVIEW_INPUT_ROOT = "Z:\Raspi_face\pi-vital2"
python master.py
```

いずれの方法でもフォルダが見つからない場合は、エラーメッセージに表示される候補パスのいずれかに日付フォルダを用意してください。

### パイプラインモードでレビューを後回しにする

撮影を定期的に続けつつ、レビューは後からまとめて行いたい場合は `--defer-review` を利用します。

```powershell
python master.py --mode pipeline --defer-review
```
