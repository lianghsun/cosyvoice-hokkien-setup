# cosyvoice-hokkien-setup

使用 [CosyVoice3](https://github.com/FunAudioLLM/CosyVoice)（Fun-CosyVoice3-0.5B）批次合成台語（閩南語）音頻資料集。

種子說話者來源：
- [lianghsun/tat_open_source](https://huggingface.co/datasets/lianghsun/tat_open_source)（TAT dev/hok，722 筆）
- [OKHand/Clean_Common_Voice_Speech_24.0-TW](https://huggingface.co/datasets/OKHand/Clean_Common_Voice_Speech_24.0-TW)（32,506 筆）

合成結果上傳至：[lianghsun/tw-hokkien-audio](https://huggingface.co/datasets/lianghsun/tw-hokkien-audio)

---

## 環境需求

- Linux（建議 NVIDIA B200 / H100 / A100）
- Python 3.10
- CUDA 12.8+

---

## 安裝

```bash
git clone https://github.com/lianghsun/cosyvoice-hokkien-setup
cd cosyvoice-hokkien-setup
bash setup.sh
```

`setup.sh` 會自動完成：
1. 安裝系統套件（ffmpeg）
2. Clone CosyVoice repo 及 submodules
3. 安裝 Python 依賴（含 Python 3.12 相容性修正）
4. 下載 Fun-CosyVoice3-0.5B 模型權重

---

## 執行合成

```bash
# clone 文本資料集
git clone https://huggingface.co/datasets/lianghsun/tw-hokkien-seed-text

# 設定 HuggingFace token（用於上傳結果）
export HF_TOKEN="hf_..."

# 開始合成（自動偵測 GPU 數）
python synthesize_audio.py --no-vllm --src-dir ./tw-hokkien-seed-text
```

### 常用選項

| 參數 | 預設 | 說明 |
|------|------|------|
| `--n-gpus` | 自動偵測 | 使用幾張 GPU |
| `--workers-per-gpu` | `1` | 每張 GPU 跑幾個 model instance（B200 183GB VRAM 最多約 15）|
| `--upload-every` | `2000` | 每幾筆上傳一次 HuggingFace |
| `--max-disk-gb` | `20` | 本地暫存超過幾 GB 時強制上傳 |
| `--src-dir` | — | 本地 seed-text 資料集目錄 |
| `--no-vllm` | — | 停用 vLLM（建議使用，避免版本衝突）|

### B200 範例（2 GPU × 8 workers = 16 instances）

```bash
python synthesize_audio.py \
    --no-vllm \
    --src-dir ./tw-hokkien-seed-text \
    --workers-per-gpu 8 \
    --upload-every 100
```

---

## 斷點續跑

合成進度存於 `synthesis_checkpoint.db`（SQLite），中斷後直接重跑即可從上次斷點繼續。

清除進度重來：

```bash
rm synthesis_checkpoint.db
```
