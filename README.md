# cosyvoice-hokkien-setup

CosyVoice3 閩南語音頻合成環境，含種子音頻。

## Server 上一鍵安裝

```bash
git clone https://github.com/lianghsun/cosyvoice-hokkien-setup
cd cosyvoice-hokkien-setup
bash setup.sh
```

## 執行合成

```bash
# 先 clone 文本資料集
git clone https://huggingface.co/datasets/lianghsun/tw-hokkien-seed-text

# 開始合成（自動偵測 GPU 數）
python synthesize_audio.py --src-dir ./tw-hokkien-seed-text
```
