#!/usr/bin/env python3
"""
台語音頻批次生成器

一鍵執行（H100×8，自動偵測 GPU 數）：
    python synthesize_audio.py

常用選項：
    python synthesize_audio.py --n-gpus 4
    python synthesize_audio.py --upload-every 2000 --max-disk-gb 30
    python synthesize_audio.py --workers-per-gpu 8   # 每張 GPU 塞 8 個 model instance（B200 183GB VRAM）
"""

import os, sys, io, json, sqlite3, logging, argparse, time, csv

# Monkey-patch torchaudio to silently ignore C++ extension load failures.
# CosyVoice only uses torchaudio.compliance.kaldi which is pure Python/torch
# and does NOT require the C++ extension (_torchaudio.so).
# This is needed when torchaudio ABI doesn't match the installed PyTorch
# (e.g. NVIDIA custom builds like 2.8.0a0+nv25.06).
try:
    import torchaudio._extension.utils as _ta_ext_utils
    _orig_load_lib = _ta_ext_utils._load_lib
    def _safe_load_lib(lib):
        try:
            return _orig_load_lib(lib)
        except Exception:
            return False
    _ta_ext_utils._load_lib = _safe_load_lib
except Exception:
    pass
import numpy as np
import soundfile as sf
import pandas as pd
import torch
import torch.multiprocessing as mp
from pathlib import Path
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
HF_TOKEN      = os.getenv("HF_TOKEN", "")
HF_SRC_REPO   = "lianghsun/tw-hokkien-seed-text"
HF_AUDIO_REPO = "lianghsun/tw-hokkien-audio"
INSTRUCTION   = "You are a helpful assistant. 请用闽南话表達。<|endofprompt|>"

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-gpus",        type=int, default=None,
                   help="GPU 數量（預設自動偵測）")
    p.add_argument("--model-path",    default=os.path.join(SCRIPT_DIR,
                   "apps/synthesis/pretrained_models/Fun-CosyVoice3-0.5B"))
    p.add_argument("--cosyvoice-dir", default=os.path.join(SCRIPT_DIR,
                   "apps/synthesis/repositories/CosyVoice"))
    p.add_argument("--tat-dir",       default=os.path.join(SCRIPT_DIR,
                   "tat_open_source/dev"))
    p.add_argument("--hanzi-json",    default=os.path.join(SCRIPT_DIR,
                   "conversion_results_tailo_gemini.json"))
    p.add_argument("--audio-dir",     default=os.path.join(SCRIPT_DIR, "audio_output"))
    p.add_argument("--db-path",       default=os.path.join(SCRIPT_DIR,
                   "synthesis_checkpoint.db"))
    p.add_argument("--upload-every",  type=int, default=2000,
                   help="每幾筆上傳一次 HF（per GPU）")
    p.add_argument("--max-disk-gb",   type=float, default=20.0,
                   help="本地暫存超過幾 GB 時強制上傳")
    p.add_argument("--src-dir",       default=None,
                   help="本地 seed-text 資料集目錄（git clone 後的路徑）。"
                        "不設則從 HF 串流讀取")
    p.add_argument("--use-vllm",      action="store_true", default=True)
    p.add_argument("--no-vllm",       dest="use_vllm", action="store_false")
    p.add_argument("--hf-seed-repo",  default="OKHand/Clean_Common_Voice_Speech_24.0-TW",
                   help="額外種子音頻 HF dataset repo（設為空字串可停用）")
    p.add_argument("--hf-seed-cache", default=os.path.join(SCRIPT_DIR, "hf_seed_cache"),
                   help="HF 種子音頻本地快取目錄")
    p.add_argument("--workers-per-gpu", type=int, default=1,
                   help="每張 GPU 上同時跑幾個 CosyVoice3 instance（預設 1）。"
                        "B200 183GB VRAM / ~10GB per model ≈ 最多 15。")
    return p.parse_args()


# ── Seed Speakers ──────────────────────────────────────────────────────────────
def load_seed_speakers(tat_dir: str, hanzi_json: str) -> list:
    """
    從 TAT dev set 載入種子說話者清單。
    回傳: [{"speaker_id", "utt_id", "wav_path", "hanzi"}, ...]
    """
    tsv_path = os.path.join(tat_dir, "dev.tsv")
    hok_dir  = os.path.join(tat_dir, "hok")

    # 載入 hanzi 轉換結果（有的話優先用）
    hanzi_map = {}
    if os.path.exists(hanzi_json):
        with open(hanzi_json, encoding="utf-8") as f:
            data = json.load(f)
            hanzi_map = {k: v["translated_hanzi"] for k, v in data.items()}

    seeds = []
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            utt_id     = row["id"]
            speaker_id = row["hok_speaker"]
            wav_path   = os.path.join(tat_dir, row["hok_audio"])
            hanlo_text = row["hok_text_hanlo_tai"]
            hanzi      = hanzi_map.get(utt_id, hanlo_text)

            if os.path.exists(wav_path):
                seeds.append({
                    "speaker_id": speaker_id,
                    "utt_id":     utt_id,
                    "wav_path":   wav_path,
                    "hanzi":      hanzi,
                })

    logger.info("Loaded %d seed speakers from TAT", len(seeds))
    return seeds


def load_hf_seed_speakers(repo_id: str, cache_dir: str, hf_token: str) -> list:
    """
    從 HuggingFace dataset 載入種子說話者，將音頻快取為本地 WAV。
    支援 Common Voice 格式：audio（array+sampling_rate 或 bytes）、sentence、client_id。
    回傳: [{"speaker_id", "utt_id", "wav_path", "hanzi"}, ...]
    """
    from datasets import load_dataset

    os.makedirs(cache_dir, exist_ok=True)
    done_marker = os.path.join(cache_dir, ".done")

    # 若已快取完畢，直接讀目錄
    seeds = []
    if os.path.exists(done_marker):
        for wav_file in sorted(Path(cache_dir).glob("*.wav")):
            meta_file = wav_file.with_suffix(".txt")
            speaker_id = wav_file.stem.split("_")[0]
            hanzi = meta_file.read_text(encoding="utf-8").strip() if meta_file.exists() else ""
            seeds.append({
                "speaker_id": speaker_id,
                "utt_id":     wav_file.stem,
                "wav_path":   str(wav_file),
                "hanzi":      hanzi,
            })
        logger.info("Loaded %d HF seed speakers from cache (%s)", len(seeds), cache_dir)
        return seeds

    logger.info("Downloading HF seed speakers from %s …", repo_id)
    import datasets as hf_datasets
    ds = load_dataset(repo_id, split="train", token=hf_token or None)
    # decode=False: get raw bytes instead of decoded array (avoids torchcodec requirement)
    ds = ds.cast_column("audio", hf_datasets.Audio(decode=False))

    for idx, item in enumerate(ds):
        audio_info  = item.get("audio", {})
        sentence    = item.get("sentence", "")
        client_id   = item.get("client_id", f"spk{idx:06d}")
        # sanitize client_id for filename
        safe_id     = client_id[:16].replace("/", "_").replace(" ", "_")
        utt_id      = f"{safe_id}_{idx:06d}"
        wav_path    = os.path.join(cache_dir, f"{utt_id}.wav")

        if not os.path.exists(wav_path):
            raw_bytes = audio_info.get("bytes") if isinstance(audio_info, dict) else None
            if raw_bytes:
                # decode with soundfile via BytesIO
                arr, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
                sf.write(wav_path, arr, sr)
            else:
                continue  # 無法取得音頻，跳過

        # 儲存文字
        txt_path = os.path.join(cache_dir, f"{utt_id}.txt")
        if not os.path.exists(txt_path):
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(sentence)

        seeds.append({
            "speaker_id": safe_id,
            "utt_id":     utt_id,
            "wav_path":   wav_path,
            "hanzi":      sentence,
        })

        if (idx + 1) % 1000 == 0:
            logger.info("  cached %d / %d HF seeds", idx + 1, len(ds))

    # 標記快取完成
    with open(done_marker, "w") as f:
        f.write(str(len(seeds)))

    logger.info("Loaded %d HF seed speakers from %s", len(seeds), repo_id)
    return seeds


# ── Database ───────────────────────────────────────────────────────────────────
def init_db(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS synthesis (
            text_id    INTEGER,
            gpu_idx    INTEGER,
            status     TEXT DEFAULT 'done',
            hf_batch   TEXT,
            error_msg  TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (text_id, gpu_idx)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS gpu_batches (
            gpu_idx   INTEGER,
            batch_num INTEGER,
            hf_path   TEXT,
            PRIMARY KEY (gpu_idx, batch_num)
        )
    """)
    conn.commit()
    conn.close()


def get_done_ids(db_path: str, gpu_idx: int) -> set:
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT text_id FROM synthesis WHERE gpu_idx=? AND status='done'", (gpu_idx,)
    ).fetchall()
    conn.close()
    return {r[0] for r in rows}


def mark_done(db_path, text_id, gpu_idx, hf_batch=""):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        INSERT OR REPLACE INTO synthesis (text_id, gpu_idx, status, hf_batch)
        VALUES (?, ?, 'done', ?)
    """, (text_id, gpu_idx, hf_batch))
    conn.commit()
    conn.close()


def mark_error(db_path, text_id, gpu_idx, error_msg):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        INSERT OR REPLACE INTO synthesis (text_id, gpu_idx, status, error_msg)
        VALUES (?, ?, 'error', ?)
    """, (text_id, gpu_idx, error_msg))
    conn.commit()
    conn.close()


def next_batch_num(db_path: str, gpu_idx: int) -> int:
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT COALESCE(MAX(batch_num), 0) FROM gpu_batches WHERE gpu_idx=?", (gpu_idx,)
    ).fetchone()
    conn.close()
    return row[0] + 1


def record_batch(db_path, gpu_idx, batch_num, hf_path):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO gpu_batches (gpu_idx, batch_num, hf_path) VALUES (?,?,?)",
        (gpu_idx, batch_num, hf_path)
    )
    conn.commit()
    conn.close()


# ── Audio Utils ────────────────────────────────────────────────────────────────
def tensor_to_wav_bytes(audio_tensor: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio_tensor, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def dir_size_gb(path: str) -> float:
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return total / (1024 ** 3)


# ── HuggingFace Upload ─────────────────────────────────────────────────────────
def upload_batch(samples: list, worker_id: int, batch_num: int,
                 audio_dir: str, db_path: str, hf_api: HfApi):
    """samples: list of dicts with audio bytes + metadata"""
    if not samples:
        return

    from datasets import Dataset, Features
    from datasets import Audio as HFAudio, Value

    sr = samples[0]["sample_rate"]
    features = Features({
        "audio":        HFAudio(sampling_rate=sr),
        "text":         Value("string"),
        "duration":     Value("float32"),
        "sample_rate":  Value("int32"),
        "speaker_id":   Value("string"),
        "seed_audio_id": Value("string"),
        "domain":       Value("string"),
        "subdomain":    Value("string"),
        "scene":        Value("string"),
        "speaker":      Value("string"),
        "emotion":      Value("string"),
        "accent":       Value("string"),
        "seed_text_id": Value("int32"),
    })

    data = {
        "audio":        [{"bytes": s["audio_bytes"], "path": None} for s in samples],
        "text":         [s["text"]         for s in samples],
        "duration":     [s["duration"]     for s in samples],
        "sample_rate":  [s["sample_rate"]  for s in samples],
        "speaker_id":   [s["speaker_id"]   for s in samples],
        "seed_audio_id": [s["seed_audio_id"] for s in samples],
        "domain":       [s["domain"]       for s in samples],
        "subdomain":    [s["subdomain"]    for s in samples],
        "scene":        [s["scene"]        for s in samples],
        "speaker":      [s["speaker"]      for s in samples],
        "emotion":      [s["emotion"]      for s in samples],
        "accent":       [s["accent"]       for s in samples],
        "seed_text_id": [s["seed_text_id"] for s in samples],
    }

    ds       = Dataset.from_dict(data, features=features)
    hf_path  = f"data/worker{worker_id}/batch_{batch_num:06d}.parquet"
    tmp_path = os.path.join(audio_dir, f"tmp_w{worker_id}_b{batch_num}.parquet")

    ds.to_parquet(tmp_path)
    hf_api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo=hf_path,
        repo_id=HF_AUDIO_REPO,
        repo_type="dataset",
    )
    os.remove(tmp_path)
    record_batch(db_path, worker_id, batch_num, hf_path)
    logger.info("Worker%d uploaded batch %06d (%d samples) → %s",
                worker_id, batch_num, len(samples), hf_path)


# ── Worker ─────────────────────────────────────────────────────────────────────
def worker_fn(worker_id: int, total_workers: int, args_dict: dict):
    """
    每個 worker 跑一個 CosyVoice3 instance。
    physical GPU = worker_id // workers_per_gpu
    partition    = global_idx % total_workers == worker_id
    """
    workers_per_gpu = args_dict.get("workers_per_gpu", 1)
    phys_gpu = worker_id // workers_per_gpu

    # ① 隔離 GPU（必須在 import torch/vllm 之前設定）
    os.environ["CUDA_VISIBLE_DEVICES"] = str(phys_gpu)

    log = logging.getLogger(f"W{worker_id}(GPU{phys_gpu})")
    log.info("Starting worker %d on physical GPU %d (%d workers/gpu)",
             worker_id, phys_gpu, workers_per_gpu)

    # ② 設定 CosyVoice 路徑
    cosyvoice_dir = args_dict["cosyvoice_dir"]
    sys.path.insert(0, cosyvoice_dir)
    sys.path.insert(0, os.path.join(cosyvoice_dir, "third_party/Matcha-TTS"))

    # ③ 載入模型（vLLM 需先 register）
    if args_dict["use_vllm"]:
        try:
            from vllm import ModelRegistry
            from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
            ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
            log.info("vLLM ModelRegistry registered")
        except ImportError as e:
            log.warning("vLLM not available (%s), falling back to standard mode", e)
            args_dict["use_vllm"] = False

    from cosyvoice.cli.cosyvoice import AutoModel
    model_kwargs = dict(model_dir=args_dict["model_path"])
    if args_dict["use_vllm"]:
        model_kwargs.update(load_trt=True, load_vllm=True, fp16=False)
    cosyvoice  = AutoModel(**model_kwargs)
    sample_rate = cosyvoice.sample_rate
    log.info("Model loaded (sample_rate=%d)", sample_rate)

    # ④ 種子說話者
    seeds    = args_dict["seed_speakers"]
    n_seeds  = len(seeds)

    # ⑤ 斷點（DB 用 worker_id 當 gpu_idx 欄位）
    db_path  = args_dict["db_path"]
    init_db(db_path)
    done_ids = get_done_ids(db_path, worker_id)
    log.info("Already done: %d", len(done_ids))

    # ⑥ 讀取文本資料集（本地 or HF streaming）
    from datasets import load_dataset
    src_dir = args_dict.get("src_dir")
    if src_dir:
        dataset = load_dataset("parquet",
                               data_files=str(Path(src_dir) / "data" / "*.parquet"),
                               split="train", streaming=True)
    else:
        dataset = load_dataset(HF_SRC_REPO, split="train", streaming=True,
                               token=args_dict["hf_token"])

    # ⑦ HF API
    hf_api     = HfApi(token=args_dict["hf_token"])
    audio_dir  = args_dict["audio_dir"]
    os.makedirs(audio_dir, exist_ok=True)

    local_samples = []
    batch_num = next_batch_num(db_path, worker_id)

    for global_idx, item in enumerate(dataset):
        # 每個 worker 只處理 global_idx % total_workers == worker_id 的項目
        if global_idx % total_workers != worker_id:
            continue

        text_id = int(item.get("id", global_idx))
        if text_id in done_ids:
            continue

        target_text = item.get("text", "")
        if not target_text:
            continue

        # 選種子說話者（按 text_id 循環）
        seed        = seeds[text_id % n_seeds]
        prompt_text = INSTRUCTION + seed["hanzi"]

        try:
            output = next(cosyvoice.inference_zero_shot(
                target_text, prompt_text, seed["wav_path"], stream=False
            ))
            audio_np = output["tts_speech"].squeeze().cpu().numpy()
            duration = round(len(audio_np) / sample_rate, 3)

            local_samples.append({
                "audio_bytes":   tensor_to_wav_bytes(audio_np, sample_rate),
                "text":          target_text,
                "duration":      duration,
                "sample_rate":   sample_rate,
                "speaker_id":    seed["speaker_id"],
                "seed_audio_id": seed["utt_id"],
                "domain":        item.get("domain", ""),
                "subdomain":     item.get("subdomain", ""),
                "scene":         item.get("scene", ""),
                "speaker":       item.get("speaker", ""),
                "emotion":       item.get("emotion", ""),
                "accent":        item.get("accent", ""),
                "seed_text_id":  text_id,
            })
            done_ids.add(text_id)
            mark_done(db_path, text_id, worker_id, f"batch_{batch_num:06d}")

        except Exception as e:
            log.error("text_id=%d: %s", text_id, e)
            mark_error(db_path, text_id, worker_id, str(e))
            continue

        # ⑧ 達到上傳門檻時上傳
        should_upload = (
            len(local_samples) >= args_dict["upload_every"] or
            dir_size_gb(audio_dir) >= args_dict["max_disk_gb"]
        )
        if should_upload:
            upload_batch(local_samples, worker_id, batch_num, audio_dir, db_path, hf_api)
            local_samples = []
            batch_num += 1

    # ⑨ 收尾上傳
    if local_samples:
        upload_batch(local_samples, worker_id, batch_num, audio_dir, db_path, hf_api)

    log.info("Worker %d (GPU %d) finished.", worker_id, phys_gpu)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args            = parse_args()
    n_gpus          = args.n_gpus or torch.cuda.device_count()
    workers_per_gpu = args.workers_per_gpu
    total_workers   = n_gpus * workers_per_gpu

    if n_gpus == 0:
        logger.error("No CUDA GPUs found. Exiting.")
        sys.exit(1)

    logger.info("Starting synthesis on %d GPU(s) × %d worker(s)/GPU = %d total workers",
                n_gpus, workers_per_gpu, total_workers)

    # 確保 HF audio repo 存在
    hf_api = HfApi(token=HF_TOKEN)
    try:
        hf_api.repo_info(repo_id=HF_AUDIO_REPO, repo_type="dataset")
        logger.info("HF audio repo already exists: %s", HF_AUDIO_REPO)
    except Exception:
        hf_api.create_repo(repo_id=HF_AUDIO_REPO, repo_type="dataset", private=False)
        logger.info("Created HF audio repo: %s", HF_AUDIO_REPO)

    # 建立本地暫存目錄與 DB
    os.makedirs(args.audio_dir, exist_ok=True)
    init_db(args.db_path)

    # 載入種子說話者（主進程載入，傳給所有 workers）
    seed_speakers = load_seed_speakers(args.tat_dir, args.hanzi_json)

    # 合併 HF 額外種子
    if args.hf_seed_repo:
        hf_seeds = load_hf_seed_speakers(args.hf_seed_repo, args.hf_seed_cache, HF_TOKEN)
        seed_speakers = seed_speakers + hf_seeds
        logger.info("Total seed speakers (TAT + HF): %d", len(seed_speakers))

    if not seed_speakers:
        logger.error("No seed speakers found. Check --tat-dir and --hanzi-json.")
        sys.exit(1)

    # 打包 worker 參數（需可 pickle）
    args_dict = {
        "model_path":      args.model_path,
        "cosyvoice_dir":   args.cosyvoice_dir,
        "tat_dir":         args.tat_dir,
        "hanzi_json":      args.hanzi_json,
        "audio_dir":       args.audio_dir,
        "db_path":         args.db_path,
        "upload_every":    args.upload_every,
        "max_disk_gb":     args.max_disk_gb,
        "use_vllm":        args.use_vllm,
        "hf_token":        HF_TOKEN,
        "seed_speakers":   seed_speakers,
        "src_dir":         args.src_dir,
        "workers_per_gpu": workers_per_gpu,
    }

    # 啟動 total_workers 個進程（spawn 避免 CUDA fork 問題）
    # worker_id → physical GPU = worker_id // workers_per_gpu
    # 例：2 GPU × 8 workers = 16 workers，worker 0-7 → GPU0，worker 8-15 → GPU1
    mp.set_start_method("spawn", force=True)
    processes = []
    for worker_id in range(total_workers):
        phys_gpu = worker_id // workers_per_gpu
        p = mp.Process(target=worker_fn, args=(worker_id, total_workers, args_dict))
        p.start()
        processes.append(p)
        logger.info("Started worker %d on GPU %d (PID %d)", worker_id, phys_gpu, p.pid)

    for p in processes:
        p.join()

    logger.info("All workers finished.")


if __name__ == "__main__":
    main()
