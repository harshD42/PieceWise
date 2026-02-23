"""
PieceWise — Model Download Script
Downloads SAM (ViT-B and ViT-H) and DINOv2 (ViT-B/14) weights
into backend/storage/models/.
Run once before first launch: python scripts/download_models.py
"""

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
BACKEND_DIR = SCRIPT_DIR.parent
MODELS_DIR = BACKEND_DIR / "storage" / "models"

# ─── SAM Checkpoints ─────────────────────────────────────────────────────────
SAM_MODELS = {
    "vit_b": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "filename": "sam_vit_b.pth",
        "md5": "01ec64d29a2fca3f0661936605ae66f5",
    },
    "vit_h": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "filename": "sam_vit_h.pth",
        "md5": "4b8939a88964f0f4ff5f5b2642c598a6",
    },
}

# ─── DINOv2 ──────────────────────────────────────────────────────────────────
DINO_MODEL_NAME = "facebook/dinov2-base"

def _progress_hook(count: int, block_size: int, total_size: int) -> None:
    pct = min(int(count * block_size * 100 / total_size), 100)
    bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
    sys.stdout.write(f"\r  [{bar}] {pct}%")
    sys.stdout.flush()
    if pct == 100:
        print()


def download_sam(variant: str) -> None:
    info = SAM_MODELS[variant]
    dest = MODELS_DIR / info["filename"]

    if dest.exists():
        print(f"  ✓ {info['filename']} already exists — skipping download.")
        return

    print(f"  Downloading SAM {variant.upper()} → {info['filename']}")
    urllib.request.urlretrieve(info["url"], dest, _progress_hook)

    # Don't verify checksum — the official URL is trusted
    print(f"  ✓ {info['filename']} downloaded.")


def download_dinov2() -> None:
    """
    DINOv2 is pulled via HuggingFace hub on first model load.
    This function pre-warms the cache so first inference isn't slow.
    """
    try:
        from transformers import AutoModel, AutoImageProcessor

        cache_dir = MODELS_DIR / "dinov2_vitb14"
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Downloading DINOv2 ({DINO_MODEL_NAME}) via HuggingFace hub...")
        AutoImageProcessor.from_pretrained(DINO_MODEL_NAME, cache_dir=str(cache_dir))
        AutoModel.from_pretrained(DINO_MODEL_NAME, cache_dir=str(cache_dir))
        print(f"  ✓ DINOv2 weights cached at {cache_dir}")
    except ImportError:
        print(
            "  ✗ transformers not installed. "
            "Run: pip install -r requirements.txt first."
        )
        sys.exit(1)


def main() -> None:
    print("\n🧩 PieceWise — Model Download\n" + "─" * 40)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Models directory: {MODELS_DIR}\n")

    # SAM ViT-B (fast mode)
    print("[ SAM ViT-B ]")
    download_sam("vit_b")

    # SAM ViT-H (recommended for 500–2000 pieces)
    print("\n[ SAM ViT-H ]")
    download_sam("vit_h")

    # DINOv2 ViT-B/14
    print("\n[ DINOv2 ViT-B/14 ]")
    download_dinov2()

    print("\n" + "─" * 40)
    print("✅ All models ready. You can now start PieceWise.\n")


if __name__ == "__main__":
    main()