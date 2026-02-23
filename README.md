# 🧩 PieceWise

> **AI-powered jigsaw puzzle solver — because every piece has a place.**
>
> *Scatter to solution, one piece at a time.*

PieceWise is a production-grade computer vision system that takes two photos — one of your complete puzzle reference image and one of your scattered puzzle pieces — and returns a fully sequenced, step-by-step assembly guide with visual overlays.

---

## How It Works

PieceWise combines three state-of-the-art AI systems in a hybrid pipeline:

| Stage | Technology | Purpose |
|---|---|---|
| Piece Segmentation | Meta SAM (ViT-H) | Isolate every piece from any background |
| Feature Matching | Meta DINOv2 (ViT-B/14) | Spatial token correlation for precise placement |
| Assembly Sequencing | Rule-based BFS + Graph | Corner → edge → interior ordered guide |

The system also applies classical computer vision techniques — curvature encoding, histogram matching, watershed segmentation, and adjacency consistency refinement — making it robust to real-world conditions like varied lighting, complex backgrounds, and touching or overlapping pieces.

---

## Architecture Overview

```
Input: Reference Image + Scattered Pieces Photo
         │
         ├─── [Parallel] ──────────────────────────────┐
         │    SAM Segmentation                          │ DINOv2 Reference Embedding
         │    + Watershed Refiner                       │ + PCA Reduction
         │    + Curvature Encoding                      │ + GPU Token Cache
         └─────────────────────────────────────────────┘
                         │
               DINOv2 Piece Embedding
                         │
              Coarse-to-Fine Matching
              (CLS filter → Spatial torch.mm)
                         │
               Hungarian Assignment
                         │
              Adjacency Refinement
              (Color histogram + Curvature complement)
                         │
              BFS Assembly Sequencing
                         │
Output: Annotated overlays + Step-by-step cards + solution.json
```

---

## Supported Puzzle Sizes

| Size | Status |
|---|---|
| 100–500 pieces | ✅ Fast mode (SAM ViT-B) |
| 500–1000 pieces | ✅ Standard mode (SAM ViT-H + PCA) |
| 1000–2000 pieces | ✅ High-res mode (full pipeline, GPU required) |

---

## Tech Stack

**Backend**
- Python 3.11
- FastAPI + Uvicorn
- PyTorch (CUDA)
- Meta SAM — instance segmentation
- Meta DINOv2 — spatial feature extraction
- OpenCV — classical CV and contour analysis
- SciPy — Hungarian algorithm
- Redis — production job store

**Frontend**
- React 18 + TypeScript
- Vite
- Zustand
- React Dropzone

**Infrastructure**
- Docker + Docker Compose
- GitHub Actions CI

---

## Getting Started

### Prerequisites
- Python 3.11+
- Node 20+
- CUDA-capable GPU (recommended) or Apple MPS
- Docker (optional)

### 1. Clone the repo
```bash
git clone https://github.com/harshD42/piecewise.git
cd piecewise
```

### 2. Download model weights
```bash
cd backend
python scripts/download_models.py
```

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env as needed
```

### 4. Run backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 5. Run frontend
```bash
cd frontend
npm install
npm run dev
```

### 6. Or run everything with Docker
```bash
docker-compose up --build
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/solve` | Submit reference + pieces images |
| `GET` | `/status/{job_id}` | Poll job progress |
| `GET` | `/assets/{job_id}/{filename}` | Retrieve output files |
| `PATCH` | `/solve/{job_id}/correct` | Human-in-the-loop piece correction |

---

## Project Structure

```
piecewise/
├── backend/
│   ├── app/
│   │   ├── api/          # Routes and middleware
│   │   ├── core/         # Pipeline orchestrator and job store
│   │   ├── modules/      # Segmentation, matching, rendering, etc.
│   │   ├── models/       # Pydantic data contracts
│   │   └── utils/        # Shared helpers
│   ├── scripts/          # Model download, benchmarking
│   ├── tests/            # Unit and integration tests
│   └── storage/          # Uploads, outputs, model weights (gitignored)
└── frontend/
    └── src/
        ├── components/   # Upload, progress, solution viewer
        ├── hooks/        # useSolverJob, useImageUpload
        ├── api/          # Typed API client
        └── store/        # Zustand state
```

---

## Roadmap

- [x] v1: SAM + DINOv2 + BFS sequencer
- [ ] v2: Graph-based piece adjacency inference
- [ ] v2: Learnable MLP matching layer (fine-tuned on synthetic puzzles)
- [ ] v2: Edge curvature-only mode for blank/solid-color puzzles

---

## License

PieceWise is released under the **Harsh Non-Commercial Attribution License (HNCAL) v1.0**.

- ✅ Free for personal, academic, and research use
- ✅ Attribution to Harsh Dwivedi required in all uses and derivative works
- ❌ Commercial use prohibited without a separate written agreement

For commercial licensing inquiries: [harsh.dwivedi42@gmail.com]

By contributing to this project, you agree that your contributions are licensed
under the same terms. See [CONTRIBUTING.md](CONTRIBUTING.md) for full details.