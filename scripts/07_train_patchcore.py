"""Fit Anomalib's PatchCore on tiles of OK towels only (no defect labels needed).

PatchCore stores a memory bank of patch features from normal fabric; at
inference, a patch far from every stored feature is anomalous. The model
is fitted with Anomalib's ``PatchcoreModel`` in a manual loop so that the
feature bank is accumulated in CPU RAM (the full bank does not fit in a
6 GB GPU) before coreset sub-sampling.

    python scripts/07_train_patchcore.py --max-tiles 1500 --coreset 0.01
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from galia import config as C  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def main():
    from anomalib.models.components.sampling import KCenterGreedy
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="wide_resnet50_2")
    ap.add_argument("--max-tiles", type=int, default=1500)
    ap.add_argument("--coreset", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    random.seed(C.SEED)
    torch.manual_seed(C.SEED)

    files = sorted((C.PATCHCORE_TILES_DIR / "train" / "good").glob("*.png"))
    files = random.sample(files, min(args.max_tiles, len(files)))
    layers = ["layer2", "layer3"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchcoreModel(backbone=args.backbone, layers=layers, pre_trained=True).to(device)
    model.train()
    model.feature_extractor.eval()  # keep frozen BatchNorm statistics

    t0 = time.time()
    feats = []
    with torch.no_grad():
        for i in range(0, len(files), args.batch):
            x = torch.from_numpy(np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files[i:i + args.batch]]))
            x = ((x.permute(0, 3, 1, 2).float() / 255 - MEAN) / STD).to(device)
            emb = model(x)
            model.embedding_store.clear()
            feats.append(emb.half().cpu())  # fp16 halves the RAM footprint
    bank = torch.cat(feats).float()
    print(f"{len(files)} tiles -> {bank.shape[0]:,} patch features ({bank.shape[1]} dims) in {time.time() - t0:.0f}s",
          flush=True)

    t0 = time.time()
    # Same algorithm as anomalib's KCenterGreedy.select_coreset_idxs, but the
    # random projection is computed on CPU and the greedy loop runs on the GPU
    # (the projected features fit in GPU memory, the full bank does not).
    sampler = KCenterGreedy(embedding=bank, sampling_ratio=args.coreset)
    sampler.model.fit(bank)
    proj = sampler.model.transform(bank).to(device)
    n = proj.shape[0]
    idx = torch.randint(high=n, size=(1,), device=device).squeeze()
    selected = [idx]
    min_d = None
    for _ in range(sampler.coreset_size - 1):
        d = torch.linalg.norm(proj - proj[idx], dim=1)
        min_d = d if min_d is None else torch.minimum(min_d, d)
        idx = torch.argmax(min_d)
        min_d[idx] = 0.0
        selected.append(idx)
    coreset = bank[torch.stack(selected).cpu()]
    del proj
    print(f"coreset {coreset.shape[0]:,} features in {time.time() - t0:.0f}s", flush=True)

    C.MODELS_DIR.mkdir(exist_ok=True)
    out = C.MODELS_DIR / "patchcore_wrn50.pt"
    torch.save({"backbone": args.backbone, "layers": layers, "num_neighbors": 9,
                "memory_bank": coreset.contiguous(), "n_tiles": len(files)}, out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
