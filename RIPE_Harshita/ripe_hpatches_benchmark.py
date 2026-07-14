"""
RIPE evaluation on HPatches, matching the MMA benchmark format used for
SIFT / ORB / BRISK / LATCH / SURF.

RIPE (Kunzel et al., 2025) is trained end-to-end with reinforcement learning
on MegaDepth + Tokyo24/7 (~3 days on an A100) -- it is not something to
retrain from scratch. This script instead uses the authors' official
implementation and pretrained checkpoint for inference:
    https://github.com/fraunhoferhhi/RIPE

Setup (run once, needs full internet access -- not just PyPI):
    git clone https://github.com/fraunhoferhhi/RIPE.git
    pip install torch torchvision kornia opencv-python numpy pandas matplotlib
    # weights auto-download on first run from:
    #   https://cvg.hhi.fraunhofer.de/RIPE/ripe_weights.pth
    # (cached to /tmp/ripe_weights.pth)

Expected HPatches layout (standard release):
    hpatches-sequences-release/
        v_XXXX/ i_XXXX/     # 'v_' = viewpoint, 'i_' = illumination
            1.ppm ... 6.ppm
            H_1_2 ... H_1_6

Usage:
    python ripe_hpatches_benchmark.py \
        --hpatches_dir /path/to/hpatches-sequences-release \
        --ripe_repo /path/to/RIPE \
        --out_dir ./ripe_results \
        --top_k 2048 \
        --visualize --vis_sequences 3 --vis_pairs 2 4 6
    # --visualize saves keypoint-detection overlays and RANSAC-filtered
    # match images to ./ripe_results/visualizations/<sequence_name>/
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

MMA_THRESHOLDS_PX = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # report AUC@1/3/5 like the paper


RIPE_WEIGHTS_URL = "https://cvg.hhi.fraunhofer.de/RIPE/ripe_weights.pth"


def load_ripe(ripe_repo: str, device: torch.device, weights_dir: str = None):
    """Import the RIPE package from the cloned repo and load the pretrained model.

    NOTE: RIPE's own vgg_hyper() hardcodes '/tmp/ripe_weights.pth' as the
    default cache path, which does not exist on Windows. To stay
    cross-platform we download/cache the weights ourselves and pass an
    explicit model_path instead of relying on their default.
    """
    sys.path.insert(0, str(Path(ripe_repo).resolve()))
    from ripe import vgg_hyper  # noqa: E402

    cache_dir = Path(weights_dir) if weights_dir else Path.home() / ".cache" / "ripe"
    cache_dir.mkdir(parents=True, exist_ok=True)
    weights_path = cache_dir / "ripe_weights.pth"

    if not weights_path.exists():
        print(f"Downloading RIPE weights to {weights_path} ...")
        torch.hub.download_url_to_file(RIPE_WEIGHTS_URL, str(weights_path))
    else:
        print(f"Using cached weights from {weights_path}")

    model = vgg_hyper(model_path=weights_path).to(device)
    model.eval()
    return model


def load_image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).permute(2, 0, 1).float().to(device) / 255.0
    return t


def resize_keep_aspect(img: torch.Tensor, min_size=512, max_size=768):
    from torchvision.transforms.functional import resize

    h, w = img.shape[-2:]
    aspect = w / h
    if w > h:
        new_w = max(min_size, min(max_size, w))
        new_h = int(new_w / aspect)
    else:
        new_h = max(min_size, min(max_size, h))
        new_w = int(new_h * aspect)
    scale_x, scale_y = w / new_w, h / new_h
    return resize(img, [new_h, new_w], antialias=True), (scale_x, scale_y)


@torch.no_grad()
def extract(model, img_path: Path, device, top_k=2048, threshold=0.5):
    """Returns keypoints in ORIGINAL image pixel coordinates + descriptors."""
    img = load_image_tensor(img_path, device)
    img_r, (sx, sy) = resize_keep_aspect(img)
    kpts, desc, score = model.detectAndCompute(img_r, threshold=threshold, top_k=top_k)
    kpts = kpts.clone()
    kpts[:, 0] *= sx
    kpts[:, 1] *= sy
    return kpts.cpu().numpy(), desc, score.cpu().numpy()


def mutual_nn_match(desc1: torch.Tensor, desc2: torch.Tensor):
    import kornia.feature as KF

    matcher = KF.DescriptorMatcher("mnn")
    dists, idxs = matcher(desc1, desc2)
    return idxs.cpu().numpy()  # (N, 2) -> [idx_in_1, idx_in_2]


def reprojection_errors(kpts1, kpts2, matches, H_gt):
    """For each match, warp kpt1 -> image2 via ground-truth H and measure
    pixel distance to the matched kpt2."""
    if len(matches) == 0:
        return np.array([])
    pts1 = kpts1[matches[:, 0]]
    pts2 = kpts2[matches[:, 1]]
    pts1_h = np.concatenate([pts1, np.ones((len(pts1), 1))], axis=1)
    warped = (H_gt @ pts1_h.T).T
    warped = warped[:, :2] / warped[:, 2:3]
    return np.linalg.norm(warped - pts2, axis=1)


def mma_at_thresholds(errors, thresholds=MMA_THRESHOLDS_PX):
    if len(errors) == 0:
        return {t: 0.0 for t in thresholds}
    return {t: float(np.mean(errors <= t)) for t in thresholds}


def draw_keypoints(img_path: Path, kpts: np.ndarray, scores: np.ndarray, out_path: Path):
    """Draw detected RIPE keypoints on an image, colored/sized by confidence score."""
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

    if len(scores) > 0:
        s_norm = (scores - scores.min()) / (np.ptp(scores) + 1e-8)
    else:
        s_norm = scores

    for (x, y), s in zip(kpts, s_norm):
        radius = 2 + int(4 * s)  # more confident keypoints drawn larger
        color = (int(255 * (1 - s)), int(255 * s), 0)  # blue (low) -> green (high) in BGR
        cv2.circle(img, (int(round(x)), int(round(y))), radius, color, 1, lineType=cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def draw_matches(
    img1_path: Path,
    img2_path: Path,
    kpts1: np.ndarray,
    kpts2: np.ndarray,
    matches: np.ndarray,
    out_path: Path,
    device: torch.device,
    ransac_filter: bool = True,
):
    """Draw mutual-NN matches between two images, optionally highlighting
    RANSAC (fundamental-matrix) inliers in green vs outliers in red."""
    img1 = cv2.imread(str(img1_path), cv2.IMREAD_COLOR)
    img2 = cv2.imread(str(img2_path), cv2.IMREAD_COLOR)

    cv_kp1 = [cv2.KeyPoint(float(x), float(y), 6) for x, y in kpts1]
    cv_kp2 = [cv2.KeyPoint(float(x), float(y), 6) for x, y in kpts2]
    cv_matches = [cv2.DMatch(int(m[0]), int(m[1]), 0) for m in matches]

    mask = None
    if ransac_filter and len(matches) >= 8:
        import kornia.geometry as KG

        pts1 = torch.from_numpy(kpts1[matches[:, 0]]).float().to(device)
        pts2 = torch.from_numpy(kpts2[matches[:, 1]]).float().to(device)
        try:
            _, inlier_mask = KG.ransac.RANSAC(model_type="fundamental", inl_th=1.0)(pts1, pts2)
            mask = inlier_mask.int().cpu().numpy().ravel().tolist()
        except Exception:
            mask = None  # fall back to drawing all matches unfiltered

    result = cv2.drawMatches(
        img1,
        cv_kp1,
        img2,
        cv_kp2,
        cv_matches,
        None,
        matchColor=(0, 255, 0),
        matchesMask=mask,
        singlePointColor=(0, 0, 255),
        flags=cv2.DrawMatchesFlags_DEFAULT,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), result)


def visualize_sequence(model, seq_dir: Path, device, top_k, threshold, out_dir: Path, pair_idxs=(2,)):
    """Save keypoint + match visualizations for one HPatches sequence."""
    vis_dir = out_dir / "visualizations" / seq_dir.name
    ref_path = seq_dir / "1.ppm"
    kpts_ref, desc_ref, score_ref = extract(model, ref_path, device, top_k, threshold)
    draw_keypoints(ref_path, kpts_ref, score_ref, vis_dir / "1_keypoints.png")

    for i in pair_idxs:
        tgt_path = seq_dir / f"{i}.ppm"
        if not tgt_path.exists():
            continue
        kpts_i, desc_i, score_i = extract(model, tgt_path, device, top_k, threshold)
        draw_keypoints(tgt_path, kpts_i, score_i, vis_dir / f"{i}_keypoints.png")

        matches = mutual_nn_match(desc_ref, desc_i)
        draw_matches(ref_path, tgt_path, kpts_ref, kpts_i, matches, vis_dir / f"matches_1-{i}.png", device)

    print(f"  saved visualizations -> {vis_dir}")


def run_sequence(model, seq_dir: Path, device, top_k, threshold):
    ref_path = seq_dir / "1.ppm"
    kpts_ref, desc_ref, _ = extract(model, ref_path, device, top_k, threshold)

    rows = []
    for i in range(2, 7):
        tgt_path = seq_dir / f"{i}.ppm"
        H_path = seq_dir / f"H_1_{i}"
        if not tgt_path.exists() or not H_path.exists():
            continue

        H_gt = np.loadtxt(H_path)
        kpts_i, desc_i, _ = extract(model, tgt_path, device, top_k, threshold)

        matches = mutual_nn_match(desc_ref, desc_i)
        errors = reprojection_errors(kpts_ref, kpts_i, matches, H_gt)

        mma = mma_at_thresholds(errors)
        rows.append(
            {
                "sequence": seq_dir.name,
                "pair": f"1-{i}",
                "type": "viewpoint" if seq_dir.name.startswith("v_") else "illumination",
                "num_kpts_ref": len(kpts_ref),
                "num_kpts_tgt": len(kpts_i),
                "num_matches": len(matches),
                **{f"mma@{t}px": mma[t] for t in MMA_THRESHOLDS_PX},
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hpatches_dir", required=True)
    ap.add_argument("--ripe_repo", required=True, help="path to cloned fraunhoferhhi/RIPE repo")
    ap.add_argument("--out_dir", default="./ripe_results")
    ap.add_argument("--top_k", type=int, default=2048)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument(
        "--weights_dir",
        default=None,
        help="Where to cache ripe_weights.pth (default: ~/.cache/ripe). "
        "Avoids RIPE's hardcoded /tmp path, which breaks on Windows.",
    )
    ap.add_argument(
        "--visualize",
        action="store_true",
        help="Save keypoint-detection and match images for a few sample sequences.",
    )
    ap.add_argument(
        "--vis_sequences",
        type=int,
        default=3,
        help="How many sequences to visualize (first N of each type: viewpoint + illumination).",
    )
    ap.add_argument(
        "--vis_pairs",
        type=int,
        nargs="+",
        default=[2, 4, 6],
        help="Which target frames (2-6) to visualize matches against the reference frame.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_ripe(args.ripe_repo, device, weights_dir=args.weights_dir)

    hpatches_dir = Path(args.hpatches_dir)
    seq_dirs = sorted([d for d in hpatches_dir.iterdir() if d.is_dir()])
    print(f"Found {len(seq_dirs)} HPatches sequences")

    if args.visualize:
        v_seqs = [d for d in seq_dirs if d.name.startswith("v_")][: args.vis_sequences]
        i_seqs = [d for d in seq_dirs if d.name.startswith("i_")][: args.vis_sequences]
        print(f"Visualizing {len(v_seqs) + len(i_seqs)} sample sequences...")
        for seq_dir in v_seqs + i_seqs:
            try:
                visualize_sequence(
                    model, seq_dir, device, args.top_k, args.threshold, out_dir, pair_idxs=args.vis_pairs
                )
            except Exception as e:
                print(f"  visualization skipped for {seq_dir.name}: {e}")

    all_rows = []
    for idx, seq_dir in enumerate(seq_dirs):
        try:
            rows = run_sequence(model, seq_dir, device, args.top_k, args.threshold)
            all_rows.extend(rows)
            print(f"[{idx + 1}/{len(seq_dirs)}] {seq_dir.name}: {len(rows)} pairs")
        except Exception as e:
            print(f"  skipped {seq_dir.name}: {e}")

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "ripe_hpatches_raw.csv", index=False)

    mma_cols = [f"mma@{t}px" for t in MMA_THRESHOLDS_PX]
    summary = df.groupby("type")[mma_cols].mean()
    summary.loc["overall"] = df[mma_cols].mean()
    summary.to_csv(out_dir / "ripe_hpatches_summary.csv")

    print("\n=== RIPE on HPatches ===")
    print(summary[["mma@1px", "mma@3px", "mma@5px"]].round(4))
    print(f"\nSaved: {out_dir / 'ripe_hpatches_raw.csv'}")
    print(f"Saved: {out_dir / 'ripe_hpatches_summary.csv'}")


if __name__ == "__main__":
    main()
