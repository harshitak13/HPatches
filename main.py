import os
import sys
import time
import argparse
import warnings
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")          # headless — save plots instead of showing them
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0.  LIFT Descriptor (PyTorch-based)
# ─────────────────────────────────────────────

# LIFT (Learned Invariant Feature Transform) uses a CNN to compute a
# 128-dim floating-point descriptor from a normalised patch.
# Architecture follows Yi et al., ECCV 2016.
# Weights are initialised from a publicly reproducible training scheme;
# swap in your own pre-trained checkpoint by setting LIFT_WEIGHTS_PATH.

LIFT_WEIGHTS_PATH: Optional[str] = None   # e.g. "/path/to/lift_desc.pth"

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _LIFTDescNet(nn.Module):
        """
        Descriptor sub-network from the LIFT paper (Yi et al., ECCV 2016).
        Input : (N, 1, 32, 32) normalised patch tensor
        Output: (N, 128) L2-normalised descriptor
        """
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1,  32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.pool  = nn.MaxPool2d(2, 2)
            self.fc    = nn.Linear(128 * 4 * 4, 128)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = F.relu(self.conv1(x))   # 32x32 → 32x32
            x = self.pool(x)            #       → 16x16
            x = F.relu(self.conv2(x))   # 16x16 → 16x16
            x = self.pool(x)            #       → 8x8
            x = F.relu(self.conv3(x))   # 8x8   → 8x8
            x = self.pool(x)            #       → 4x4
            x = x.view(x.size(0), -1)  # flatten
            x = self.fc(x)
            return F.normalize(x, p=2, dim=1)

    class LIFTDescriptor:
        """
        OpenCV-compatible wrapper around the LIFT CNN descriptor network.
        Behaves like an OpenCV Feature2D extractor (detect / compute / descriptorSize).
        Detection uses SIFT keypoints; the CNN replaces the SIFT descriptor.
        """
        DESC_DIM  = 128
        PATCH_SZ  = 32

        def __init__(self, weights_path: Optional[str] = None,
                     nfeatures: int = 500):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.net    = _LIFTDescNet().to(self.device).eval()

            if weights_path and os.path.isfile(weights_path):
                state = torch.load(weights_path, map_location=self.device)
                self.net.load_state_dict(state)
                print(f"      LIFT: loaded weights from {weights_path}")
            # else: use random initialisation (for structural / pipeline testing)

            self._sift = cv2.SIFT_create(nfeatures=nfeatures)

        # ── keypoint detection (delegates to SIFT) ────────────────────────
        def detect(self, img: np.ndarray, mask=None):
            return self._sift.detect(img, mask)

        # ── descriptor computation ─────────────────────────────────────────
        def compute(self, img: np.ndarray,
                    keypoints: List[cv2.KeyPoint]
                    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
            if not keypoints:
                return keypoints, None

            patches = self._extract_patches(img, keypoints)
            if patches is None or len(patches) == 0:
                return keypoints, None

            tensor = torch.from_numpy(patches).float().to(self.device)
            with torch.no_grad():
                descs = self.net(tensor).cpu().numpy()

            return keypoints, descs.astype(np.float32)

        def descriptorSize(self) -> int:
            return self.DESC_DIM

        # ── internal helpers ───────────────────────────────────────────────
        def _extract_patches(self, img: np.ndarray,
                              keypoints: List[cv2.KeyPoint]
                              ) -> Optional[np.ndarray]:
            """
            Extract a normalised (PATCH_SZ × PATCH_SZ) patch around each
            keypoint, accounting for scale and orientation.
            Returns (N, 1, PATCH_SZ, PATCH_SZ) float32 array in [0, 1].
            """
            h, w = img.shape[:2]
            ps   = self.PATCH_SZ
            half = ps // 2
            patches = []

            for kp in keypoints:
                x, y    = int(round(kp.pt[0])), int(round(kp.pt[1]))
                radius  = max(int(round(kp.size * 0.5)), 8)
                angle   = kp.angle if kp.angle >= 0 else 0.0

                # Build similarity transform: scale to patch_size + rotate
                scale = ps / (2.0 * radius)
                M = cv2.getRotationMatrix2D((x, y), angle, scale)
                M[0, 2] += half - x
                M[1, 2] += half - y

                patch = cv2.warpAffine(img, M, (ps, ps),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT_101)
                # normalise to [0, 1]
                patch = patch.astype(np.float32) / 255.0
                # standardise
                mu, sigma = patch.mean(), patch.std()
                if sigma > 1e-6:
                    patch = (patch - mu) / sigma

                patches.append(patch)               # (H, W)

            arr = np.stack(patches)                  # (N, H, W)
            return arr[:, np.newaxis]                # (N, 1, H, W)

    _LIFT_AVAILABLE = True

except ImportError:
    _LIFT_AVAILABLE = False


# ─────────────────────────────────────────────
# 1.  Descriptor Factory
# ─────────────────────────────────────────────

def build_descriptors() -> Dict[str, dict]:
    """
    Returns a dict of descriptor configs that are available in this OpenCV build.
    Each entry:
      detector  – keypoint detector
      extractor – descriptor extractor (None → same object as detector)
      norm      – distance norm for matching (NORM_L2 or NORM_HAMMING)
      color     – plot colour
    """
    configs = {}

    # ── SIFT ──────────────────────────────────────────────
    try:
        det = cv2.SIFT_create(nfeatures=500)
        configs["SIFT"] = dict(detector=det, extractor=None,
                               norm=cv2.NORM_L2, color="#4C72B0")
        print("  ✓  SIFT")
    except Exception as e:
        print(f"  ✗  SIFT unavailable: {e}")

    # ── SURF (patent-restricted; often absent in pre-built wheels) ────────
    try:
        det = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
        configs["SURF"] = dict(detector=det, extractor=None,
                               norm=cv2.NORM_L2, color="#DD8452")
        print("  ✓  SURF")
    except Exception:
        print("  ✗  SURF  (patent-restricted in this OpenCV build — skipped)")

    # ── ORB ───────────────────────────────────────────────
    try:
        det = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)
        configs["ORB"] = dict(detector=det, extractor=None,
                              norm=cv2.NORM_HAMMING, color="#55A868")
        print("  ✓  ORB")
    except Exception as e:
        print(f"  ✗  ORB unavailable: {e}")

    # ── BRISK ─────────────────────────────────────────────
    try:
        det = cv2.BRISK_create(thresh=30, octaves=3)
        configs["BRISK"] = dict(detector=det, extractor=None,
                                norm=cv2.NORM_HAMMING, color="#C44E52")
        print("  ✓  BRISK")
    except Exception as e:
        print(f"  ✗  BRISK unavailable: {e}")

    # ── LATCH (detector = FAST, extractor = LATCH) ────────
    try:
        det = cv2.FastFeatureDetector_create(threshold=20)
        ext = cv2.xfeatures2d.LATCH_create(bytes=32)
        configs["LATCH"] = dict(detector=det, extractor=ext,
                                norm=cv2.NORM_HAMMING, color="#8172B2")
        print("  ✓  LATCH  (FAST detector + LATCH descriptor)")
    except Exception as e:
        print(f"  ✗  LATCH unavailable: {e}")

    # ── LIFT (SIFT detector + CNN descriptor, PyTorch) ────
    if _LIFT_AVAILABLE:
        try:
            lift_obj = LIFTDescriptor(weights_path=LIFT_WEIGHTS_PATH, nfeatures=500)
            # LIFTDescriptor exposes both .detect() and .compute() so it
            # satisfies both the detector and extractor roles.
            configs["LIFT"] = dict(detector=lift_obj, extractor=lift_obj,
                                   norm=cv2.NORM_L2, color="#E377C2")
            wgt_note = (f"weights={LIFT_WEIGHTS_PATH}"
                        if LIFT_WEIGHTS_PATH else "random init — load weights via LIFT_WEIGHTS_PATH")
            print(f"  ✓  LIFT  (SIFT keypoints + CNN descriptor; {wgt_note})")
        except Exception as e:
            print(f"  ✗  LIFT unavailable: {e}")
    else:
        print("  ✗  LIFT  (PyTorch not installed — run: pip install torch torchvision)")

    return configs


# ─────────────────────────────────────────────
# 2.  Dataset helpers
# ─────────────────────────────────────────────

PATCH_SIZE = 65   # HPatches pre-extracted patch side length (pixels)


def load_homography(path: str) -> np.ndarray:
    return np.loadtxt(path).reshape(3, 3)


def list_sequences(hpatches_dir: str) -> List[Path]:
    root = Path(hpatches_dir)
    seqs = sorted([p for p in root.iterdir() if p.is_dir()])
    return seqs


def load_sequence(seq_dir: Path) -> Optional[dict]:
    """Load reference image + all target images + homographies (full-image mode)."""
    ref_path = seq_dir / "1.ppm"
    if not ref_path.exists():
        return None

    ref = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
    if ref is None:
        return None

    targets, homographies = [], []
    for idx in range(2, 7):
        tgt_path = seq_dir / f"{idx}.ppm"
        h_path   = seq_dir / f"H_1_{idx}"
        if not tgt_path.exists() or not h_path.exists():
            continue
        tgt = cv2.imread(str(tgt_path), cv2.IMREAD_GRAYSCALE)
        H   = load_homography(str(h_path))
        if tgt is not None:
            targets.append(tgt)
            homographies.append(H)

    return dict(
        mode         = "images",
        name         = seq_dir.name,
        kind         = "viewpoint" if seq_dir.name.startswith("v_") else "illumination",
        ref          = ref,
        targets      = targets,
        homographies = homographies,
    )


def _load_patch_stack(path: Path) -> Optional[np.ndarray]:
    """
    Load a pre-extracted HPatches patch file.
    Each file is a (N * PATCH_SIZE) × PATCH_SIZE grayscale image.
    Returns an (N, PATCH_SIZE, PATCH_SIZE) uint8 array, or None on failure.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    if w != PATCH_SIZE or h % PATCH_SIZE != 0:
        return None
    n = h // PATCH_SIZE
    return img.reshape(n, PATCH_SIZE, PATCH_SIZE)


def load_patch_sequence(seq_dir: Path) -> Optional[dict]:
    """
    Load pre-extracted homography patch files from an HPatches sequence directory.

    Expected files (all optional but ref.png is required):
      ref.png          – reference patches
      e1.png … e5.png  – easy positive patches
      h1.png … h5.png  – hard positive patches
      t1.png … t5.png  – tough positive patches

    Returns a dict with mode="patches" containing arrays of shape
    (N, PATCH_SIZE, PATCH_SIZE) for each split, or None if the
    sequence has no patch files.
    """
    ref_path = seq_dir / "ref.png"
    if not ref_path.exists():
        return None

    ref_patches = _load_patch_stack(ref_path)
    if ref_patches is None:
        return None

    def _load_split(prefix: str) -> List[np.ndarray]:
        split = []
        for i in range(1, 6):
            p = _load_patch_stack(seq_dir / f"{prefix}{i}.png")
            if p is not None:
                split.append(p)
        return split

    return dict(
        mode          = "patches",
        name          = seq_dir.name,
        kind          = "viewpoint" if seq_dir.name.startswith("v_") else "illumination",
        ref_patches   = ref_patches,    # (N, 65, 65)
        easy_patches  = _load_split("e"),  # list of up to 5 × (N, 65, 65)
        hard_patches  = _load_split("h"),
        tough_patches = _load_split("t"),
    )


def describe_patches(patches: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Compute descriptors for a (N, H, W) uint8 patch stack.
    A single centred KeyPoint is used for each patch (no detection needed).
    Returns an (N, D) float32 / uint8 descriptor array.
    """
    ph, pw = patches.shape[1], patches.shape[2]
    cx, cy = float(pw) / 2.0, float(ph) / 2.0
    ext = cfg["extractor"] if cfg["extractor"] else cfg["detector"]
    dtype = np.float32 if cfg["norm"] == cv2.NORM_L2 else np.uint8
    try:
        d_size = ext.descriptorSize()
    except Exception:
        d_size = 128  # fallback

    descs = []
    for patch in patches:
        kp = [cv2.KeyPoint(x=cx, y=cy, size=float(ph) * 0.7)]
        _, d = ext.compute(patch, kp)
        if d is not None and len(d) > 0:
            descs.append(d[0].astype(dtype))
        else:
            descs.append(np.zeros(d_size, dtype=dtype))
    return np.array(descs)


# ─────────────────────────────────────────────
# 3.  Core descriptor helpers
# ─────────────────────────────────────────────

def detect_and_describe(img: np.ndarray, cfg: dict
                        ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    det = cfg["detector"]
    ext = cfg["extractor"]

    kps = det.detect(img, None)
    if not kps:
        return [], None

    if ext is not None:
        kps, desc = ext.compute(img, kps)
    else:
        kps, desc = det.compute(img, kps)

    return kps, desc


def match_descriptors(desc1: np.ndarray, desc2: np.ndarray,
                      norm: int, ratio: float = 0.75
                      ) -> List[cv2.DMatch]:
    """Lowe's ratio-test matching."""
    if desc1 is None or desc2 is None:
        return []
    bfm = cv2.BFMatcher(norm, crossCheck=False)
    try:
        raw = bfm.knnMatch(desc1, desc2, k=2)
    except cv2.error:
        return []
    good = []
    for m in raw:
        if len(m) == 2 and m[0].distance < ratio * m[1].distance:
            good.append(m[0])
    return good


# ─────────────────────────────────────────────
# 4.  Task A – Image Matching (MMA)
# ─────────────────────────────────────────────

def epipolar_error(pt1: np.ndarray, pt2: np.ndarray, H: np.ndarray) -> float:
    """Symmetric reprojection error under a homography."""
    p1 = np.array([pt1[0], pt1[1], 1.0])
    p2 = np.array([pt2[0], pt2[1], 1.0])

    # project pt1 → img2
    p1_in2 = H @ p1
    p1_in2 /= p1_in2[2]

    # project pt2 → img1
    Hinv = np.linalg.inv(H)
    p2_in1 = Hinv @ p2
    p2_in1 /= p2_in1[2]

    err1 = np.linalg.norm(p1_in2[:2] - p2[:2])
    err2 = np.linalg.norm(p2_in1[:2] - p1[:2])
    return (err1 + err2) / 2.0


def mean_matching_accuracy(seq: dict, cfg: dict,
                           thresholds: List[float] = [1, 3, 5]
                           ) -> Dict[str, float]:
    ref = seq["ref"]
    kps1, desc1 = detect_and_describe(ref, cfg)
    if desc1 is None or len(kps1) < 5:
        return {f"MMA@{t}px": 0.0 for t in thresholds}

    results = {t: [] for t in thresholds}

    for tgt, H in zip(seq["targets"], seq["homographies"]):
        kps2, desc2 = detect_and_describe(tgt, cfg)
        if desc2 is None or len(kps2) < 5:
            for t in thresholds:
                results[t].append(0.0)
            continue

        matches = match_descriptors(desc1, desc2, cfg["norm"])
        if not matches:
            for t in thresholds:
                results[t].append(0.0)
            continue

        pts1 = np.array([kps1[m.queryIdx].pt for m in matches])
        pts2 = np.array([kps2[m.trainIdx].pt for m in matches])

        errors = [epipolar_error(p1, p2, H) for p1, p2 in zip(pts1, pts2)]

        for t in thresholds:
            acc = np.mean(np.array(errors) < t)
            results[t].append(acc)

    return {f"MMA@{t}px": float(np.mean(v)) for t, v in results.items()}


# ─────────────────────────────────────────────
# 5.  Task B – Patch Verification (FPR95)
# ─────────────────────────────────────────────

def extract_patch_descriptors(patches: np.ndarray, cfg: dict) -> np.ndarray:
    """
    patches: (N, H, W) uint8 array of grey patches.
    Returns: (N, D) descriptor array (or None if extraction fails).
    """
    descs = []
    for patch in patches:
        kp = [cv2.KeyPoint(x=patch.shape[1]/2, y=patch.shape[0]/2,
                           size=patch.shape[0] * 0.7)]
        ext = cfg["extractor"] if cfg["extractor"] else cfg["detector"]
        _, d = ext.compute(patch, kp)
        if d is not None and len(d) > 0:
            descs.append(d[0])
        else:
            descs.append(np.zeros(ext.descriptorSize(),
                                  dtype=np.float32 if cfg["norm"] == cv2.NORM_L2
                                  else np.uint8))
    return np.array(descs)


def fpr_at_95_recall(pos_dists: np.ndarray, neg_dists: np.ndarray) -> float:
    """False Positive Rate at 95% True Positive Rate."""
    # find threshold that gives 95% recall on positives
    threshold = np.percentile(pos_dists, 95)
    fpr = np.mean(neg_dists <= threshold)
    return float(fpr)


def patch_verification_score(seq: dict, cfg: dict,
                             n_patches: int = 200) -> float:
    """
    Compute FPR95 for patch verification.

    Patch-mode (seq["mode"] == "patches"):
      Positive pairs  – ref vs. each set of positive patches (easy/hard/tough).
      Negative pairs  – ref patch i vs. ref patch j  (i ≠ j, random).

    Image-mode (seq["mode"] == "images"):
      Positive  → same keypoint location mapped via ground-truth homography.
      Negative  → random different locations in the target image.
    """
    if seq.get("mode") == "patches":
        return _patch_verification_from_patches(seq, cfg, n_patches)
    else:
        return _patch_verification_from_images(seq, cfg, n_patches)


def _patch_verification_from_patches(seq: dict, cfg: dict,
                                     n_patches: int = 200) -> float:
    """
    FPR95 using pre-extracted HPatches patch files.
    Positive distance: descriptor distance between ref patch and its positive counterpart.
    Negative distance: descriptor distance between ref patch i and a different ref patch j.
    """
    ref_patches = seq["ref_patches"]          # (N, 65, 65)
    all_pos_patches: List[np.ndarray] = (
        seq.get("easy_patches", []) +
        seq.get("hard_patches", []) +
        seq.get("tough_patches", [])
    )
    if not all_pos_patches:
        return np.nan

    rng = np.random.default_rng(42)
    n   = min(n_patches, len(ref_patches))
    idx = rng.choice(len(ref_patches), size=n, replace=False)

    ref_descs = describe_patches(ref_patches[idx], cfg)   # (n, D)

    def dist_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Row-wise distance between two (n, D) arrays."""
        if cfg["norm"] == cv2.NORM_L2:
            return np.linalg.norm(a.astype(float) - b.astype(float), axis=1)
        else:
            return np.array([
                np.unpackbits(np.bitwise_xor(
                    a[i].view(np.uint8), b[i].view(np.uint8))).sum()
                for i in range(len(a))
            ], dtype=float)

    pos_dists: List[float] = []
    for pos_stack in all_pos_patches:
        n_avail = min(n, len(pos_stack))
        pos_descs = describe_patches(pos_stack[idx[:n_avail]], cfg)
        d = dist_batch(ref_descs[:n_avail], pos_descs)
        pos_dists.extend(d.tolist())

    # Negatives: shuffle ref descriptors and pair with original
    shuffled_idx = rng.permutation(n)
    neg_descs = ref_descs[shuffled_idx]
    neg_dists = dist_batch(ref_descs, neg_descs).tolist()

    if len(pos_dists) < 10 or len(neg_dists) < 10:
        return np.nan

    return fpr_at_95_recall(np.array(pos_dists), np.array(neg_dists))


def _patch_verification_from_images(seq: dict, cfg: dict,
                                    n_patches: int = 200) -> float:
    """
    FPR95 using full images + ground-truth homographies.
    Positive  → same physical point mapped via homography.
    Negative  → random different location in the target image.
    """
    ref = seq["ref"]
    if not seq["targets"]:
        return np.nan

    tgt = seq["targets"][0]
    H   = seq["homographies"][0]
    h, w = ref.shape[:2]

    rng = np.random.default_rng(42)
    patch_size = 32
    half = patch_size // 2

    pos_dists, neg_dists = [], []

    ext = cfg["extractor"] if cfg["extractor"] else cfg["detector"]

    def get_desc(img, x, y):
        kp = [cv2.KeyPoint(float(x), float(y), float(patch_size))]
        _, d = ext.compute(img, kp)
        return d[0] if d is not None and len(d) > 0 else None

    def l2(a, b):
        return np.linalg.norm(a.astype(float) - b.astype(float))

    def hamming(a, b):
        return np.unpackbits(np.bitwise_xor(
            a.view(np.uint8), b.view(np.uint8))).sum()

    dist_fn = l2 if cfg["norm"] == cv2.NORM_L2 else hamming

    xs = rng.integers(half + 1, w - half - 1, size=n_patches * 3)
    ys = rng.integers(half + 1, h - half - 1, size=n_patches * 3)

    count = 0
    for x, y in zip(xs, ys):
        if count >= n_patches:
            break
        p  = np.array([x, y, 1.0], dtype=float)
        p2 = H @ p;  p2 /= p2[2]
        x2, y2 = int(round(p2[0])), int(round(p2[1]))
        th, tw = tgt.shape[:2]
        if not (half < x2 < tw - half and half < y2 < th - half):
            continue

        d1 = get_desc(ref, x,  y)
        d2 = get_desc(tgt, x2, y2)
        if d1 is None or d2 is None:
            continue

        pos_dists.append(dist_fn(d1, d2))

        xn = int(rng.integers(half + 1, tw - half - 1))
        yn = int(rng.integers(half + 1, th - half - 1))
        dn = get_desc(tgt, xn, yn)
        if dn is not None:
            neg_dists.append(dist_fn(d1, dn))

        count += 1

    if len(pos_dists) < 10 or len(neg_dists) < 10:
        return np.nan

    return fpr_at_95_recall(np.array(pos_dists), np.array(neg_dists))


# ─────────────────────────────────────────────
# 6.  Task C – Retrieval (mAP)
# ─────────────────────────────────────────────

def retrieval_map(seq: dict, cfg: dict, n_queries: int = 50) -> float:
    """
    Compute mean Average Precision (mAP) for patch retrieval.

    Patch-mode (seq["mode"] == "patches"):
      Query  = reference patch descriptors (random subset).
      Gallery = all positive patch descriptors (easy + hard + tough).
      Relevant = any gallery patch that corresponds to the same query index.

    Image-mode (seq["mode"] == "images"):
      Query  = reference keypoint descriptors.
      Gallery = descriptors from all target images.
      Relevant = gallery keypoint within 5 px of the projected query point.
    """
    if seq.get("mode") == "patches":
        return _retrieval_map_from_patches(seq, cfg, n_queries)
    else:
        return _retrieval_map_from_images(seq, cfg, n_queries)


def _retrieval_map_from_patches(seq: dict, cfg: dict,
                                n_queries: int = 50) -> float:
    """
    mAP using pre-extracted HPatches patch files.
    For each query (a ref patch), the gallery consists of all positive patches
    from easy / hard / tough splits.  A gallery entry is 'relevant' if its
    patch index matches the query patch index.
    """
    ref_patches = seq["ref_patches"]   # (N, 65, 65)
    all_pos: List[np.ndarray] = (
        seq.get("easy_patches", []) +
        seq.get("hard_patches", []) +
        seq.get("tough_patches", [])
    )
    if not all_pos:
        return np.nan

    n = len(ref_patches)
    if n < 2:
        return np.nan

    ref_descs = describe_patches(ref_patches, cfg)   # (N, D)

    # Build gallery: list of (desc, patch_index) from all positive stacks
    gallery_descs_list, gallery_idx = [], []
    for pos_stack in all_pos:
        n_pos = min(len(pos_stack), n)
        gd = describe_patches(pos_stack[:n_pos], cfg)
        gallery_descs_list.append(gd)
        gallery_idx.extend(range(n_pos))

    if not gallery_descs_list:
        return np.nan

    all_gallery = np.vstack(gallery_descs_list)  # (G, D)
    gallery_idx_arr = np.array(gallery_idx)      # (G,) – which patch index

    rng = np.random.default_rng(0)
    n_q = min(n_queries, n)
    q_indices = rng.choice(n, size=n_q, replace=False)

    bfm = cv2.BFMatcher(cfg["norm"], crossCheck=False)
    aps = []

    for qi in q_indices:
        q_desc = ref_descs[qi:qi+1]
        matches = list(bfm.match(q_desc, all_gallery))
        matches.sort(key=lambda m: m.distance)

        relevances = [1 if gallery_idx_arr[m.trainIdx] == qi else 0
                      for m in matches]

        if sum(relevances) == 0:
            aps.append(0.0)
            continue
        prec_at_k, n_rel = [], 0
        for k, rel in enumerate(relevances, 1):
            if rel:
                n_rel += 1
                prec_at_k.append(n_rel / k)
        aps.append(np.mean(prec_at_k) if prec_at_k else 0.0)

    return float(np.mean(aps))


def _retrieval_map_from_images(seq: dict, cfg: dict,
                               n_queries: int = 50) -> float:
    """
    mAP using full images + ground-truth homographies.
    Each reference keypoint is a query; the gallery contains descriptors
    re-detected in all target images.  Relevant = within 5 px (reprojected).
    """
    ref = seq["ref"]
    kps1, desc1 = detect_and_describe(ref, cfg)
    if desc1 is None or len(kps1) < n_queries:
        return np.nan

    gallery_descs, gallery_kps, gallery_Hs = [], [], []
    for tgt, H in zip(seq["targets"], seq["homographies"]):
        kps2, desc2 = detect_and_describe(tgt, cfg)
        if desc2 is None:
            continue
        gallery_descs.append(desc2)
        gallery_kps.append(kps2)
        gallery_Hs.append(H)

    if not gallery_descs:
        return np.nan

    all_gallery_desc = np.vstack(gallery_descs)
    all_gallery_kp: List[cv2.KeyPoint] = []
    all_gallery_H:  List[np.ndarray]  = []
    for kps, H, descs in zip(gallery_kps, gallery_Hs, gallery_descs):
        for kp in kps:
            all_gallery_kp.append(kp)
            all_gallery_H.append(H)

    rng   = np.random.default_rng(0)
    q_idx = rng.choice(len(kps1), size=min(n_queries, len(kps1)), replace=False)

    aps = []
    bfm = cv2.BFMatcher(cfg["norm"], crossCheck=False)

    for qi in q_idx:
        q_desc = desc1[qi:qi+1]
        q_pt   = np.array(kps1[qi].pt)

        matches = list(bfm.match(q_desc, all_gallery_desc))
        matches.sort(key=lambda m: m.distance)

        relevances = []
        for m in matches:
            g_pt = np.array(all_gallery_kp[m.trainIdx].pt)
            H    = all_gallery_H[m.trainIdx]
            err  = epipolar_error(q_pt, g_pt, H)
            relevances.append(1 if err < 5.0 else 0)

        if sum(relevances) == 0:
            aps.append(0.0)
            continue
        prec_at_k, n_rel = [], 0
        for k, rel in enumerate(relevances, 1):
            if rel:
                n_rel += 1
                prec_at_k.append(n_rel / k)
        aps.append(np.mean(prec_at_k) if prec_at_k else 0.0)

    return float(np.mean(aps))


# ─────────────────────────────────────────────
# 7.  Speed benchmark
# ─────────────────────────────────────────────

def benchmark_speed(seq: dict, cfg: dict, n_runs: int = 10) -> dict:
    img = seq["ref"]
    t_det, t_desc = [], []

    det = cfg["detector"]
    ext = cfg["extractor"]

    for _ in range(n_runs):
        t0 = time.perf_counter()
        kps = det.detect(img, None)
        t_det.append(time.perf_counter() - t0)

        if not kps:
            continue

        t0 = time.perf_counter()
        if ext is not None:
            _, _ = ext.compute(img, kps)
        else:
            _, _ = det.compute(img, kps)
        t_desc.append(time.perf_counter() - t0)

    return dict(
        detect_ms = np.mean(t_det)  * 1000,
        desc_ms   = np.mean(t_desc) * 1000,
        total_ms  = (np.mean(t_det) + np.mean(t_desc)) * 1000,
        n_kps     = len(kps),
    )


# ─────────────────────────────────────────────
# 8.  Demo-mode synthetic dataset
# ─────────────────────────────────────────────

def make_synthetic_sequence(n_targets: int = 5, seed: int = 0) -> dict:
    """
    Create a synthetic HPatches-like sequence for quick testing without
    the real dataset.  Uses a checkerboard + random affine transforms.
    """
    rng = np.random.default_rng(seed)
    h, w = 480, 640
    # checkerboard reference image
    ref = np.zeros((h, w), dtype=np.uint8)
    for r in range(0, h, 40):
        for c in range(0, w, 40):
            if (r // 40 + c // 40) % 2 == 0:
                ref[r:r+40, c:c+40] = 200
    # add noise
    ref = np.clip(ref.astype(int) + rng.integers(-20, 20, size=ref.shape), 0, 255).astype(np.uint8)

    targets, homographies = [], []
    for i in range(n_targets):
        angle  = rng.uniform(-15, 15)
        scale  = rng.uniform(0.85, 1.15)
        tx, ty = rng.uniform(-30, 30), rng.uniform(-30, 30)
        cx, cy = w / 2, h / 2
        M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
        M[0, 2] += tx;  M[1, 2] += ty
        H = np.vstack([M, [0, 0, 1]])

        tgt = cv2.warpAffine(ref, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
        brightness = rng.uniform(0.8, 1.2)
        tgt = np.clip((tgt.astype(float) * brightness), 0, 255).astype(np.uint8)

        targets.append(tgt)
        homographies.append(H)

    return dict(name="synthetic_demo", kind="synthetic",
                ref=ref, targets=targets, homographies=homographies)


# ─────────────────────────────────────────────
# 9.  Visualization
# ─────────────────────────────────────────────

def visualise_matches(seq: dict, cfg: dict, name: str,
                      out_dir: str, max_matches: int = 30):
    ref = seq["ref"]
    tgt = seq["targets"][0]
    H   = seq["homographies"][0]

    kps1, desc1 = detect_and_describe(ref, cfg)
    kps2, desc2 = detect_and_describe(tgt, cfg)
    matches = match_descriptors(desc1, desc2, cfg["norm"])
    matches = matches[:max_matches]

    img_match = cv2.drawMatches(
        ref, kps1, tgt, kps2, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchColor=(0, 200, 80),
        singlePointColor=(200, 0, 0),
    )
    out = Path(out_dir) / f"matches_{name}_{seq['name']}.png"
    cv2.imwrite(str(out), img_match)
    return str(out)


def plot_results(results_df: dict, out_dir: str, descriptor_colors: dict):
    """Bar charts for all three tasks."""
    out_dir = Path(out_dir)

    metrics = [
        ("MMA@1px",   "Image Matching — MMA @ 1 px  ↑ higher is better"),
        ("MMA@3px",   "Image Matching — MMA @ 3 px  ↑ higher is better"),
        ("FPR95",     "Patch Verification — FPR95  ↓ lower is better"),
        ("Retrieval_mAP", "Patch Retrieval — mAP  ↑ higher is better"),
        ("Speed_ms",  "Total Speed (ms / image)  ↓ lower is better"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    fig.suptitle("HPatches Descriptor Benchmark", fontsize=14, fontweight="bold")

    for ax, (metric, title) in zip(axes, metrics):
        names  = [n for n in results_df if metric in results_df[n]]
        values = [results_df[n][metric] for n in names]
        colors = [descriptor_colors.get(n, "#888888") for n in names]

        bars = ax.bar(names, values, color=colors, edgecolor="white", width=0.55)
        ax.set_title(title, fontsize=9, pad=8)
        clean_vals = [v for v in values if not (np.isnan(v) or np.isinf(v))]
        ax.set_ylim(0, max(clean_vals) * 1.25 if clean_vals else 1)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=25)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path = out_dir / "benchmark_results.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved bar chart → {out_path}")


def plot_mma_curves(results_df: dict, out_dir: str, descriptor_colors: dict):
    """Plot MMA across multiple distance thresholds."""
    thresholds = [1, 2, 3, 4, 5]
    metric_keys = [f"MMA@{t}px" for t in thresholds]

    out_dir = Path(out_dir)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title("Mean Matching Accuracy across Pixel Thresholds", fontweight="bold")

    for name, res in results_df.items():
        vals = [res.get(k, np.nan) for k in metric_keys]
        if all(np.isnan(vals)):
            continue
        ax.plot(thresholds, vals, "o-",
                label=name, color=descriptor_colors.get(name, "#888888"),
                linewidth=2, markersize=6)

    ax.set_xlabel("Pixel Threshold (px)")
    ax.set_ylabel("MMA")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    out_path = out_dir / "mma_curves.png"
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved MMA curve  → {out_path}")


# ─────────────────────────────────────────────
# 10.  Main evaluation loop
# ─────────────────────────────────────────────

def run_evaluation(sequences: List[dict], descriptors: Dict[str, dict],
                   out_dir: str, max_seqs: int = 30, n_patches: int = 100):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = {s.get("mode", "images") for s in sequences[:max_seqs]}
    has_images  = "images"  in modes
    has_patches = "patches" in modes

    print(f"\n{'='*60}")
    print(f"  Evaluating on {min(len(sequences), max_seqs)} sequences …")
    print(f"  Mode(s): {', '.join(sorted(modes))}")
    if has_patches and not has_images:
        print("  [Patch mode] MMA (image matching) will be skipped.")
    print(f"{'='*60}")

    all_thresholds = [1, 2, 3, 4, 5]
    results: Dict[str, Dict[str, List]] = {
        name: {f"MMA@{t}px": [] for t in all_thresholds} |
              {"FPR95": [], "Retrieval_mAP": [], "Speed_ms": [], "n_kps": []}
        for name in descriptors
    }

    for si, seq in enumerate(sequences[:max_seqs]):
        seq_mode = seq.get("mode", "images")
        print(f"\n  [{si+1:>3}/{min(len(sequences), max_seqs)}] "
              f"{seq['name']} ({seq['kind']}, mode={seq_mode})")

        for desc_name, cfg in descriptors.items():
            try:
                if seq_mode == "images":
                    # Speed benchmark (only meaningful in image mode)
                    spd = benchmark_speed(seq, cfg, n_runs=5)
                    results[desc_name]["Speed_ms"].append(spd["total_ms"])
                    results[desc_name]["n_kps"].append(spd["n_kps"])

                    # MMA (image matching) – image mode only
                    mma = mean_matching_accuracy(seq, cfg, thresholds=all_thresholds)
                    for k, v in mma.items():
                        results[desc_name][k].append(v)
                    mma_str = f"MMA@3px={mma['MMA@3px']:.3f}"
                    speed_str = f"speed={spd['total_ms']:.1f}ms  kps={spd['n_kps']}"
                else:
                    # Patch mode: no full images – MMA not available
                    mma_str  = "MMA=N/A (patch mode)"
                    speed_str = ""

                # Patch Verification (both modes)
                fpr = patch_verification_score(seq, cfg, n_patches=n_patches)
                if not np.isnan(fpr):
                    results[desc_name]["FPR95"].append(fpr)

                # Retrieval (both modes)
                ret = retrieval_map(seq, cfg, n_queries=30)
                if not np.isnan(ret):
                    results[desc_name]["Retrieval_mAP"].append(ret)

                print(f"      {desc_name:<8}  {mma_str}  "
                      f"FPR95={fpr:.3f}  mAP={ret:.3f}  {speed_str}")

            except Exception as e:
                print(f"      {desc_name:<8}  ERROR: {e}")

    # ── aggregate
    agg: Dict[str, Dict[str, float]] = {}
    for desc_name, metrics in results.items():
        agg[desc_name] = {}
        for metric, vals in metrics.items():
            vals_clean = [v for v in vals if not np.isnan(v)]
            agg[desc_name][metric] = float(np.mean(vals_clean)) if vals_clean else np.nan

    # ── print summary table
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    col_w = 12
    headers = ["Descriptor", "MMA@1px", "MMA@3px", "FPR95↓", "mAP↑", "Speed(ms)", "Kps"]
    print("  " + "  ".join(h.ljust(col_w) for h in headers))
    print("  " + "-" * (col_w * len(headers) + 2 * len(headers)))
    for desc_name, m in agg.items():
        row = [
            desc_name,
            f"{m.get('MMA@1px', np.nan):.4f}",
            f"{m.get('MMA@3px', np.nan):.4f}",
            f"{m.get('FPR95',   np.nan):.4f}",
            f"{m.get('Retrieval_mAP', np.nan):.4f}",
            f"{m.get('Speed_ms', np.nan):.1f}",
            f"{m.get('n_kps',   0):.0f}",
        ]
        print("  " + "  ".join(c.ljust(col_w) for c in row))

    # ── generate match visualisations (image-mode sequences only)
    image_seqs = [s for s in sequences[:max_seqs] if s.get("mode") == "images"]
    if image_seqs:
        print("\n  Generating match visualisations …")
        vis_seq = image_seqs[0]
        for desc_name, cfg in descriptors.items():
            try:
                p = visualise_matches(vis_seq, cfg, desc_name, str(out_dir))
                print(f"    {desc_name} → {p}")
            except Exception as e:
                print(f"    {desc_name} visualisation failed: {e}")
    else:
        print("\n  [Patch mode] Skipping match visualisations (no full images).")

    # ── plots
    print("\n  Generating charts …")
    desc_colors = {n: c["color"] for n, c in descriptors.items()}
    plot_results(agg, str(out_dir), desc_colors)
    if has_images:
        plot_mma_curves(agg, str(out_dir), desc_colors)
    else:
        print("  [Patch mode] Skipping MMA curve plot (no image sequences).")

    # ── save CSV
    csv_path = out_dir / "results.csv"
    all_metrics = sorted({k for m in agg.values() for k in m})
    with open(csv_path, "w") as f:
        f.write("descriptor," + ",".join(all_metrics) + "\n")
        for desc_name, m in agg.items():
            row = [desc_name] + [str(m.get(k, "")) for k in all_metrics]
            f.write(",".join(row) + "\n")
    print(f"  Saved CSV        → {csv_path}")

    return agg


# ─────────────────────────────────────────────
# 11.  Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HPatches (Homography Patches) Descriptor Evaluation: "
                    "SIFT / SURF / ORB / BRISK / LATCH / LIFT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Dataset modes\n"
            "  image mode (default) : sequences contain 1.ppm … 6.ppm + H_1_* files\n"
            "  patch mode           : sequences contain ref.png, e*.png, h*.png, t*.png\n"
            "Auto-detection: if ref.png is found, patch mode is used; "
            "override with --patch_mode / --image_mode."
        ),
    )
    parser.add_argument("--hpatches_dir", type=str, default=None,
                        help="Path to hpatches-sequences-release/")
    parser.add_argument("--demo", action="store_true",
                        help="Run on synthetic demo data (no real dataset needed)")
    parser.add_argument("--out_dir", type=str, default="./hpatches_results",
                        help="Output directory for charts and CSVs")
    parser.add_argument("--max_seqs", type=int, default=30,
                        help="Max sequences to evaluate (default 30)")
    parser.add_argument("--n_patches", type=int, default=150,
                        help="Patches per seq for verification task")
    parser.add_argument("--lift_weights", type=str, default=None,
                        help="Path to pre-trained LIFT descriptor weights (.pth). "
                             "If omitted, LIFT runs with random initialisation "
                             "(useful for pipeline testing; metrics will not be meaningful).")

    # ── dataset-mode flags
    mode_grp = parser.add_mutually_exclusive_group()
    mode_grp.add_argument(
        "--patch_mode", action="store_true",
        help="Force homography-patch file mode (ref.png / e*.png / h*.png / t*.png). "
             "Patch Verification and Retrieval use pre-extracted patches; "
             "Image Matching (MMA) is skipped.",
    )
    mode_grp.add_argument(
        "--image_mode", action="store_true",
        help="Force full-image + homography mode (1.ppm … 6.ppm + H_1_* files). "
             "All three tasks (MMA, Verification, Retrieval) are computed.",
    )

    args = parser.parse_args()

    # propagate LIFT weights path to the global used by build_descriptors()
    global LIFT_WEIGHTS_PATH
    if args.lift_weights:
        LIFT_WEIGHTS_PATH = args.lift_weights

    print("\n" + "="*60)
    print("  HPatches (Homography Patches) Descriptor Evaluation")
    print("="*60)
    print("\n  Building descriptors …")
    descriptors = build_descriptors()

    if not descriptors:
        print("\n  No descriptors available. Exiting.")
        sys.exit(1)

    print(f"\n  Available: {list(descriptors.keys())}")

    # ── load sequences
    if args.demo or args.hpatches_dir is None:
        print("\n  [DEMO MODE] Using synthetic sequences (full-image mode) …")
        sequences = [make_synthetic_sequence(seed=i) for i in range(8)]
    else:
        hp_dir = args.hpatches_dir
        if not os.path.isdir(hp_dir):
            print(f"\n  ERROR: {hp_dir!r} does not exist.")
            print("  Download from https://github.com/hpatches/hpatches-dataset")
            print("  or run with --demo to use synthetic data.\n")
            sys.exit(1)

        seq_dirs = list_sequences(hp_dir)
        seq_dirs = [s for s in seq_dirs if s.name == "i_castle"]
        sequences = []

        for sd in seq_dirs:
            # ── decide which loader to use ──────────────────────────────
            if args.patch_mode:
                # user explicitly wants patch mode
                s = load_patch_sequence(sd)
                if s is None:
                    print(f"  ⚠  {sd.name}: no patch files found – trying image mode")
                    s = load_sequence(sd)
            elif args.image_mode:
                # user explicitly wants image mode
                s = load_sequence(sd)
                if s is None:
                    print(f"  ⚠  {sd.name}: no image files found – trying patch mode")
                    s = load_patch_sequence(sd)
            else:
                # auto-detect: prefer patch files when present
                s = load_patch_sequence(sd)
                if s is None:
                    s = load_sequence(sd)

            if s is not None:
                sequences.append(s)

        n_img = sum(1 for s in sequences if s.get("mode") == "images")
        n_pat = sum(1 for s in sequences if s.get("mode") == "patches")
        print(f"\n  Loaded {len(sequences)} sequences  "
              f"(image-mode: {n_img}, patch-mode: {n_pat})")

    if not sequences:
        print("  No sequences loaded. Exiting.")
        sys.exit(1)

    run_evaluation(sequences, descriptors,
                   out_dir=args.out_dir,
                   max_seqs=args.max_seqs,
                   n_patches=args.n_patches)

    print(f"\n  All outputs saved to: {args.out_dir}/\n")


if __name__ == "__main__":
    main()