import cv2
import kornia.feature as KF
import kornia.geometry as KG
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.io import decode_image

from ripe import vgg_hyper
from ripe.utils.utils import cv2_matches_from_kornia, resize_image, to_cv_kpts

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from pathlib import Path
weights = Path("ripe/weights/ripe_weights.pth")

model = vgg_hyper(weights).to(dev)
model.eval()

img1 = cv2.imread(r"C:\Users\SSD\Downloads\hpatches\data\i_castle\1.ppm")
img2 = cv2.imread(r"C:\Users\SSD\Downloads\hpatches\data\i_castle\2.ppm")

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

image1 = torch.from_numpy(img1).permute(2, 0, 1).float().to(dev) / 255.0
image2 = torch.from_numpy(img2).permute(2, 0, 1).float().to(dev) / 255.0

image1 = resize_image(image1)
image2 = resize_image(image2)

kpts_1, desc_1, score_1 = model.detectAndCompute(image1, threshold=0.5, top_k=2048)
kpts_2, desc_2, score_2 = model.detectAndCompute(image2, threshold=0.5, top_k=2048)

matcher = KF.DescriptorMatcher("mnn")  # threshold is not used with mnn
match_dists, match_idxs = matcher(desc_1, desc_2)

matched_pts_1 = kpts_1[match_idxs[:, 0]]
matched_pts_2 = kpts_2[match_idxs[:, 1]]

H, mask = KG.ransac.RANSAC(model_type="fundamental", inl_th=1.0)(matched_pts_1, matched_pts_2)
matchesMask = mask.int().ravel().tolist()

result_ransac = cv2.drawMatches(
    (image1.cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8),
    to_cv_kpts(kpts_1, score_1),
    (image2.cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8),
    to_cv_kpts(kpts_2, score_2),
    cv2_matches_from_kornia(match_dists, match_idxs),
    None,
    matchColor=(0, 255, 0),
    matchesMask=matchesMask,
    # matchesMask=None, # without RANSAC filtering
    singlePointColor=(0, 0, 255),
    flags=cv2.DrawMatchesFlags_DEFAULT,
)

plt.imshow(result_ransac)
plt.axis("off")
plt.tight_layout()

# plt.show()
plt.savefig("result_ransac.png")
