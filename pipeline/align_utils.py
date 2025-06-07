# # align_utils.py  (replace the old file)
# from pathlib import Path
# import cv2, numpy as np

# def orb_align(
#     crop_moving: np.ndarray,
#     reference:   np.ndarray,
#     save_dir:    Path,
#     n_feat:int   = 5000,
#     ratio:float  = 0.75,
#     ransac:float = 4.0,
# ):
#     """
#     1) ORB + BFMatcher + Lowe filtering
#     2) RANSAC homography
#     3) writes  side_by_side.jpg  &  aligned_keypoints.jpg

#     Returns the warped moving patch.
#     """
#     orb = cv2.ORB_create(nfeatures=n_feat)
#     k1,d1 = orb.detectAndCompute(crop_moving, None)
#     k2,d2 = orb.detectAndCompute(reference,    None)
#     if d1 is None or d2 is None:
#         raise RuntimeError("No ORB key-points")

#     bf   = cv2.BFMatcher(cv2.NORM_HAMMING)
#     raw  = bf.knnMatch(d1, d2, k=2)
#     good = [m for m,n in raw if m.distance < ratio*n.distance]
#     if len(good) < 4:
#         raise RuntimeError("Not enough good matches")

#     src = np.float32([k1[m.queryIdx].pt for m in good])
#     dst = np.float32([k2[m.trainIdx].pt for m in good])
#     H, inl = cv2.findHomography(src, dst, cv2.RANSAC, ransac)
#     if H is None:
#         raise RuntimeError("Homography failed")

#     h,w = reference.shape[:2]
#     warped = cv2.warpPerspective(crop_moving, H, (w, h))

#     # ─── diagnostics ─────────────────────────────────────────────
#     save_dir.mkdir(parents=True, exist_ok=True)
#     cv2.imwrite(str(save_dir/'side_by_side.jpg'),
#                 cv2.hconcat([reference, warped]))

#     vis, off = cv2.hconcat([reference, warped]), np.array([w,0], int)
#     for s_pt, t_pt, keep in zip(src, dst, inl.ravel()):
#         if not keep: continue
#         s_w = cv2.perspectiveTransform(s_pt.reshape(1,1,2), H).reshape(2)
#         cv2.circle(vis, tuple(t_pt.astype(int)),         3, (0,255,0), -1)
#         cv2.circle(vis, tuple((s_w + off).astype(int)),  3, (0,0,255), -1)
#         cv2.line  (vis, tuple(t_pt.astype(int)),
#                         tuple((s_w + off).astype(int)), (200,200,255), 1)
#     cv2.imwrite(str(save_dir/'aligned_keypoints.jpg'), vis)
#     return warped


# align_utils.py
"""
ORB alignment of a cropped moving patch to a reference template.
Produces:
    side_by_side.jpg
    aligned_keypoints.jpg
"""
from pathlib import Path
import cv2, numpy as np


def orb_align(
    crop_moving: np.ndarray,
    reference:   np.ndarray,
    save_dir:    Path,
    n_feat:int   = 5000,
    ratio:float  = 0.75,
    ransac:float = 4.0,
):
    orb = cv2.ORB_create(nfeatures=n_feat)
    k1,d1 = orb.detectAndCompute(crop_moving, None)
    k2,d2 = orb.detectAndCompute(reference,    None)
    if d1 is None or d2 is None:
        raise RuntimeError("No ORB key-points.")

    bf   = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw  = bf.knnMatch(d1, d2, k=2)
    good = [m for m,n in raw if m.distance < ratio*n.distance]
    if len(good) < 4:
        raise RuntimeError("Not enough good matches.")

    src = np.float32([k1[m.queryIdx].pt for m in good])
    dst = np.float32([k2[m.trainIdx].pt for m in good])
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, ransac)
    if H is None:
        raise RuntimeError("Homography failed.")

    h,w = reference.shape[:2]
    warped = cv2.warpPerspective(crop_moving, H, (w,h))

    # ─── diagnostics ────────────────────────────────────────────────
    save_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_dir/'side_by_side.jpg'),
                cv2.hconcat([reference, warped]))

    vis, offs = cv2.hconcat([reference, warped]), np.array([w,0],int)
    for s,t,ok in zip(src, dst, inl.ravel()):
        if not ok: continue
        s_w = cv2.perspectiveTransform(s.reshape(1,1,2), H).reshape(2)
        cv2.circle(vis, tuple(t.astype(int)),         3, (0,255,0), -1)
        cv2.circle(vis, tuple((s_w+offs).astype(int)),3, (0,0,255), -1)
        cv2.line  (vis, tuple(t.astype(int)),
                        tuple((s_w+offs).astype(int)), (200,200,255), 1)
    cv2.imwrite(str(save_dir/'aligned_keypoints.jpg'), vis)
    return warped
