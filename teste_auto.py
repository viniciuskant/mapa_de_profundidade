import cv2
import numpy as np
import os
import json

LEFT_DIR = "videos/estereo_24/video_esquerda"
RIGHT_DIR = "videos/estereo_24/video_direita"
OUTPUT_DIR = "retificacao_auto"

os.makedirs(OUTPUT_DIR, exist_ok=True)

frame_start = 1
frame_end = 100

orb = cv2.ORB_create(
    nfeatures=5000,
    scaleFactor=1.2,
    nlevels=8
)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

def find_matches(imgL, imgR):

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    kpL, desL = orb.detectAndCompute(grayL, None)
    kpR, desR = orb.detectAndCompute(grayR, None)

    if desL is None or desR is None:
        return None, None

    matches = matcher.knnMatch(desL, desR, k=2)

    good = []

    for m, n in matches:
        if m.distance < 0.90 * n.distance:
            good.append(m)

    if len(good) < 8:
        return None, None

    good = sorted(good, key=lambda x: x.distance)

    # deixa o RANSAC trabalhar com bastante ponto
    good = good[:300]

    ptsL = np.float32([kpL[m.queryIdx].pt for m in good])
    ptsR = np.float32([kpR[m.trainIdx].pt for m in good])

    return ptsL, ptsR


def select_12_spread_points(ptsL, ptsR):

    if len(ptsL) <= 12:
        return ptsL, ptsR

    selected_idx = [0]

    while len(selected_idx) < 12:

        best_idx = None
        best_dist = -1

        for i in range(len(ptsL)):

            if i in selected_idx:
                continue

            dist = min(
                np.linalg.norm(ptsL[i] - ptsL[j])
                for j in selected_idx
            )

            if dist > best_dist:
                best_dist = dist
                best_idx = i

        selected_idx.append(best_idx)

    return ptsL[selected_idx], ptsR[selected_idx]


def compute_rectification(ptsL, ptsR, size):

    F, mask = cv2.findFundamentalMat(
        ptsL,
        ptsR,
        cv2.FM_RANSAC,
        1.0,
        0.99
    )

    if F is None:
        return None

    ptsL = ptsL[mask.ravel() == 1]
    ptsR = ptsR[mask.ravel() == 1]

    if len(ptsL) < 8:
        return None

    ptsL, ptsR = select_12_spread_points(ptsL, ptsR)

    ok, H1, H2 = cv2.stereoRectifyUncalibrated(
        ptsL,
        ptsR,
        F,
        imgSize=size
    )

    if not ok:
        return None

    return {
        "H1": H1,
        "H2": H2,
        "pts_left": ptsL,
        "pts_right": ptsR
    }


for frame_num in range(frame_start, frame_end + 1):

    fname = f"frame_{frame_num:04d}.jpg"

    left_path = os.path.join(LEFT_DIR, fname)
    right_path = os.path.join(RIGHT_DIR, fname)

    if not os.path.exists(left_path):
        print(f"{fname}: esquerda não encontrada")
        continue

    if not os.path.exists(right_path):
        print(f"{fname}: direita não encontrada")
        continue

    imgL = cv2.imread(left_path)
    imgR = cv2.imread(right_path)

    if imgL is None or imgR is None:
        print(f"{fname}: erro ao carregar")
        continue

    h, w = imgL.shape[:2]

    ptsL, ptsR = find_matches(imgL, imgR)

    if ptsL is None:
        print(f"{fname}: matches insuficientes")
        continue

    result = compute_rectification(
        ptsL,
        ptsR,
        (w, h)
    )

    if result is None:
        print(f"{fname}: falha na retificação")
        continue

    H1 = result["H1"]
    H2 = result["H2"]

    rectL = cv2.warpPerspective(imgL, H1, (w, h))
    rectR = cv2.warpPerspective(imgR, H2, (w, h))

    base_name = f"frame_{frame_num:04d}"

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, f"{base_name}_left_rect.jpg"),
        rectL
    )

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, f"{base_name}_right_rect.jpg"),
        rectR
    )

    coords = {
        "pts_left": result["pts_left"].tolist(),
        "pts_right": result["pts_right"].tolist()
    }

    with open(
        os.path.join(OUTPUT_DIR, f"{base_name}_coords.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(coords, f, indent=4)

    print(
        f"{base_name}: {len(result['pts_left'])} pontos salvos"
    )