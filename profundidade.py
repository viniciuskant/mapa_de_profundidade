import cv2
import numpy as np
import os
import glob
import json

PATH_TESTE = "testes/teste_01.json"

base_name   = "frame"
mapa_profundidade_dir  = "mapa_profundidade"
retificacao_dir  = "retificacao"
os.makedirs(mapa_profundidade_dir, exist_ok=True)
os.makedirs(retificacao_dir, exist_ok=True)

def load_stereo_config(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    return {
        "frame_start": cfg["frame_start"],
        "frame_end": cfg["frame_end"],
        "left_dir": cfg["left_dir"],
        "right_dir": cfg["right_dir"],
        "pts_left": np.array(cfg["pts_left"], dtype=np.float32),
        "pts_right": np.array(cfg["pts_right"], dtype=np.float32),
    }

def rectification_error(ptsL, ptsR, size):
    if len(ptsL) < 8:
        return np.inf, None, None

    F, mask = cv2.findFundamentalMat(
        ptsL, ptsR,
        cv2.FM_RANSAC,
        1.0, 0.99
    )

    if F is None:
        return np.inf, None, None

    ptsL = ptsL[mask.ravel() == 1]
    ptsR = ptsR[mask.ravel() == 1]

    ok, H1, H2 = cv2.stereoRectifyUncalibrated(
        ptsL, ptsR, F, imgSize=size
    )

    if not ok:
        return np.inf, None, None

    pL = cv2.perspectiveTransform(ptsL.reshape(-1, 1, 2), H1).reshape(-1, 2)

    pR = cv2.perspectiveTransform(ptsR.reshape(-1, 1, 2), H2).reshape(-1, 2)

    err = np.mean(np.abs(pL[:, 1] - pR[:, 1]))

    return err, H1, H2

def draw_horizontal_lines(img, step=40):
    h, w = img.shape[:2]
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), (0, 255, 0), 1)
    return img

def normalize_disp(disp):
    disp = disp.astype(np.float32)
    valid = disp > 0

    if np.any(valid):
        mn = disp[valid].min()
        mx = disp[valid].max()

        disp = (disp - mn) / (mx - mn + 1e-6)
        disp *= 255
    return disp.astype(np.uint8)

cfg = load_stereo_config(PATH_TESTE)

frame_start = cfg["frame_start"]
frame_end = cfg["frame_end"]

left_dir = cfg["left_dir"]
right_dir = cfg["right_dir"]

pts_left = cfg["pts_left"]
pts_right = cfg["pts_right"]

sample_img = cv2.imread(sorted(glob.glob(os.path.join(left_dir, "*.jpg")))[0])
h, w = sample_img.shape[:2]
size = (w, h)

currentL = pts_left.copy()
currentR = pts_right.copy()

best_err, best_H1, best_H2 = rectification_error(currentL, currentR, size)

while len(currentL) > 8:
    candidate_err = best_err
    candidate_idx = None

    for i in range(len(currentL)):
        err, H1, H2 = rectification_error(
            np.delete(currentL, i, axis=0),
            np.delete(currentR, i, axis=0),
            size
        )

        if err < candidate_err:
            candidate_err = err
            candidate_idx = i
            candidate_H1 = H1
            candidate_H2 = H2

    if candidate_idx is None:
        break

    currentL = np.delete(currentL, candidate_idx, axis=0)
    currentR = np.delete(currentR, candidate_idx, axis=0)

    best_err = candidate_err
    best_H1 = candidate_H1
    best_H2 = candidate_H2

print(len(currentL), "pontos finais escolhidos")
print("Erro final:", best_err)

stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,
    blockSize=11,
    P1=8 * 3 * 11**2,
    P2=32 * 3 * 11**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

right_matcher = cv2.ximgproc.createRightMatcher(stereo)

wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
wls_filter.setLambda(8000)
wls_filter.setSigmaColor(1.5)

for frame_num in range(frame_start, frame_end + 1):
    fname = f"{base_name}_{frame_num:04d}.jpg"
    print(f"Processando {fname}")

    imgL = cv2.imread(os.path.join(left_dir, fname))
    imgR = cv2.imread(os.path.join(right_dir, fname))

    if imgL is None or imgR is None:
        continue

    rectL = cv2.warpPerspective(imgL, best_H1, size)
    rectR = cv2.warpPerspective(imgR, best_H2, size)

    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    dispL = stereo.compute(grayL, grayR)
    dispR = right_matcher.compute(grayR, grayL)

    filtered_disp = wls_filter.filter(dispL, grayL, None, dispR)

    filtered_disp = filtered_disp.astype(np.float32) / 16.0

    filtered_disp[filtered_disp < 0] = 0

    filtered_disp = cv2.bilateralFilter(filtered_disp, d=7, sigmaColor=25, sigmaSpace=25)
    disp_vis = normalize_disp(filtered_disp)

    H1_inv = np.linalg.inv(best_H1)

    # volta para a imagem original
    disp_original_view = cv2.warpPerspective(
        filtered_disp, H1_inv, size, flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # suavizar bordas 
    disp_original_view = cv2.bilateralFilter(disp_original_view, d=7, sigmaColor=25, sigmaSpace=25)

    # normaliza para visualização
    disp_vis_original = normalize_disp(disp_original_view)

    cv2.imwrite(os.path.join(mapa_profundidade_dir, f"disparity_original_{frame_num:04d}.png"), disp_vis_original)

    # cv2.imwrite(os.path.join(mapa_profundidade_dir, f"disparity_{frame_num:04d}.png"), disp_vis)

    combined = np.hstack([ draw_horizontal_lines(rectL.copy()), draw_horizontal_lines(rectR.copy())])

    cv2.imwrite(os.path.join(retificacao_dir, f"{base_name}_{frame_num:04d}_rect.png"), combined)

