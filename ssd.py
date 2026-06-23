import cv2
import numpy as np
import os
import glob
import json
import shutil

PATH_TESTE = "testes/teste_14.json"

base_name   = "frame"
mapa_profundidade_dir  = "mapa_profundidade"
retificacao_dir  = "retificacao"

pastas_para_limpar = [
    mapa_profundidade_dir, 
    retificacao_dir
]

for pasta in pastas_para_limpar:
    if os.path.exists(pasta):
        shutil.rmtree(pasta)
    os.makedirs(pasta, exist_ok=True) 

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

    F, mask = cv2.findFundamentalMat(ptsL, ptsR, cv2.FM_RANSAC, 1.0, 0.99)
    if F is None:
        return np.inf, None, None

    ptsL = ptsL[mask.ravel() == 1]
    ptsR = ptsR[mask.ravel() == 1]

    ok, H1, H2 = cv2.stereoRectifyUncalibrated(ptsL, ptsR, F, imgSize=size)

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

def box_sum_from_integral(integral, half, h, w):
    return (integral[2*half+1:h+1, 2*half+1:w+1] -
            integral[0:h-2*half, 2*half+1:w+1] -
            integral[2*half+1:h+1, 0:w-2*half] +
            integral[0:h-2*half, 0:w-2*half])

def compute_disparity_ssd_fast(left, right, num_disparities=128, win_size=3):
    left = left.astype(np.float32)
    right = right.astype(np.float32)
    h, w = left.shape
    half = win_size // 2

    # imagem integral
    L2 = left ** 2
    integral_L2 = cv2.integral(L2, sdepth=cv2.CV_64F)
    sum_L2_all = box_sum_from_integral(integral_L2, half, h, w)

    # init mapa e custo mínimo
    disp_map = np.zeros((h, w), dtype=np.float32)
    best_cost = np.full((h, w), np.inf, dtype=np.float64)

    # apenas centros com janela totalmente dentro da imagem esquerda
    y_start, y_end = half, h - half
    x_start, x_end = half, w - half

    for d in range(num_disparities):
        # desloca a imagem direita para a esquerda em 'd' pixels
        # formula usa I_R(u - d, v)
        # u < d, a janela sairia da borda
        R_shift = np.zeros_like(right)
        if d == 0:
            R_shift = right.copy()
        else:
            R_shift[:, d:] = right[:, :-d] 

        # I_R^2 e I_L * I_R
        R2_shift = R_shift ** 2
        LR = left * R_shift

        # disparidade atual
        integral_R2 = cv2.integral(R2_shift, sdepth=cv2.CV_64F)
        integral_LR = cv2.integral(LR, sdepth=cv2.CV_64F)

        # soma das janelas para todos os centros válidos
        sum_R2_all = box_sum_from_integral(integral_R2, half, h, w)
        sum_LR_all = box_sum_from_integral(integral_LR, half, h, w)

        #sum(L^2) + sum(R^2) - 2*sum(L*R)
        cost = sum_L2_all + sum_R2_all - 2.0 * sum_LR_all

        if d > 0: # descarta disparidades inválidas
            cost[:, :d] = np.inf 

        # atualiza o mapa de disparidade (mantém o menor custo)
        region_cost = best_cost[half:h-half, half:w-half]
        mask = cost < region_cost
        region_cost[mask] = cost[mask]
        disp_region = disp_map[half:h-half, half:w-half]
        disp_region[mask] = float(d)

    return disp_map

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

print(len(currentL), "pontos finais escolhidos")
print("Erro final:", best_err)

for frame_num in range(frame_start, frame_end + 1):
    fname = f"{base_name}_{frame_num:04d}.jpg"
    print(f"Processando {fname}")

    imgL = cv2.imread(os.path.join(left_dir, fname))
    imgR = cv2.imread(os.path.join(right_dir, fname))

    if imgL is None or imgR is None:
        continue

    # retificação
    rectL = cv2.warpPerspective(imgL, best_H1, size)
    rectR = cv2.warpPerspective(imgR, best_H2, size)

    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    #janela 3x3
    disp = compute_disparity_ssd_fast(grayL, grayR, num_disparities=8, win_size=15)

    # para reduzir ruído
    disp = cv2.bilateralFilter(disp, d=7, sigmaColor=25, sigmaSpace=25)

    #volta para a perspectiva original
    H1_inv = np.linalg.inv(best_H1)
    disp_original_view = cv2.warpPerspective(disp, H1_inv, size, flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    #suavizar bordas e normaliza para visualização
    disp_original_view = cv2.bilateralFilter(disp_original_view, d=7, sigmaColor=25, sigmaSpace=25)
    disp_vis_original = normalize_disp(disp_original_view)
    cv2.imwrite(os.path.join(mapa_profundidade_dir, f"disparity_original_{frame_num:04d}.png"), disp_vis_original)

    combined = np.hstack([draw_horizontal_lines(rectL.copy()), draw_horizontal_lines(rectR.copy())])
    cv2.imwrite(os.path.join(retificacao_dir, f"{base_name}_{frame_num:04d}_rect.png"), combined)