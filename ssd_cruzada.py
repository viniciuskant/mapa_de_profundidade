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

# --------------------------------------------------------------
# FUNÇÃO COM CORRELAÇÃO NORMALIZADA (ZNCC) E IMAGEM INTEGRAL
# --------------------------------------------------------------
def box_sum_from_integral(integral, half, h, w):
    """
    Retorna uma matriz (h-2*half) x (w-2*half) onde cada elemento 
    é a soma da janela centrada em (y, x) na imagem original.
    """
    return (integral[2*half+1:h+1, 2*half+1:w+1] -
            integral[0:h-2*half, 2*half+1:w+1] -
            integral[2*half+1:h+1, 0:w-2*half] +
            integral[0:h-2*half, 0:w-2*half])

def compute_disparity_zncc(left, right, num_disparities=128, win_size=3):
    """
    Calcula o mapa de disparidade usando Zero-mean Normalized Cross-Correlation (ZNCC).
    Ref: d* = arg max { w_L · w_R(d) }  (com vetores normalizados)
    Utiliza imagens integrais para garantir alta performance.
    """
    left = left.astype(np.float64)   # Precisão dupla para evitar overflow
    right = right.astype(np.float64)
    h, w = left.shape
    half = win_size // 2
    N = win_size * win_size

    # --- Pré-cálculo constante para a imagem esquerda (soma e soma dos quadrados) ---
    integral_L = cv2.integral(left, sdepth=cv2.CV_64F)
    integral_L2 = cv2.integral(left**2, sdepth=cv2.CV_64F)

    # Somas para todas as janelas válidas da esquerda (shape: h-2, w-2)
    sum_L = box_sum_from_integral(integral_L, half, h, w)
    sum_L2 = box_sum_from_integral(integral_L2, half, h, w)

    # Variância da janela esquerda (usando fórmula: E[X^2] - E[X]^2)
    var_L = (sum_L2 / N) - (sum_L / N) ** 2
    var_L = np.maximum(var_L, 1e-8)  # Evita divisão por zero

    # Inicializa o mapa de disparidade e a melhor pontuação (NCC máximo = 1)
    disp_map = np.zeros((h, w), dtype=np.float32)
    best_score = np.full((h - 2*half, w - 2*half), -np.inf, dtype=np.float64)

    # Loop pelas disparidades
    for d in range(num_disparities):
        # Desloca a imagem direita para a esquerda em 'd' pixels: R(u - d, v)
        R_shift = np.zeros_like(right)
        if d == 0:
            R_shift = right.copy()
        else:
            R_shift[:, d:] = right[:, :-d]

        # Integrais para a imagem deslocada e para o produto L * R_shift
        integral_R = cv2.integral(R_shift, sdepth=cv2.CV_64F)
        integral_R2 = cv2.integral(R_shift**2, sdepth=cv2.CV_64F)
        integral_LR = cv2.integral(left * R_shift, sdepth=cv2.CV_64F)

        # Somas das janelas para a disparidade atual (vetorizado)
        sum_R = box_sum_from_integral(integral_R, half, h, w)
        sum_R2 = box_sum_from_integral(integral_R2, half, h, w)
        sum_LR = box_sum_from_integral(integral_LR, half, h, w)

        # Médias
        mean_R = sum_R / N
        mean_L = sum_L / N

        # Covariância: E[L*R] - E[L]*E[R]
        cov = (sum_LR / N) - (mean_L * mean_R)

        # Variância da janela direita
        var_R = (sum_R2 / N) - (mean_R ** 2)
        var_R = np.maximum(var_R, 1e-8)

        # Coeficiente de Correlação Normalizada (ZNCC)
        # Equivalente a w_L · w_R(d) no espaço normalizado
        ncc = cov / np.sqrt(var_L * var_R)

        # Invalida disparidades onde a janela direita ultrapassa a borda esquerda
        # (condição: x - d - half >= 0  =>  x >= d + half)
        # No espaço recortado (sem as bordas), o índice x_local = x - half, então inválido se x_local < d
        if d > 0:
            ncc[:, :d] = -np.inf

        # Atualiza o melhor score e disparidade (maximiza a correlação)
        mask = ncc > best_score
        best_score[mask] = ncc[mask]
        # Atualiza a região central da imagem de disparidade
        disp_map[half:h-half, half:w-half][mask] = float(d)

    return disp_map
# --------------------------------------------------------------

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

# Loop sobre os frames
for frame_num in range(frame_start, frame_end + 1):
    fname = f"{base_name}_{frame_num:04d}.jpg"
    print(f"Processando {fname}")

    imgL = cv2.imread(os.path.join(left_dir, fname))
    imgR = cv2.imread(os.path.join(right_dir, fname))

    if imgL is None or imgR is None:
        continue

    # Filtro de média (kernel 5x5)
    imgL = cv2.blur(imgL, (7, 7))
    imgR = cv2.blur(imgR, (7, 7))


    # Retificação
    rectL = cv2.warpPerspective(imgL, best_H1, size)
    rectR = cv2.warpPerspective(imgR, best_H2, size)

    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    # ---------------------------------------------------------
    # Cálculo da disparidade usando ZNCC com janela 3x3
    # ---------------------------------------------------------
    disp = compute_disparity_zncc(grayL, grayR, num_disparities=8, win_size=3)

    # Suavização opcional para reduzir ruído
    disp = cv2.bilateralFilter(disp, d=7, sigmaColor=25, sigmaSpace=25)

    # Normalização para visualização
    disp_vis = normalize_disp(disp)

    # Transforma a disparidade de volta para a perspectiva original
    H1_inv = np.linalg.inv(best_H1)
    disp_original_view = cv2.warpPerspective(
        disp, H1_inv, size, flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Suavizar bordas
    disp_original_view = cv2.bilateralFilter(disp_original_view, d=7, sigmaColor=25, sigmaSpace=25)

    # Normaliza para visualização
    disp_vis_original = normalize_disp(disp_original_view)
    cv2.imwrite(os.path.join(mapa_profundidade_dir, f"disparity_original_{frame_num:04d}.png"), disp_vis_original)

    # Gera imagem de retificação com linhas horizontais
    combined = np.hstack([draw_horizontal_lines(rectL.copy()), draw_horizontal_lines(rectR.copy())])
    cv2.imwrite(os.path.join(retificacao_dir, f"{base_name}_{frame_num:04d}_rect.png"), combined)