import cv2
import numpy as np
import math
import sys
# tqdm é opcional, apenas para mostrar a barra de progresso no terminal
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

def computar_homografias_retificacao(frame_esq, frame_dir):
    """Calcula as homografias de retificação usando SIFT e Matriz Fundamental."""
    gray_esq = cv2.cvtColor(frame_esq, cv2.COLOR_BGR2GRAY)
    gray_dir = cv2.cvtColor(frame_dir, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_esq, None)
    kp2, des2 = sift.detectAndCompute(gray_dir, None)

    # Flann matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    pts1, pts2 = [], []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    h, w = gray_esq.shape
    _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), F, imgSize=(w, h))
    
    return H1, H2

def renderizar_vista_45_graus(frame_ref, disparity, Q):
    """Gera uma projeção 2D da nuvem de pontos rotacionada a 45 graus."""
    h, w = frame_ref.shape[:2]
    
    # 1. Reprojetar para 3D (Z = profundidade real)
    pontos_3D = cv2.reprojectImageTo3D(disparity, Q)
    cores = frame_ref
    
    # Filtrar pontos inválidos (disparidade muito pequena ou infinita)
    mask = disparity > 0
    pontos_3D = pontos_3D[mask]
    cores = cores[mask]
    
    # 2. Matriz de Rotação de 45 graus (Eixo Y)
    theta = math.radians(45)
    Ry = np.array([
        [math.cos(theta), 0, math.sin(theta)],
        [0, 1, 0],
        [-math.sin(theta), 0, math.cos(theta)]
    ])
    
    # Aplicar rotação
    pontos_rotacionados = np.dot(pontos_3D, Ry.T)
    
    # Transladar um pouco no eixo Z para a câmera virtual não ficar "dentro" do objeto
    translacao_z = np.mean(pontos_rotacionados[:, 2]) * 0.5
    pontos_rotacionados[:, 2] += translacao_z
    
    # 3. Projetar de volta para 2D usando uma matriz de câmera virtual P
    focal_length = w * 0.8
    P = np.array([
        [focal_length, 0, w/2],
        [0, focal_length, h/2],
        [0, 0, 1]
    ])
    
    x_2d = (pontos_rotacionados[:, 0] / pontos_rotacionados[:, 2]) * P[0,0] + P[0,2]
    y_2d = (pontos_rotacionados[:, 1] / pontos_rotacionados[:, 2]) * P[1,1] + P[1,2]
    
    frame_45 = np.zeros((h, w, 3), dtype=np.uint8)
    
    x_2d = np.int32(np.round(x_2d))
    y_2d = np.int32(np.round(y_2d))
    
    # Desenhar os pixels no novo frame (mantendo limites)
    validos = (x_2d >= 0) & (x_2d < w) & (y_2d >= 0) & (y_2d < h)
    for x, y, cor in zip(x_2d[validos], y_2d[validos], cores[validos]):
        frame_45[y, x] = cor
        
    # Aplicar um filtro de mediana leve para preencher buracos causados pela projeção
    frame_45 = cv2.medianBlur(frame_45, 3)
    return frame_45

def processar_stereo(video_esq_path, video_dir_path):
    cap_esq = cv2.VideoCapture(video_esq_path)
    cap_dir = cv2.VideoCapture(video_dir_path)

    if not cap_esq.isOpened() or not cap_dir.isOpened():
        print("Erro ao abrir os vídeos.")
        return

    # Pegar propriedades
    w = int(cap_esq.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_esq.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_esq.get(cv2.CAP_PROP_FPS)
    frames_totais = int(cap_esq.get(cv2.CAP_PROP_FRAME_COUNT))

    # Configurar Writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_esq_rect = cv2.VideoWriter('Esquerda_Retificado.mp4', fourcc, fps, (w, h))
    out_dir_rect = cv2.VideoWriter('Direita_Retificado.mp4', fourcc, fps, (w, h))
    out_frontal_combinado = cv2.VideoWriter('Frontal_Combinado.mp4', fourcc, fps, (w*2, h))
    out_mapa_prof = cv2.VideoWriter('Mapa_Profundidade.mp4', fourcc, fps, (w, h), isColor=False)
    out_vista_45 = cv2.VideoWriter('Vista_45_Graus.mp4', fourcc, fps, (w, h))

    # 1. Obter Matrizes de Retificação pelo primeiro frame
    ret_e, frame_e_ini = cap_esq.read()
    ret_d, frame_d_ini = cap_dir.read()
    if not (ret_e and ret_d): return
    
    print("Calculando matrizes de retificação...")
    H_esq, H_dir = computar_homografias_retificacao(frame_e_ini, frame_d_ini)

    # Configuração do Algoritmo de Block Matching SGBM
    # OBRIGATORIAMENTE janela 3x3 (blockSize = 3)
    janela = 3
    num_disparities = 64 # Deve ser divisível por 16
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=janela,
        P1=8 * 3 * janela**2,
        P2=32 * 3 * janela**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Matriz Q Genérica para conversão Disparidade -> Profundidade
    focal = w * 0.8
    baseline = 0.1 # Distância base assumida entre as câmeras
    Q = np.float32([
        [1, 0, 0, -w/2],
        [0, -1, 0, h/2],
        [0, 0, 0, focal],
        [0, 0, -1/baseline, 0]
    ])

    cap_esq.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap_dir.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("Processando frames...")
    for _ in tqdm(range(frames_totais)):
        ret_e, frame_e = cap_esq.read()
        ret_d, frame_d = cap_dir.read()
        if not (ret_e and ret_d):
            break

        # A. Retificação
        frame_e_rect = cv2.warpPerspective(frame_e, H_esq, (w, h))
        frame_d_rect = cv2.warpPerspective(frame_d, H_dir, (w, h))
        
        out_esq_rect.write(frame_e_rect)
        out_dir_rect.write(frame_d_rect)

        # B. Vídeo Frontal Combinado (Lado a Lado)
        frontal_combinado = np.hstack((frame_e_rect, frame_d_rect))
        out_frontal_combinado.write(frontal_combinado)

        # C. Cálculo de Profundidade/Disparidade com janela 3x3
        gray_e = cv2.cvtColor(frame_e_rect, cv2.COLOR_BGR2GRAY)
        gray_d = cv2.cvtColor(frame_d_rect, cv2.COLOR_BGR2GRAY)
        
        disparity = stereo.compute(gray_e, gray_d).astype(np.float32) / 16.0
        
        # D. Mapa de Profundidade em Tons de Cinza (Pixels mais profundos mais claros)
        # Disparidade é inversamente proporcional à profundidade (D = 1/Z).
        # Assim, disparidades baixas (longe/fundo) devem ficar claras.
        disp_normalizada = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        disp_normalizada = np.uint8(disp_normalizada)
        
        # Invertemos a matriz: objetos longe (profundos) que antes eram escuros ficam claros.
        mapa_profundidade_invertido = cv2.bitwise_not(disp_normalizada)
        out_mapa_prof.write(mapa_profundidade_invertido)

        # E. Projeção 3D com Rotação de 45 Graus (Usando a Esquerda como referencial de cor)
        frame_45 = renderizar_vista_45_graus(frame_e_rect, disparity, Q)
        out_vista_45.write(frame_45)

    # Liberação de Memória
    cap_esq.release()
    cap_dir.release()
    out_esq_rect.release()
    out_dir_rect.release()
    out_frontal_combinado.release()
    out_mapa_prof.release()
    out_vista_45.release()
    print("Processamento concluído. Vídeos gerados com sucesso.")

if __name__ == '__main__':
    # Substitua pelos nomes dos seus arquivos locais
    processar_stereo('Esquerda.mp4', 'Direita.mp4')