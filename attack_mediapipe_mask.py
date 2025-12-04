import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image

import numpy as np
import cv2
import mediapipe as mp

from facenet_pytorch import InceptionResnetV1


# Pontos do MediaPipe FaceMesh fornecidos
MASK_LANDMARKS = {
    "top_left": 68,
    "top_right": 298,
    "bottom_left": 187,
    "bottom_right": 411,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ataque 2D com máscara baseada em landmarks do MediaPipe FaceMesh."
    )
    parser.add_argument("--attacker", type=str, required=True,
                        help="Caminho da imagem do atacante.")
    parser.add_argument("--victim", type=str, required=True,
                        help="Caminho da imagem da vítima.")
    parser.add_argument("--output", type=str, default="adv_attacker_mediapipe.png",
                        help="Caminho para salvar a imagem adversária final.")
    parser.add_argument("--image-size", type=int, default=160,
                        help="Tamanho de entrada (Facenet-PyTorch usa 160x160).")
    parser.add_argument("--steps", type=int, default=300,
                        help="Número de iterações de otimização.")
    parser.add_argument("--frames", type=int, default=10,
                        help="Quantidade de frames do atacante para a timeline.")
    parser.add_argument("--lr", type=float, default=0.05,
                        help="Learning rate do otimizador.")
    parser.add_argument("--reg-weight", type=float, default=0.01,
                        help="Peso da regularização L2 na perturbação.")
    parser.add_argument("--max-eps", type=float, default=0.3,
                        help="Máximo de variação por canal (em [0,1]) na perturbação.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda ou cpu.")
    return parser.parse_args()


def load_image_and_np(path: str, image_size: int, device: torch.device):
    """
    Carrega imagem, redimensiona para (image_size, image_size) e retorna:
    - tensor [C,H,W] em [0,1] no device
    - array numpy [H,W,3] uint8 (RGB) para uso com MediaPipe.
    """
    img = Image.open(path).convert("RGB")
    img_resized = img.resize((image_size, image_size))

    transform = transforms.ToTensor()
    img_t = transform(img_resized).to(device)  # [C,H,W], 0-1

    img_np = np.array(img_resized)  # [H,W,3], RGB, uint8
    return img_t, img_np

def camera_capture_simulation(img, blur_ksize=7, noise_std=10):
    """
    Tc simplificado:
    - img: tensor [3,H,W] em [0,1]
    - blur_ksize: tamanho (ímpar) do kernel de blur
    - noise_std: desvio padrão do ruído gaussiano

    Tudo é fixo ao longo do loop (como no paper).
    """

    x = torch.clamp(img, 0.0, 1.0)

    # 1) Desfoque leve (blur) via média
    if blur_ksize > 1:
        pad = blur_ksize // 2
        kernel = torch.ones(1, 1, blur_ksize, blur_ksize, device=x.device) / (blur_ksize ** 2)
        # aplica em cada canal separadamente
        x_ch = x.unsqueeze(0)  # [1,3,H,W]
        x_blurred = []
        for c in range(3):
            xc = x_ch[:, c:c+1, :, :]
            xc = F.conv2d(xc, kernel, padding=pad)
            x_blurred.append(xc)
        x = torch.cat(x_blurred, dim=1).squeeze(0)  # [3,H,W]

    # 2) Ruído gaussiano leve
    if noise_std > 0:
        noise = torch.randn_like(x) * noise_std
        x = x + noise

    x = torch.clamp(x, 0.0, 1.0)
    return x


def lift_midtones_sigmoid(x, strength=0.5):
    """
    Aumenta brilho de forma suave usando logit + sigmoid.
    strength > 0 clareia, < 0 escurece.
    """
    eps = 1e-6
    x = torch.clamp(x, eps, 1 - eps)
    x = torch.log(x / (1 - x))         # logit
    x = x + strength                   # desloca curva (brilho)
    x = torch.sigmoid(x)               # volta pra [0,1]
    return x


def light_reflection_simplified(base_img, pattern,
                                alpha=0,
                                brightness=1.5,
                                contrast=1.0,
                                gamma=1.0):
    """
    LRF simplificado com 'opacidade':
    - base_img: tensor [3,H,W] em [0,1] (rosto do atacante sem projeção)
    - pattern:  tensor [3,H,W] em ~[-eps, eps] (perturbação / luz projetada)
    - alpha:    quanta 'força' da luz projetada aparece (0=nenhuma, 1=só padrão)
    - brightness, contrast, gamma: ajustes leves opcionais

    Retorna tensor [3,H,W] em [0,1].
    """

    # 1) Simula o rosto iluminado (base + padrão)
    x = base_img + pattern
    x = torch.clamp(x, 0.0, 1.0)

    # 2) Ajuste de brilho/contraste (opcional, pode deixar 1.0 se quiser neutro)
    x = (x - 0.5) * contrast + 0.5
    x = lift_midtones_sigmoid(x, strength=brightness)
    x = torch.clamp(x, 0.0, 1.0)

    # 3) Gamma (opcional)
    if gamma != 1.0:
        x = torch.clamp(x, 1e-6, 1.0)
        x = x ** (1.0 / gamma)
        x = torch.clamp(x, 0.0, 1.0)

    # 4) Mistura com o rosto original, usando 'opacidade' alpha
    #    alpha=0 -> só base_img
    #    alpha=1 -> só x (base+pattern pós-ajustes)
    x_out = (1.0 - alpha) * base_img + alpha * x
    x_out = torch.clamp(x_out, 0.0, 1.0)
    return x_out


def get_quad_from_mediapipe(image_rgb_np: np.ndarray):
    """
    Executa MediaPipe FaceMesh em image_rgb_np (H,W,3, RGB, uint8)
    e retorna a lista de quatro pontos (x,y) correspondentes aos índices
    definidos em MASK_LANDMARKS.

    Se nenhum rosto for detectado, levanta RuntimeError.
    """
    H, W, _ = image_rgb_np.shape
    mp_face_mesh = mp.solutions.face_mesh

    img_mp = image_rgb_np.copy()

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(img_mp)

    if not results.multi_face_landmarks:
        raise RuntimeError("Nenhum rosto detectado pelo MediaPipe FaceMesh.")

    face_landmarks = results.multi_face_landmarks[0].landmark

    def get_xy(idx: int):
        lm = face_landmarks[idx]
        x = int(lm.x * W)
        y = int(lm.y * H)
        x = max(0, min(W - 1, x))
        y = max(0, min(H - 1, y))
        return x, y

    x_tl, y_tl = get_xy(MASK_LANDMARKS["top_left"])
    x_tr, y_tr = get_xy(MASK_LANDMARKS["top_right"])
    x_bl, y_bl = get_xy(MASK_LANDMARKS["bottom_left"])
    x_br, y_br = get_xy(MASK_LANDMARKS["bottom_right"])

    pts = np.array(
        [
            [x_tl, y_tl],
            [x_tr, y_tr],
            [x_br, y_br],
            [x_bl, y_bl],
        ],
        dtype=np.int32,
    )
    return pts  # shape (4,2)


def mask_from_quad(pts: np.ndarray, H: int, W: int, device: torch.device):
    """
    Cria uma máscara [1,H,W] com 1 dentro do polígono pts e 0 fora.
    pts: np.ndarray (4,2) com coordenadas inteiras (x,y).
    """
    mask_np = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask_np, [pts], 1)
    mask_t = torch.from_numpy(mask_np).float().unsqueeze(0).to(device)  # [1,H,W]
    return mask_t


def draw_quad_debug(image_rgb_np: np.ndarray, pts: np.ndarray, path: Path):
    """
    Desenha o quadrilátero pts na imagem e salva para debug.
    """
    img_bgr = cv2.cvtColor(image_rgb_np.copy(), cv2.COLOR_RGB2BGR)
    cv2.polylines(img_bgr, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.imwrite(str(path), img_bgr)


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Usando device: {device}")

    # 1. Carregar modelo de face (embedding)
    print("[INFO] Carregando modelo InceptionResnetV1 (FaceNet)...")
    model = InceptionResnetV1(pretrained='vggface2', classify=False).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    # 2. Carregar imagens (atacante e vítima)
    print("[INFO] Carregando imagens...")
    attacker_img, attacker_np = load_image_and_np(args.attacker, args.image_size, device)
    victim_img, victim_np = load_image_and_np(args.victim, args.image_size, device)

    _, H, W = attacker_img.shape

    # 3. MediaPipe em atacante e vítima
    print("[INFO] Extraindo landmarks com MediaPipe (atacante e vítima)...")
    attacker_pts = get_quad_from_mediapipe(attacker_np)  # (4,2)
    victim_pts = get_quad_from_mediapipe(victim_np)      # (4,2)

    # Máscara só no atacante
    mask = mask_from_quad(attacker_pts, H, W, device)  # [1,H,W]
    mask_3c = mask.expand_as(attacker_img)  # [3,H,W]

    # 3.1 Salvar debug das máscaras desenhadas
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dbg_attacker_mask_path = out_path.with_name(out_path.stem + "_mask_attacker.png")
    dbg_victim_mask_path = out_path.with_name(out_path.stem + "_mask_victim.png")
    draw_quad_debug(attacker_np, attacker_pts, dbg_attacker_mask_path)
    draw_quad_debug(victim_np, victim_pts, dbg_victim_mask_path)
    print(f"[INFO] Máscara desenhada salva (atacante): {dbg_attacker_mask_path.resolve()}")
    print(f"[INFO] Máscara desenhada salva (vítima):   {dbg_victim_mask_path.resolve()}")

    # 4. Calcular embedding fixo da vítima
    victim_emb = model(victim_img.unsqueeze(0))  # [1,512]
    victim_emb = F.normalize(victim_emb, p=2, dim=1).detach()

    # 5. Inicializar perturbação (delta) somente na região da máscara
    delta = torch.zeros_like(attacker_img, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([delta], lr=args.lr)

    # 6. Steps para snapshots (timeline)
    frames = max(1, args.frames)
    if frames == 1:
        snapshot_steps = {args.steps}
    else:
        snapshot_steps = set()
        for i in range(frames):
            s = int(round(1 + i * (args.steps - 1) / (frames - 1)))
            s = max(1, min(args.steps, s))
            snapshot_steps.add(s)
    snapshot_steps = sorted(snapshot_steps)
    print(f"[INFO] Steps para snapshots: {snapshot_steps}")

    progress_imgs = []

    print("[INFO] Iniciando otimização...")
    for step in range(1, args.steps + 1):
        pattern = delta * mask_3c   # [3,H,W]
        # 2) LRF simplificado (projetor + pele)
        region_lrf = light_reflection_simplified(attacker_img, pattern)

        # 3) Reconstrói a imagem adversária no rosto
        #    fora da máscara: atacante original
        #    dentro da máscara: região passada pela LRF
        adv_img = attacker_img * (1.0 - mask_3c) + region_lrf * mask_3c
        adv_img = torch.clamp(adv_img, 0.0, 1.0)

        # 4) Camera Capture Simulation (Tc)
        adv_img_cam = camera_capture_simulation(adv_img)

        # 5) Passa no modelo de face usando a imagem depois da câmera
        adv_emb = model(adv_img_cam.unsqueeze(0))
        adv_emb = F.normalize(adv_emb, p=2, dim=1)

        # --- daqui pra baixo continua igual ---
        cosine_sim = (adv_emb * victim_emb).sum(dim=1)
        loss_sim = -cosine_sim.mean()
        loss_reg = torch.mean(delta ** 2)
        loss = loss_sim + args.reg_weight * loss_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            delta.data = torch.clamp(delta.data, -args.max_eps, args.max_eps)


        if step % 20 == 0 or step == 1 or step == args.steps:
            print(
                f"[STEP {step:04d}] "
                f"Loss total: {loss.item():.4f} | "
                f"Cosine sim: {cosine_sim.item():.4f} | "
                f"Reg: {loss_reg.item():.6f}"
            )

        if step in snapshot_steps:
            progress_imgs.append(adv_img.detach().cpu())

    # 7. Imagem adversária final
    adv_final = torch.clamp(attacker_img + delta * mask_3c, 0.0, 1.0).detach().cpu()
    save_image(adv_final, str(out_path))
    print(f"[INFO] Imagem adversária final salva em: {out_path.resolve()}")

    # 8. Timeline: vítima + frames do atacante
    gallery_imgs = [victim_img.detach().cpu()] + progress_imgs
    all_imgs = torch.stack(gallery_imgs, dim=0)  # [N,3,H,W]

    grid = make_grid(all_imgs, nrow=len(gallery_imgs), padding=2)
    timeline_path = out_path.with_name(out_path.stem + "_timeline.png")
    save_image(grid, str(timeline_path))
    print(f"[INFO] Timeline salva em: {timeline_path.resolve()}")


if __name__ == "__main__":
    main()
