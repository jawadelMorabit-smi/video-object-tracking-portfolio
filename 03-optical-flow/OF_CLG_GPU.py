import numpy as np
import cv2
import torch


print(f"PyTorch version: {torch.__version__}")

# Check if GPU is available
if torch.cuda.is_available():
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
else:
    print("GPU is NOT available. Running on CPU.")
device = "cpu"
#torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def to_tensor(img):
    """Convertit image OpenCV → torch.float32 [0,1] sur GPU"""
    return torch.from_numpy(img).float().div(255.).to(device)

def build_pyramid(img, levels):
    """Construit pyramide coarse to fine"""
    pyr = [img]
    for _ in range(1, levels):
        img = torch.nn.functional.avg_pool2d(img.unsqueeze(0).unsqueeze(0),
                                             kernel_size=2, stride=2,
                                             ceil_mode=True).squeeze()
        pyr.append(img)
    return pyr[::-1]   # coarse → fine

def warp(img, u, v):
    """Warp bilinéaire : img(x+u, y+v)"""
    H, W = img.shape
    y, x = torch.meshgrid(torch.arange(H, device=device),
                          torch.arange(W, device=device),
                          indexing='ij')
    grid_x = 2 * (x + u) / (W - 1) - 1
    grid_y = 2 * (y + v) / (H - 1) - 1
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)

    img_t = img.unsqueeze(0).unsqueeze(0)
    warped = torch.nn.functional.grid_sample(img_t, grid, mode='bilinear', align_corners=True)
    return warped.squeeze()

def gradients(I1, I2w):
    """Gradients spatiaux + temporel"""
    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], device=device).float().view(1,1,3,3)/8
    sobel_y = sobel_x.transpose(2,3)
    I1_4d = I1.unsqueeze(0).unsqueeze(0)
    Ix = torch.nn.functional.conv2d(I1_4d, sobel_x, padding=1).squeeze()
    Iy = torch.nn.functional.conv2d(I1_4d, sobel_y, padding=1).squeeze()
    It = I2w - I1
    return Ix, Iy, It

def horn_schunck(Ix, Iy, It, alpha=40, iterations=40):
    """Solveur Horn–Schunck GPU"""
    H, W = Ix.shape
    u = torch.zeros((H, W), device=device)
    v = torch.zeros((H, W), device=device)

    avg_kernel = torch.tensor([[1,1,1],[1,0,1],[1,1,1]], device=device).float().view(1,1,3,3)/8

    for _ in range(iterations):
        u_avg = torch.nn.functional.conv2d(u.unsqueeze(0).unsqueeze(0), avg_kernel, padding=1).squeeze()
        v_avg = torch.nn.functional.conv2d(v.unsqueeze(0).unsqueeze(0), avg_kernel, padding=1).squeeze()
        P = Ix * u_avg + Iy * v_avg + It
        D = alpha*alpha + Ix*Ix + Iy*Iy
        u = u_avg - Ix * P / D
        v = v_avg - Iy * P / D

    return u, v

def optical_flow_pytorch(I1, I2, levels=5, alpha=40):
    """Flot optique coarse-to-fine GPU"""
    pyr1 = build_pyramid(I1, levels)
    pyr2 = build_pyramid(I2, levels)
    H0, W0 = pyr1[0].shape
    u = torch.zeros((H0, W0), device=device)
    v = torch.zeros((H0, W0), device=device)

    for lvl in range(levels):
        I1_lvl = pyr1[lvl]
        I2_lvl = pyr2[lvl]

        # 1) warp
        I2w = warp(I2_lvl, u, v)

        # 2) gradients
        Ix, Iy, It = gradients(I1_lvl, I2w)

        # 3) solveur variationnel
        du, dv = horn_schunck(Ix, Iy, It, alpha=alpha, iterations=30)
        u += du
        v += dv

        # 4) upsample
        if lvl < levels - 1:
            u = torch.nn.functional.interpolate(u.unsqueeze(0).unsqueeze(0),
                                                size=pyr1[lvl+1].shape,
                                                mode='bilinear', align_corners=True).squeeze()
            v = torch.nn.functional.interpolate(v.unsqueeze(0).unsqueeze(0),
                                                size=pyr1[lvl+1].shape,
                                                mode='bilinear', align_corners=True).squeeze()
    return u, v

cap = cv2.VideoCapture("personnes_en_mouvement.mp4")
ret, frame_prev = cap.read()
prev_gray = to_tensor(cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    # calcul flot optique GPU
    u, v = optical_flow_pytorch(prev_gray, gray, levels=4, alpha=40)

    # visualisation HSV
    mag = torch.sqrt(u*u + v*v)
    ang = torch.atan2(v, u)
    mag_n = (mag / (mag.max() + 1e-6) * 255).cpu().numpy().astype('uint8')
    ang_n = ((ang + 3.1416) * 180 / 3.1416 / 2).cpu().numpy().astype('uint8')

    hsv = cv2.merge([cv2.resize(ang_n, (frame.shape[1], frame.shape[0])),
                     np.full_like(ang_n, 255),
                     cv2.resize(mag_n, (frame.shape[1], frame.shape[0]))])
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow("Flot optique variationnel GPU (Bruhn)", flow_rgb)

    prev_gray = gray

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
