import os
import gc
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import matplotlib.pyplot as plt
from scipy.special import roots_jacobi, roots_legendre, gamma

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用计算设备: {device}")

if device.type == "cuda":
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


cleanup_cuda()



def in_material_mask(x, y):
    """
    判断点是否在实体材料（围岩）内。返回 False 表示在隧道内部。
    """
    center_x, center_y = 0.5, 0.5
    radius = 0.15
    dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2
    return dist_sq >= (radius ** 2)


def generate_forward_tunnel_data(
        horizon=0.25, t_max=1.0,
        n_edge=21, n_layer=11, n_time_bc=11, n_init_grid=21,
        N_collocation=1200, alpha=0.6, s=-0.3
):
    δ = float(horizon)
    t_list = np.linspace(0.0, t_max, n_time_bc, dtype=np.float32)

    y01 = np.linspace(0.0, 1.0, n_edge, dtype=np.float32)
    x01 = np.linspace(0.0, 1.0, n_edge, dtype=np.float32)

    xL = np.linspace(-δ, 0.0, n_layer, dtype=np.float32)
    xR = np.linspace(1.0, 1.0 + δ, n_layer, dtype=np.float32)
    yB = np.linspace(-δ, 0.0, n_layer, dtype=np.float32)
    yT = np.linspace(1.0, 1.0 + δ, n_layer, dtype=np.float32)

    pts = []
    u_bc_list = []
    v_bc_list = []


    for tt in t_list:

        XX, YY = np.meshgrid(xL, y01, indexing="xy")
        pts.append(np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, tt, np.float32)], axis=1))
        u_bc_list.append(-0.01 * (XX.ravel() - 0.5) * tt)
        v_bc_list.append(-0.01 * (YY.ravel() - 0.5) * tt)

        XX, YY = np.meshgrid(xR, y01, indexing="xy")
        pts.append(np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, tt, np.float32)], axis=1))
        u_bc_list.append(-0.01 * (XX.ravel() - 0.5) * tt)
        v_bc_list.append(-0.01 * (YY.ravel() - 0.5) * tt)

        XX, YY = np.meshgrid(x01, yB, indexing="xy")
        pts.append(np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, tt, np.float32)], axis=1))
        u_bc_list.append(-0.01 * (XX.ravel() - 0.5) * tt)
        v_bc_list.append(-0.01 * (YY.ravel() - 0.5) * tt)

        XX, YY = np.meshgrid(x01, yT, indexing="xy")
        pts.append(np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, tt, np.float32)], axis=1))
        u_bc_list.append(-0.01 * (XX.ravel() - 0.5) * tt)
        v_bc_list.append(-0.01 * (YY.ravel() - 0.5) * tt)

    X_bc = np.concatenate(pts, axis=0).astype(np.float32)
    U_bc = np.concatenate(u_bc_list, axis=0).reshape(-1, 1).astype(np.float32)
    V_bc = np.concatenate(v_bc_list, axis=0).reshape(-1, 1).astype(np.float32)


    x_all = np.linspace(-δ, 1.0 + δ, n_init_grid, dtype=np.float32)
    y_all = np.linspace(-δ, 1.0 + δ, n_init_grid, dtype=np.float32)
    XX0, YY0 = np.meshgrid(x_all, y_all, indexing="xy")
    X_ic_raw = np.stack([XX0.ravel(), YY0.ravel(), np.zeros(XX0.size, np.float32)], axis=1).astype(np.float32)
    ic_mask = in_material_mask(X_ic_raw[:, 0], X_ic_raw[:, 1])
    X_ic = X_ic_raw[ic_mask]
    U_ic = np.zeros((X_ic.shape[0], 1), dtype=np.float32)
    V_ic = np.zeros((X_ic.shape[0], 1), dtype=np.float32)

    X_u = np.concatenate([X_bc, X_ic], axis=0).astype(np.float32)
    U_u = np.concatenate([U_bc, U_ic], axis=0).astype(np.float32)
    V_u = np.concatenate([V_bc, V_ic], axis=0).astype(np.float32)


    xy_f_list = []
    while len(xy_f_list) < N_collocation:
        sample_pts = np.random.uniform(0.0, 1.0, (N_collocation * 2, 2))
        valid_mask = in_material_mask(sample_pts[:, 0], sample_pts[:, 1])
        xy_f_list.extend(sample_pts[valid_mask])
    xy_f = np.array(xy_f_list[:N_collocation], dtype=np.float32)

    t_f = np.random.uniform(1e-3, t_max, N_collocation).reshape(-1, 1).astype(np.float32)
    X_f_raw = np.column_stack([xy_f, t_f]).astype(np.float32)


    bx_true = np.zeros((N_collocation, 1), dtype=np.float32)
    by_true = np.full((N_collocation, 1), -0.05, dtype=np.float32)

    X_f = np.column_stack([X_f_raw, bx_true, by_true]).astype(np.float32)


    nx, ny = 45, 45
    x = np.linspace(0, 1, nx, dtype=np.float32)
    y = np.linspace(0, 1, ny, dtype=np.float32)
    Xg, Yg = np.meshgrid(x, y, indexing="xy")
    X_test_raw = np.column_stack([Xg.ravel(), Yg.ravel(), np.ones(Xg.size, np.float32)]).astype(np.float32)
    test_mask = in_material_mask(X_test_raw[:, 0], X_test_raw[:, 1])
    X_test = X_test_raw[test_mask]

    return X_u, U_u, V_u, X_f, X_test, Xg, Yg



class PINN_Net(nn.Module):
    def __init__(self, layers, lb, ub):
        super().__init__()
        self.lb = torch.tensor(lb, dtype=torch.float32, device=device)
        self.ub = torch.tensor(ub, dtype=torch.float32, device=device)
        self.layers = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        self._init_weights()

    def _init_weights(self):
        for layer in self.layers[:-1]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.layers[-1].weight, gain=0.1)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, X):
        X = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for layer in self.layers[:-1]:
            X = torch.tanh(layer(X))
        out = self.layers[-1](X)
        return out[:, 0:1], out[:, 1:2]


class FractionalPDIntegrator_TFAligned(nn.Module):
    def __init__(self, delta, calpha, alpha, s, num_frac=8, num_r=8, num_theta=8, eps_t=1e-12):
        super().__init__()
        self.delta = float(delta)
        self.alpha = float(alpha)
        self.s = float(s)
        self.eps_t = float(eps_t)
        self.register_buffer("calpha", torch.tensor(float(calpha), dtype=torch.float32, device=device))

        ptau, wtau = roots_jacobi(num_frac, -self.alpha, 0.0)
        self.register_buffer("ptau", torch.tensor(ptau, dtype=torch.float32, device=device))
        self.register_buffer("wtau", torch.tensor(wtau, dtype=torch.float32, device=device))
        self.num_frac = int(num_frac)

        pr, wr = roots_jacobi(num_r, 0.0, -(1.0 + 2.0 * self.s))
        pr = torch.tensor(pr, dtype=torch.float32, device=device)
        wr = torch.tensor(wr, dtype=torch.float32, device=device)
        r = (self.delta / 2.0) * (pr + 1.0)
        self.register_buffer("r_nodes", r)
        self.register_buffer("wr", wr)

        self.r_scale = (self.delta / 2.0) ** (1.0 - (1.0 + 2.0 * self.s))

        pth, wth = roots_legendre(num_theta)
        pth = torch.tensor(pth, dtype=torch.float32, device=device)
        wth = torch.tensor(wth, dtype=torch.float32, device=device)
        theta = np.pi * (pth + 1.0)
        wtheta = np.pi * wth
        theta = theta.to(device)
        wtheta = wtheta.to(device)

        R, TH = torch.meshgrid(self.r_nodes, theta, indexing="ij")
        WR, WTH = torch.meshgrid(self.wr, wtheta, indexing="ij")

        cosT = torch.cos(TH)
        sinT = torch.sin(TH)
        w_spatial = (WR * WTH) * float(self.r_scale)

        self.register_buffer("coef11", (w_spatial * cosT ** 2).flatten())
        self.register_buffer("coef12", (w_spatial * sinT * cosT).flatten())
        self.register_buffer("coef22", (w_spatial * sinT ** 2).flatten())
        self.register_buffer("xi_x", (R * cosT).flatten())
        self.register_buffer("xi_y", (R * sinT).flatten())

        self.num_spatial = int(self.xi_x.numel())
        self.caputo_coeff = float(1.0 / gamma(1.0 - self.alpha))

    @staticmethod
    def _time_derivative(net, x, y, t):
        t_leaf = t.clone().detach().requires_grad_(True)
        U, V = net(torch.cat([x, y, t_leaf], dim=1))
        Ut = grad(U.sum(), t_leaf, create_graph=True, retain_graph=True)[0]
        Vt = grad(V.sum(), t_leaf, create_graph=True, retain_graph=True)[0]
        return Ut, Vt

    def compute_pd_term(self, x, y, t, net):
        B = x.shape[0]
        n_tau = self.num_frac
        n_sp = self.num_spatial

        t_det = t.detach()
        bianhuan = t_det / 2.0

        si = bianhuan * (self.ptau + 1.0).unsqueeze(0)
        time_w = torch.pow(torch.clamp(bianhuan, min=self.eps_t), 1.0 - self.alpha) * self.wtau.unsqueeze(0)

        x_c = x.repeat_interleave(n_tau, dim=0)
        y_c = y.repeat_interleave(n_tau, dim=0)
        t_c = si.reshape(-1, 1)
        Ut_c, Vt_c = self._time_derivative(net, x_c, y_c, t_c)
        Ut_c = Ut_c.view(B, n_tau)
        Vt_c = Vt_c.view(B, n_tau)

        x_nei = x + self.xi_x.unsqueeze(0)
        y_nei = y + self.xi_y.unsqueeze(0)


        nei_mask = in_material_mask(x_nei, y_nei).to(torch.float32).unsqueeze(2)

        x_n = x_nei.unsqueeze(2).expand(B, n_sp, n_tau).reshape(-1, 1)
        y_n = y_nei.unsqueeze(2).expand(B, n_sp, n_tau).reshape(-1, 1)
        t_n = si.unsqueeze(1).expand(B, n_sp, n_tau).reshape(-1, 1)

        Ut_n, Vt_n = self._time_derivative(net, x_n, y_n, t_n)
        Ut_n = Ut_n.view(B, n_sp, n_tau)
        Vt_n = Vt_n.view(B, n_sp, n_tau)

        dUt = Ut_n - Ut_c.unsqueeze(1)
        dVt = Vt_n - Vt_c.unsqueeze(1)

        dUt = dUt * nei_mask
        dVt = dVt * nei_mask

        sum_int_v = torch.sum(time_w.unsqueeze(1) * dUt, dim=2)
        sum_int_w = torch.sum(time_w.unsqueeze(1) * dVt, dim=2)

        coef11 = self.coef11.unsqueeze(0)
        coef12 = self.coef12.unsqueeze(0)
        coef22 = self.coef22.unsqueeze(0)

        sv = torch.sum(coef11 * sum_int_v + coef12 * sum_int_w, dim=1, keepdim=True)
        sw = torch.sum(coef12 * sum_int_v + coef22 * sum_int_w, dim=1, keepdim=True)

        sv = float(self.caputo_coeff) * sv
        sw = float(self.caputo_coeff) * sw

        int_U = self.calpha * float(self.delta) * sv
        int_V = self.calpha * float(self.delta) * sw
        return int_U, int_V


class FractionalPD_PINN(nn.Module):
    def __init__(self, X_u, U_u, V_u, X_f, X_test, layers, lb, ub, horizon, alpha, s, num_frac=8, num_r=8, num_theta=8):
        super().__init__()
        self.horizon = float(horizon)
        self.alpha = float(alpha)
        self.s = float(s)
        self.lb = lb
        self.ub = ub

        self.rho = 1.0
        self.mu = 1.0
        self.h = 0.01
        self.calpha = 24 * self.mu / (np.pi * self.h * (self.horizon ** 4))

        self.net = PINN_Net(layers, lb, ub).to(device)
        self.integrator = FractionalPDIntegrator_TFAligned(
            delta=self.horizon, calpha=self.calpha, alpha=self.alpha, s=self.s,
            num_frac=num_frac, num_r=num_r, num_theta=num_theta
        ).to(device)

        self.X_u_cpu = torch.tensor(X_u, dtype=torch.float32).pin_memory()
        self.U_u_cpu = torch.tensor(U_u, dtype=torch.float32).pin_memory()
        self.V_u_cpu = torch.tensor(V_u, dtype=torch.float32).pin_memory()
        self.X_f_cpu = torch.tensor(X_f[:, :3], dtype=torch.float32).pin_memory()
        self.F_f_cpu = torch.tensor(X_f[:, 3:5], dtype=torch.float32).pin_memory()
        self.X_test_cpu = torch.tensor(X_test, dtype=torch.float32).pin_memory()

    def _pick_fixed_indices(self, N, k, seed=123):
        rng = np.random.default_rng(seed)
        k = min(int(k), int(N))
        return rng.choice(N, size=k, replace=False).astype(np.int64)

    def make_fixed_sets(self, F_fix=1024):
        self.Xu_fix = self.X_u_cpu.to(device, non_blocking=True)
        self.Uu_fix = self.U_u_cpu.to(device, non_blocking=True)
        self.Vu_fix = self.V_u_cpu.to(device, non_blocking=True)

        ic_mask = (self.X_u_cpu[:, 2] == 0.0)
        if torch.any(ic_mask):
            self.Xic_fix = self.X_u_cpu[ic_mask].to(device, non_blocking=True)
        else:
            self.Xic_fix = None

        Nf = self.X_f_cpu.shape[0]
        idx_f = self._pick_fixed_indices(Nf, F_fix, seed=3)
        self.Xf_fix = self.X_f_cpu[idx_f].to(device, non_blocking=True)
        self.Ff_fix = self.F_f_cpu[idx_f].to(device, non_blocking=True)

    def loss_bc_ic_fixed(self):
        u_pred, v_pred = self.net(self.Xu_fix)
        loss_bc = torch.mean((u_pred - self.Uu_fix) ** 2 + (v_pred - self.Vu_fix) ** 2)

        if self.Xic_fix is None:
            loss_ic = torch.zeros((), device=device)
        else:
            xic = self.Xic_fix[:, 0:1]
            yic = self.Xic_fix[:, 1:2]
            tic = self.Xic_fix[:, 2:3].clone().detach().requires_grad_(True)
            u_ic, v_ic = self.net(torch.cat([xic, yic, tic], dim=1))
            ut_ic = grad(u_ic.sum(), tic, create_graph=True, retain_graph=True)[0]
            vt_ic = grad(v_ic.sum(), tic, create_graph=True, retain_graph=True)[0]
            loss_ic = torch.mean(ut_ic ** 2 + vt_ic ** 2)

        return loss_bc, loss_ic

    def loss_pde_fixed(self):
        xf = self.Xf_fix[:, 0:1]
        yf = self.Xf_fix[:, 1:2]
        tf = self.Xf_fix[:, 2:3].clone().detach().requires_grad_(True)

        u, v = self.net(torch.cat([xf, yf, tf], dim=1))
        ut = grad(u.sum(), tf, create_graph=True, retain_graph=True)[0]
        vt = grad(v.sum(), tf, create_graph=True, retain_graph=True)[0]
        utt = grad(ut.sum(), tf, create_graph=True, retain_graph=True)[0]
        vtt = grad(vt.sum(), tf, create_graph=True, retain_graph=True)[0]

        int_U, int_V = self.integrator.compute_pd_term(xf, yf, tf, self.net)

        bv = self.Ff_fix[:, 0:1]
        bw = self.Ff_fix[:, 1:2]

        res_U = self.rho * utt - int_U - bv
        res_V = self.rho * vtt - int_V - bw
        return torch.mean(res_U ** 2 + res_V ** 2)


if __name__ == "__main__":
    horizon = 0.25
    alpha = 0.6
    s = -0.9
    t_max = 1.0

    layers = [3, 128, 128, 128, 128, 2]
    lb = np.array([-horizon, -horizon, 0.0], dtype=np.float32)
    ub = np.array([1 + horizon, 1 + horizon, 1.0], dtype=np.float32)

    print("正在构建传统隧道前向求解算例...")
    X_u, U_u, V_u, X_f, X_test, Xg, Yg = generate_forward_tunnel_data(
        horizon=horizon, t_max=t_max,
        n_edge=21, n_layer=11, n_time_bc=11, n_init_grid=21,
        N_collocation=1200, alpha=alpha, s=s
    )
    print(f"边界约束点数: {len(X_u)}, 围岩内部配置点数: {len(X_f)}, 待求解测试网格: {len(X_test)}")

    model = FractionalPD_PINN(
        X_u, U_u, V_u, X_f, X_test,
        layers=layers, lb=lb, ub=ub, horizon=horizon, alpha=alpha, s=s
    ).to(device)

    F_LBFGS = 256
    model.make_fixed_sets(F_fix=F_LBFGS)


    loss_bc0, loss_ic0 = model.loss_bc_ic_fixed()
    loss_pde0 = model.loss_pde_fixed()
    eps = 1e-12
    w_bc = 1.0
    w_ic = 1.0
    w_pde = (loss_bc0 + loss_ic0).item() / (loss_pde0.item() + eps)
    w_pde = max(w_pde, 1e-12)


    adam = optim.Adam(model.net.parameters(), lr=1e-3)
    print("=" * 80)
    print(f"开始 Adam 前向进化求解 | 传统隧道工程算例 α={alpha}, s={s}")
    print("=" * 80)

    adam_iters = 12000
    warmup_steps = 2000
    log_interval = 2000
    start_time = time.time()

    for i in range(adam_iters + 1):
        adam.zero_grad(set_to_none=True)
        loss_bc, loss_ic = model.loss_bc_ic_fixed()
        loss_pde = model.loss_pde_fixed()

        if i > 0 and i % 2000 == 0:
            w_pde = (loss_bc.item() + loss_ic.item()) / (loss_pde.item() + eps)
            w_pde = max(min(w_pde, 1e-2), 1e-9)

        if i < warmup_steps:
            alpha_warm = i / warmup_steps
            current_w_pde = alpha_warm * w_pde
        else:
            current_w_pde = w_pde

        loss = w_bc * loss_bc + w_ic * loss_ic + current_w_pde * loss_pde
        loss.backward()
        nn.utils.clip_grad_norm_(model.net.parameters(), 1.0)
        adam.step()

        if i % log_interval == 0:
            elapsed = time.time() - start_time
            print(
                f"Adam {i:5d} | Total Loss: {loss.item():.3e} | BC-Loss: {loss_bc.item():.3e} | PDE-Res: {loss_pde.item():.3e} | Time: {elapsed:.1f}s")

    loss_bc_final, loss_ic_final = model.loss_bc_ic_fixed()
    loss_pde_final = model.loss_pde_fixed()
    w_pde = (loss_bc_final.item() + loss_ic_final.item()) / (loss_pde_final.item() + eps)

    lbfgs = optim.LBFGS(
        model.net.parameters(), lr=1.0, max_iter=600, max_eval=800,
        tolerance_grad=1e-9, tolerance_change=1e-11, history_size=50,
        line_search_fn="strong_wolfe",
    )

    lbfgs_iter = 0


    def closure():
        global lbfgs_iter
        lbfgs.zero_grad(set_to_none=True)
        loss_bc, loss_ic = model.loss_bc_ic_fixed()
        loss_pde = model.loss_pde_fixed()
        loss = w_bc * loss_bc + w_ic * loss_ic + w_pde * loss_pde
        loss.backward()
        lbfgs_iter += 1
        if lbfgs_iter % 50 == 0:
            print(f"LBFGS Iter {lbfgs_iter:4d} | Total Loss: {loss.item():.3e} | PDE-Res: {loss_pde.item():.3e}")
        return loss


    print("\n切换至 LBFGS 进行非 nonlocal 流场平衡精确求解...")
    lbfgs.step(closure)
    print("求解完成！")


    ny, nx = Xg.shape
    X_full_test = np.column_stack([Xg.ravel(), Yg.ravel(), np.ones(Xg.size, np.float32)]).astype(np.float32)

    model.net.eval()
    with torch.no_grad():
        X_full_tensor = torch.tensor(X_full_test, dtype=torch.float32, device=device)
        U_pred, V_pred = model.net(X_full_tensor)
        U_pred_np = U_pred.cpu().numpy().reshape(ny, nx)
        V_pred_np = V_pred.cpu().numpy().reshape(ny, nx)


    hole_mask_grid = ~in_material_mask(Xg, Yg)
    U_pred_np[hole_mask_grid] = np.nan
    V_pred_np[hole_mask_grid] = np.nan


    Disp_magnitude = np.sqrt(U_pred_np ** 2 + V_pred_np ** 2)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))

    im0 = axs[0].contourf(Xg, Yg, U_pred_np, levels=50, cmap="jet")
    fig.colorbar(im0, ax=axs[0])
    axs[0].set_title("Predicted Forward $u$ (Horizontal)")
    axs[0].axis("equal")

    im1 = axs[1].contourf(Xg, Yg, V_pred_np, levels=50, cmap="jet")
    fig.colorbar(im1, ax=axs[1])
    axs[1].set_title("Predicted Forward $v$ (Vertical)")
    axs[1].axis("equal")

    im2 = axs[2].contourf(Xg, Yg, Disp_magnitude, levels=50, cmap="plasma")
    fig.colorbar(im2, ax=axs[2])
    axs[2].set_title("Total Displacement Magnitude $\sqrt{u^2+v^2}$")
    axs[2].axis("equal")

    plt.tight_layout()
    plt.show()