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


def exact_solution(x, y, t):
    v = t**2 * x * (1 - x) * y * (1 - y)
    w = t**2 * x * (1 - x) * y * (1 - y)
    return v, w


def compute_body_force(x, y, t, alpha, s, delta, calpha):
    π = np.pi
    term1 = 2.0 * (x - x**2) * (y - y**2)
    coeff = 2 * calpha * torch.pow(t, 2 - alpha) / gamma(3 - alpha)

    delta_3_2s = delta ** (3 - 2 * s)
    delta_5_2s = delta ** (5 - 2 * s)

    part1_v = 3 * π * delta_3_2s * (y - y**2) / (8 - 8 * s)
    part2_v = π * delta_3_2s / (8 - 8 * s) * (3 * x + 2 * y - 4 * x * y - x**2 - 1)
    part3_v = π * delta_5_2s / (32 - 16 * s)
    bv = term1 + coeff * (part1_v + part2_v - part3_v)

    part1_w = 3 * π * delta_3_2s * (x - x**2) / (8 - 8 * s)
    part2_w = π * delta_3_2s / (8 - 8 * s) * (2 * x + 3 * y - 4 * x * y - y**2 - 1)
    part3_w = π * delta_5_2s / (32 - 16 * s)
    bw = term1 + coeff * (part1_w + part2_w - part3_w)

    return bv, bw


def generate_training_data_structured(
    horizon=1/8, t_max=1.0,
    n_edge=21,      
    n_layer=11, 
    n_time_bc=11,    
    n_init_grid=21,
    N_collocation=1200
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

    for tt in t_list:
        XX, YY = np.meshgrid(xL, y01, indexing="xy")
        pts.append(np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, tt, np.float32)], axis=1))

        XX, YY = np.meshgrid(xR, y01, indexing="xy")
        pts.append(np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, tt, np.float32)], axis=1))

        XX, YY = np.meshgrid(x01, yB, indexing="xy")
        pts.append(np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, tt, np.float32)], axis=1))
 
        XX, YY = np.meshgrid(x01, yT, indexing="xy")
        pts.append(np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, tt, np.float32)], axis=1))

    X_bc = np.concatenate(pts, axis=0).astype(np.float32)

    x_all = np.linspace(-δ, 1.0 + δ, n_init_grid, dtype=np.float32)
    y_all = np.linspace(-δ, 1.0 + δ, n_init_grid, dtype=np.float32)
    XX0, YY0 = np.meshgrid(x_all, y_all, indexing="xy")
    X_ic = np.stack([XX0.ravel(), YY0.ravel(), np.zeros(XX0.size, np.float32)], axis=1).astype(np.float32)

    X_u = np.concatenate([X_bc, X_ic], axis=0).astype(np.float32)

    U_u = np.zeros((X_u.shape[0], 1), dtype=np.float32)
    V_u = np.zeros((X_u.shape[0], 1), dtype=np.float32)
    for i in range(X_u.shape[0]):
        u, v = exact_solution(float(X_u[i, 0]), float(X_u[i, 1]), float(X_u[i, 2]))
        U_u[i, 0] = u
        V_u[i, 0] = v

    X_f = np.column_stack([
        np.random.uniform(0.0, 1.0, N_collocation).astype(np.float32),
        np.random.uniform(0.0, 1.0, N_collocation).astype(np.float32),
        np.random.uniform(1e-3, t_max, N_collocation).astype(np.float32),
    ]).astype(np.float32)

    nx, ny = 21, 21
    x = np.linspace(0, 1, nx, dtype=np.float32)
    y = np.linspace(0, 1, ny, dtype=np.float32)
    Xg, Yg = np.meshgrid(x, y, indexing="xy")
    X_test = np.column_stack([Xg.ravel(), Yg.ravel(), np.ones(Xg.size, np.float32)]).astype(np.float32)

    U_test = np.zeros((X_test.shape[0], 1), dtype=np.float32)
    V_test = np.zeros((X_test.shape[0], 1), dtype=np.float32)
    for i in range(X_test.shape[0]):
        u, v = exact_solution(float(X_test[i, 0]), float(X_test[i, 1]), float(X_test[i, 2]))
        U_test[i, 0] = u
        V_test[i, 0] = v

    return X_u, U_u, V_u, X_f, X_test, U_test, V_test


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
    def __init__(self, delta, calpha, alpha, s,
                 num_frac=12, num_r=25, num_theta=20,
                 eps_t=1e-12):
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

        self.r_scale = (self.delta / 2.0) ** (1.0 - (1.0 + 2.0 * self.s))  # (delta/2)^(-2s)

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

        self.register_buffer("coef11", (w_spatial * cosT**2).flatten())
        self.register_buffer("coef12", (w_spatial * sinT * cosT).flatten())
        self.register_buffer("coef22", (w_spatial * sinT**2).flatten())
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
        time_w = torch.pow(torch.clamp(bianhuan, min=self.eps_t), 1.0 - self.alpha) * self.wtau.unsqueeze(0)  # [B,n_tau]

        x_c = x.repeat_interleave(n_tau, dim=0)
        y_c = y.repeat_interleave(n_tau, dim=0)
        t_c = si.reshape(-1, 1)
        Ut_c, Vt_c = self._time_derivative(net, x_c, y_c, t_c)
        Ut_c = Ut_c.view(B, n_tau)
        Vt_c = Vt_c.view(B, n_tau)

        x_nei = x + self.xi_x.unsqueeze(0) 
        y_nei = y + self.xi_y.unsqueeze(0)

        x_n = x_nei.unsqueeze(2).expand(B, n_sp, n_tau).reshape(-1, 1)
        y_n = y_nei.unsqueeze(2).expand(B, n_sp, n_tau).reshape(-1, 1)
        t_n = si.unsqueeze(1).expand(B, n_sp, n_tau).reshape(-1, 1)

        Ut_n, Vt_n = self._time_derivative(net, x_n, y_n, t_n)
        Ut_n = Ut_n.view(B, n_sp, n_tau)
        Vt_n = Vt_n.view(B, n_sp, n_tau)

        dUt = Ut_n - Ut_c.unsqueeze(1)
        dVt = Vt_n - Vt_c.unsqueeze(1)

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
    def __init__(self, X_u, U_u, V_u, X_f, X_test, U_test, V_test,
                 layers, lb, ub, horizon, alpha, s,
                 num_frac=8, num_r=8, num_theta=8): 
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
        self.X_f_cpu = torch.tensor(X_f, dtype=torch.float32).pin_memory()
        self.X_test_cpu = torch.tensor(X_test, dtype=torch.float32).pin_memory()
        self.U_test_cpu = torch.tensor(U_test, dtype=torch.float32).pin_memory()
        self.V_test_cpu = torch.tensor(V_test, dtype=torch.float32).pin_memory()

    def _pick_fixed_indices(self, N, k, seed=123):
        rng = np.random.default_rng(seed)
        k = min(int(k), int(N))
        return rng.choice(N, size=k, replace=False).astype(np.int64)

    def make_fixed_sets(self, F_fix=256):
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
        bv, bw = compute_body_force(xf, yf, tf, self.alpha, self.s, self.horizon, self.calpha)

        res_U = self.rho * utt - int_U - bv
        res_V = self.rho * vtt - int_V - bw
        return torch.mean(res_U ** 2 + res_V ** 2)

    def evaluate(self, chunk=256):
        self.net.eval()
        du_list, dv_list = [], []
        with torch.no_grad():
            N = self.X_test_cpu.shape[0]
            for i in range(0, N, chunk):
                Xb = self.X_test_cpu[i:i + chunk].to(device, non_blocking=True)
                Ub = self.U_test_cpu[i:i + chunk].to(device, non_blocking=True)
                Vb = self.V_test_cpu[i:i + chunk].to(device, non_blocking=True)
                up, vp = self.net(Xb)
                du_list.append(up - Ub)
                dv_list.append(vp - Vb)
        du = torch.cat(du_list, dim=0)
        dv = torch.cat(dv_list, dim=0)
        err_u_Linf = torch.max(torch.abs(du)).item()
        err_v_Linf = torch.max(torch.abs(dv)).item()
        err_u_L2 = torch.sqrt(torch.mean(du ** 2)).item()
        err_v_L2 = torch.sqrt(torch.mean(dv ** 2)).item()
        self.net.train()
        return err_u_Linf, err_v_Linf, err_u_L2, err_v_L2


if __name__ == "__main__":
    horizon = 1 / 8
    alpha = 0.5
    s = -0.9                    
    t_max = 1.0

    layers = [3, 128, 128, 128, 128, 2]
    lb = np.array([-horizon, -horizon, 0.0], dtype=np.float32)
    ub = np.array([1 + horizon, 1 + horizon, 1.0], dtype=np.float32)

    print("生成训练数据...")
    X_u, U_u, V_u, X_f, X_test, U_test, V_test = generate_training_data_structured(
        horizon=horizon, t_max=t_max,
        n_edge=21, n_layer=11, n_time_bc=11,
        n_init_grid=21,
        N_collocation=1200
    )
    print(f"监督点总数: {len(X_u)}, 配置点数: {len(X_f)}, 测试点数: {len(X_test)}")


    model = FractionalPD_PINN(
        X_u, U_u, V_u, X_f, X_test, U_test, V_test,
        layers=layers, lb=lb, ub=ub,
        horizon=horizon, alpha=alpha, s=s,
        num_frac=8, num_r=8, num_theta=8
    ).to(device)

   
    F_LBFGS = 256
    model.make_fixed_sets(F_fix=F_LBFGS)

  
    print("计算初始损失以确定权重...")
    loss_bc0, loss_ic0 = model.loss_bc_ic_fixed()
    loss_pde0 = model.loss_pde_fixed()
    
    eps = 1e-12
    w_bc = 1.0
    w_ic = 1.0
    w_pde = (loss_bc0 + loss_ic0).item() / (loss_pde0.item() + eps)
    print(f"初始损失: BC={loss_bc0.item():.3e}, IC={loss_ic0.item():.3e}, PDE={loss_pde0.item():.3e}")
    print(f"计算得到的权重: w_bc={w_bc}, w_ic={w_ic}, w_pde={w_pde:.3e}")

    
    adam = optim.Adam(model.net.parameters(), lr=2e-4)
    print("=" * 80)
    print(f"开始 Adam 预热 | α={alpha}, s={s}, δ={horizon}")
    print("=" * 80)

    adam_iters = 15000
    warmup_steps = 5000  
    log_interval = 1000
    start_time = time.time()

    for i in range(adam_iters + 1):
        adam.zero_grad(set_to_none=True)
        loss_bc, loss_ic = model.loss_bc_ic_fixed()
        loss_pde = model.loss_pde_fixed()


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
            err_u_Linf, err_v_Linf, err_u_L2, err_v_L2 = model.evaluate()
            elapsed = time.time() - start_time
            print(f"Adam {i:5d} | Total: {loss.item():.3e} | BC: {loss_bc.item():.3e} | PDE: {loss_pde.item():.3e} | IC: {loss_ic.item():.3e}")
            print(f"          | U-L∞: {err_u_Linf:.4e} | U-L2: {err_u_L2:.4e} | Time: {elapsed:.1f}s")
            if device.type == "cuda":
                torch.cuda.empty_cache()

  
    print("=" * 80)
    print("开始 LBFGS 精修（固定训练子集）")
    print("=" * 80)

    lbfgs = optim.LBFGS(
        model.net.parameters(),
        lr=1.0,
        max_iter=2000,
        max_eval=2000,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure():
        lbfgs.zero_grad(set_to_none=True)
        loss_bc, loss_ic = model.loss_bc_ic_fixed()
        loss_pde = model.loss_pde_fixed()
        loss = w_bc * loss_bc + w_ic * loss_ic + w_pde * loss_pde
        loss.backward()
        return loss

    lbfgs.step(closure)

   
    err_u_Linf, err_v_Linf, err_u_L2, err_v_L2 = model.evaluate()
    print("=" * 80)
    print("训练结束，输出请求信息：")
    print("=" * 80)
    print(f"使用的 num_frac = {model.integrator.num_frac}, num_r = {model.integrator.r_nodes.numel()}, "
          f"num_theta = {int(model.integrator.coef11.numel() / model.integrator.r_nodes.numel())}")

    model.net.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        U_pred, V_pred = model.net(X_test_t)
        U_pred_np = U_pred.detach().cpu().numpy()
        V_pred_np = V_pred.detach().cpu().numpy()

    maxU_exact = float(np.max(U_test))
    maxU_pred = float(np.max(U_pred_np))
    print(f"max(U_exact) = {maxU_exact:.6f}, max(U_pred) = {maxU_pred:.6f}, 比值 = {maxU_exact / (maxU_pred + 1e-12):.6f}")
    print(f"稳定后损失（Adam 最后一步）:")
    print(f"    loss_bc = {loss_bc.item():.3e}, loss_ic = {loss_ic.item():.3e}, loss_pde = {loss_pde.item():.3e}")
    print("=" * 80)


    nx, ny = 21, 21
    X_grid = X_test[:, 0].reshape(ny, nx)
    Y_grid = X_test[:, 1].reshape(ny, nx)
    U_exact_grid = U_test.reshape(ny, nx)
    U_pred_grid = U_pred_np.reshape(ny, nx)
    V_exact_grid = V_test.reshape(ny, nx)
    V_pred_grid = V_pred_np.reshape(ny, nx)

    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.contourf(X_grid, Y_grid, U_exact_grid, levels=50, cmap="jet")
    plt.colorbar()
    plt.title("Exact $u$ at $t=1$")
    plt.axis("equal")

    plt.subplot(1, 3, 2)
    plt.contourf(X_grid, Y_grid, U_pred_grid, levels=50, cmap="jet")
    plt.colorbar()
    plt.title("Predicted $u$ at $t=1$")
    plt.axis("equal")

    plt.subplot(1, 3, 3)
    plt.contourf(X_grid, Y_grid, U_exact_grid - U_pred_grid, levels=50, cmap="RdBu")
    plt.colorbar()
    plt.title("Error of $u$ at $t=1$")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.contourf(X_grid, Y_grid, V_exact_grid, levels=50, cmap="jet")
    plt.colorbar()
    plt.title("Exact $v$ at $t=1$")
    plt.axis("equal")

    plt.subplot(1, 3, 2)
    plt.contourf(X_grid, Y_grid, V_pred_grid, levels=50, cmap="jet")
    plt.colorbar()
    plt.title("Predicted $v$ at $t=1$")
    plt.axis("equal")

    plt.subplot(1, 3, 3)
    plt.contourf(X_grid, Y_grid, V_exact_grid - V_pred_grid, levels=50, cmap="RdBu")
    plt.colorbar()
    plt.title("Error of $v$ at $t=1$")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()