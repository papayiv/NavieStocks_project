"""
Три солвера для 2D несжимаемого Навье-Стокса (vorticity-stream).

Методы:
    1. solve_fd        — конечно-разностный (upwind + substeps)
    2. solve_scipy     — спектральный + scipy RK45
    3. solve_torch     — спектральный + torchdiffeq (дифференцируемый через PyTorch)
"""

from __future__ import annotations
from typing import Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp

try:
    import torch
    from torchdiffeq import odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False


# ============================================================
#               ВСПОМОГАТЕЛЬНЫЕ
# ============================================================

def _solve_poisson_fft(rhs: np.ndarray, kx: np.ndarray, ky: np.ndarray):
    """Решает ∇²φ = rhs спектрально."""
    rhs_hat = np.fft.fft2(rhs)
    k2 = kx**2 + ky**2
    k2_safe = k2.copy()
    k2_safe[0, 0] = 1.0
    phi_hat = rhs_hat / k2_safe
    phi_hat[0, 0] = 0.0
    return np.fft.ifft2(phi_hat).real


def _velocity_from_psi(psi, dx, dy):
    """u = ∂ψ/∂y, v = -∂ψ/∂x."""
    u = (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / (2 * dy)
    v = -(np.roll(psi, 1, axis=0) - np.roll(psi, -1, axis=0)) / (2 * dx)
    return u, v


# ============================================================
#              1. FD + UPWIND + SUBSTEPS
# ============================================================

def solve_fd(
    x: np.ndarray, y: np.ndarray, t: np.ndarray,
    omega0: np.ndarray, nu: float,
) -> Dict:
    """
    Конечно-разностный солвер с устойчивой схемой:
      - Upwind для конвекции
      - Автоматические суб-шаги для CFL + диффузии
    """
    nx, ny = len(x), len(y)
    dx, dy = x[1] - x[0], y[1] - y[0]
    dt_out = t[1] - t[0]

    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)[:, None]
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)[None, :]

    omega = np.zeros((len(t), ny, nx), dtype=np.float64)
    omega[0] = omega0.copy()

    for n in range(len(t) - 1):
        w = omega[n].copy()

        # Оценка CFL и диффузии
        psi = _solve_poisson_fft(-w, kx, ky)
        u, v = _velocity_from_psi(psi, dx, dy)
        u_max = np.max(np.abs(u)) + 1e-10
        v_max = np.max(np.abs(v)) + 1e-10

        dt_cfl = 0.4 * min(dx / u_max, dy / v_max)
        dt_visc = 0.2 * min(dx, dy) ** 2 / (nu + 1e-12)
        dt_sub = min(dt_cfl, dt_visc, dt_out)
        n_substeps = max(1, int(np.ceil(dt_out / dt_sub)))
        dt_sub = dt_out / n_substeps

        # Суб-шаги
        for _ in range(n_substeps):
            psi = _solve_poisson_fft(-w, kx, ky)
            u, v = _velocity_from_psi(psi, dx, dy)

            w_xm = np.roll(w, 1, axis=0)
            w_xp = np.roll(w, -1, axis=0)
            w_ym = np.roll(w, 1, axis=1)
            w_yp = np.roll(w, -1, axis=1)

            u_pos = np.maximum(u, 0)
            u_neg = np.minimum(u, 0)
            v_pos = np.maximum(v, 0)
            v_neg = np.minimum(v, 0)

            dw_dx_up = (u_pos * (w - w_xm) + u_neg * (w_xp - w)) / dx
            dw_dy_up = (v_pos * (w - w_ym) + v_neg * (w_yp - w)) / dy

            lap_w = ((w_xp - 2*w + w_xm) / dx**2 +
                     (w_yp - 2*w + w_ym) / dy**2)

            w = w + dt_sub * (-dw_dx_up - dw_dy_up + nu * lap_w)

        omega[n + 1] = w

    # Восстановление (u, v)
    u_all = np.zeros_like(omega)
    v_all = np.zeros_like(omega)
    for n in range(len(t)):
        psi = _solve_poisson_fft(-omega[n], kx, ky)
        u_all[n], v_all[n] = _velocity_from_psi(psi, dx, dy)

    return {"u": u_all, "v": v_all, "omega": omega}


# ============================================================
#              2. SCIPY (МЕТОД ЛИНИЙ + RK45)
# ============================================================

def solve_scipy(
    x: np.ndarray, y: np.ndarray, t: np.ndarray,
    omega0: np.ndarray, nu: float,
) -> Optional[Dict]:
    """Спектральный + scipy RK45."""
    nx, ny = len(x), len(y)
    dx, dy = x[1] - x[0], y[1] - y[0]

    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)[:, None]
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)[None, :]
    k2 = kx**2 + ky**2
    k2_safe = k2.copy()
    k2_safe[0, 0] = 1.0

    def rhs(_t, omega_flat):
        w = omega_flat.reshape(ny, nx)
        w_hat = np.fft.fft2(w)
        psi_hat = -w_hat / k2_safe
        psi_hat[0, 0] = 0.0
        u = np.fft.ifft2(1j * ky * psi_hat).real
        v = np.fft.ifft2(-1j * kx * psi_hat).real
        dw_dx = np.fft.ifft2(1j * kx * w_hat).real
        dw_dy = np.fft.ifft2(1j * ky * w_hat).real
        nl = u * dw_dx + v * dw_dy
        nl_hat = np.fft.fft2(nl)
        dw_dt_hat = -nl_hat - nu * k2 * w_hat
        return np.fft.ifft2(dw_dt_hat).real.ravel()

    sol = solve_ivp(
        rhs, (t[0], t[-1]), omega0.ravel(),
        t_eval=t, method="RK45", rtol=1e-6, atol=1e-8,
    )

    omega = sol.y.T.reshape(len(t), ny, nx)
    u_all = np.zeros_like(omega)
    v_all = np.zeros_like(omega)
    for n in range(len(t)):
        w_hat = np.fft.fft2(omega[n])
        psi_hat = -w_hat / k2_safe
        psi_hat[0, 0] = 0.0
        u_all[n] = np.fft.ifft2(1j * ky * psi_hat).real
        v_all[n] = np.fft.ifft2(-1j * kx * psi_hat).real

    return {"u": u_all, "v": v_all, "omega": omega}


# ============================================================
#              3. TORCHDIFFEQ (PyTorch, дифференцируемый)
# ============================================================

def solve_torch(
    x: np.ndarray, y: np.ndarray, t: np.ndarray,
    omega0: np.ndarray, nu: float,
) -> Optional[Dict]:
    """
    Дифференцируемый солвер на PyTorch + torchdiffeq.
    
    Преимущества:
      - Float64 по умолчанию (нет проблем с переполнением)
      - Dealiasing фильтр (2/3 rule) для устойчивости
      - Адаптивный dopri5 (Runge-Kutta 5-го порядка)
      - Градиенты через PyTorch autograd (для обратных задач)
    """
    if not HAS_TORCHDIFFEQ:
        print("[torchdiffeq] не установлен — пропуск")
        return None

    nx, ny = len(x), len(y)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])

    # Частоты
    kx_np = 2 * np.pi * np.fft.fftfreq(nx, d=dx)[:, None]
    ky_np = 2 * np.pi * np.fft.fftfreq(ny, d=dy)[None, :]
    k2_np = kx_np**2 + ky_np**2
    k2_safe_np = k2_np.copy()
    k2_safe_np[0, 0] = 1.0

    # Dealiasing фильтр (2/3 rule)
    kmax_x = np.max(np.abs(kx_np))
    kmax_y = np.max(np.abs(ky_np))
    filter_mask_np = (np.abs(kx_np) < (2.0 / 3.0) * kmax_x) & \
                     (np.abs(ky_np) < (2.0 / 3.0) * kmax_y)
    filter_mask_np = filter_mask_np.astype(np.float64)

    # Конвертация в PyTorch тензоры (float64)
    device = torch.device("cpu")  # можно переключить на cuda если нужно
    kx = torch.from_numpy(kx_np).to(device, dtype=torch.float64)
    ky = torch.from_numpy(ky_np).to(device, dtype=torch.float64)
    k2 = torch.from_numpy(k2_np).to(device, dtype=torch.float64)
    k2_safe = torch.from_numpy(k2_safe_np).to(device, dtype=torch.float64)
    filter_mask = torch.from_numpy(filter_mask_np).to(device, dtype=torch.float64)
    nu_torch = torch.tensor(nu, device=device, dtype=torch.float64)

    class NavierStokesODE(torch.nn.Module):
        """ODE system для torchdiffeq."""
        
        def forward(self, t, omega_flat):
            w = omega_flat.reshape(ny, nx)
            
            # FFT для спектрального метода
            w_hat = torch.fft.fft2(w)
            
            # Пуассон: ∇²ψ = -ω
            psi_hat = -w_hat / k2_safe
            psi_hat[0, 0] = 0.0
            
            # Скорости из ψ
            u = torch.fft.ifft2(1j * ky * psi_hat).real
            v = torch.fft.ifft2(-1j * kx * psi_hat).real
            
            # Производные ω
            dw_dx = torch.fft.ifft2(1j * kx * w_hat).real
            dw_dy = torch.fft.ifft2(1j * ky * w_hat).real
            
            # Нелинейный член
            nl = u * dw_dx + v * dw_dy
            nl_hat = torch.fft.fft2(nl)
            
            # Dealiasing
            nl_hat = nl_hat * filter_mask
            
            # Полная RHS
            dw_dt_hat = -nl_hat - nu_torch * k2 * w_hat
            
            return torch.fft.ifft2(dw_dt_hat).real.ravel()

    ode_func = NavierStokesODE()
    
    # Начальное условие
    omega0_torch = torch.from_numpy(omega0.ravel()).to(device, dtype=torch.float64)
    t_torch = torch.from_numpy(t).to(device, dtype=torch.float64)
    
    # Интегрирование через torchdiffeq
    # dopri5 = Dormand-Prince 5(4) — адаптивный RK
    sol = odeint(
        ode_func,
        omega0_torch,
        t_torch,
        method='dopri5',
        rtol=1e-6,
        atol=1e-8,
        options={'max_num_steps': 1_000_000},
    )
    
    # Конвертация обратно в numpy
    omega = sol.cpu().numpy().reshape(len(t), ny, nx)
    
    # Проверка на NaN
    if np.any(np.isnan(omega)):
        first_nan = np.where(np.isnan(omega).any(axis=(1, 2)))[0]
        if len(first_nan) > 0:
            print(f"[torchdiffeq] warning: NaN появился на шаге t[{first_nan[0]}] = {t[first_nan[0]]:.3f}")

    # Восстановление скоростей
    u_all = np.zeros_like(omega)
    v_all = np.zeros_like(omega)
    for n in range(len(t)):
        w_hat = np.fft.fft2(omega[n])
        psi_hat = -w_hat / k2_safe_np
        psi_hat[0, 0] = 0.0
        u_all[n] = np.fft.ifft2(1j * ky_np * psi_hat).real
        v_all[n] = np.fft.ifft2(-1j * kx_np * psi_hat).real

    return {"u": u_all, "v": v_all, "omega": omega}


# ============================================================
#                     ВИЗУАЛИЗАЦИЯ
# ============================================================

def compare_solutions(
    x, y, t, observations, truth, sol_fd, sol_sp, sol_torch=None,
    output_path=None,
):
    """Сравнение трёх методов."""
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1.0, 1.0, 0.7])

    def _speed(sol):
        return np.sqrt(sol["u"]**2 + sol["v"]**2)

    extent = [x[0], x[-1], y[0], y[-1]]

    t_mid = len(t) // 2
    panels_u = [
        ("Ground truth",      _speed(truth)[t_mid]),
        ("Finite differences", _speed(sol_fd)[t_mid]),
        ("SciPy (RK45)",       _speed(sol_sp)[t_mid]),
        ("torchdiffeq (dopri5)" if sol_torch else "torchdiffeq — N/A",
         _speed(sol_torch)[t_mid] if sol_torch else np.full_like(truth["u"][0], np.nan)),
    ]

    vmax_u = np.nanmax(_speed(truth)[t_mid])
    for i, (title, field) in enumerate(panels_u):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(field, origin="lower", extent=extent,
                       cmap="viridis", vmin=0, vmax=vmax_u)
        ax.set_title(f"|u| — {title}", fontsize=10)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    panels_w = [
        ("Ground truth",      truth["omega"][t_mid]),
        ("Finite differences", sol_fd["omega"][t_mid]),
        ("SciPy (RK45)",       sol_sp["omega"][t_mid]),
        ("torchdiffeq" if sol_torch else "N/A",
         sol_torch["omega"][t_mid] if sol_torch else np.full_like(truth["omega"][0], np.nan)),
    ]
    wmax = np.abs(truth["omega"][t_mid]).max()
    for i, (title, field) in enumerate(panels_w):
        ax = fig.add_subplot(gs[1, i])
        im = ax.imshow(field, origin="lower", extent=extent,
                       cmap="RdBu_r", vmin=-wmax, vmax=wmax)
        ax.set_title(f"ω — {title}", fontsize=10)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    ax_err = fig.add_subplot(gs[2, :2])
    for name, sol, c in [("FD", sol_fd, "#4C9AFF"),
                         ("SciPy", sol_sp, "#FFAB4C"),
                         ("torchdiffeq", sol_torch, "#8ED18E")]:
        if sol is None:
            continue
        err = np.array([np.linalg.norm(sol["u"][n] - truth["u"][n]) /
                        (np.linalg.norm(truth["u"][n]) + 1e-12)
                        for n in range(len(t))])
        ax_err.semilogy(t, err, label=name, lw=2, color=c)

    ax_err.set_xlabel("t"); ax_err.set_ylabel("relative L2 error (|u|)")
    ax_err.set_title("Accuracy vs time"); ax_err.grid(True, alpha=0.3)
    ax_err.legend(loc="best")

    ax_obs = fig.add_subplot(gs[2, 2:])
    im = ax_obs.imshow(_speed(truth)[0], origin="lower", extent=extent,
                       cmap="viridis")
    obs = observations
    dt = t[1] - t[0]
    mask = obs["t"] < t[0] + 0.5 * dt
    if mask.any():
        ax_obs.scatter(obs["x"][mask], obs["y"][mask],
                       color="#D6336C", s=30, zorder=5,
                       edgecolor="black", linewidth=0.5,
                       label=f"observations (t≈0, n={mask.sum()})")
    ax_obs.scatter(obs["x"], obs["y"], color="gray", s=5, alpha=0.4,
                   label=f"all observations (n={len(obs['x'])})")
    ax_obs.set_xlabel("x"); ax_obs.set_ylabel("y")
    ax_obs.set_title("Observation locations")
    ax_obs.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Сохранено: {output_path}")
    plt.show()