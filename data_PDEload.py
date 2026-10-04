"""
Загрузчик данных для 2D несжимаемого Навье-Стокса.
PDEBench: (4, 1000, 512, 512, 2) → downsample до разумных размеров.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def _download_file(url: str, cache_dir: Optional[str] = None,
                   timeout: int = 600) -> Path:
    if not HAS_REQUESTS:
        raise ImportError("pip install requests")

    cache = Path(cache_dir or (Path.home() / ".cache" / "navier_stokes_data"))
    cache.mkdir(parents=True, exist_ok=True)

    filename = url.split("/")[-1].split("?")[0] or "dataset.h5"
    path = cache / filename
    if path.exists():
        print(f"[cache hit] {path} ({path.stat().st_size / 1e6:.1f} MB)")
        return path

    print(f"Скачивание ({url}) ...")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(path, "wb") as f:
            downloaded = 0
            for chunk in r.iter_content(chunk_size=2 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = 100 * downloaded / total
                        print(f"\r  {downloaded/1e6:.1f}/{total/1e6:.1f} MB ({pct:5.1f}%)",
                              end="", flush=True)
    print(f"\nСохранено: {path}")
    return path


def _add_observations(u, v, x, y, t, n_observations, noise_level, seed=42):
    rng = np.random.default_rng(seed)
    nt, ny, nx = u.shape
    obs_t = rng.integers(0, nt, size=n_observations)
    obs_y = rng.integers(0, ny, size=n_observations)
    obs_x = rng.integers(0, nx, size=n_observations)
    return {
        "x": x[obs_x], "y": y[obs_y], "t": t[obs_t],
        "u": u[obs_t, obs_y, obs_x] + rng.normal(0, noise_level, n_observations),
        "v": v[obs_t, obs_y, obs_x] + rng.normal(0, noise_level, n_observations),
        "noise_level": noise_level,
    }


PDEBENCH_NS_URLS = [
    "https://darus.uni-stuttgart.de/api/access/datafile/133280",
    "https://darus.uni-stuttgart.de/api/access/datafile/136439",
    "https://darus.uni-stuttgart.de/api/access/datafile/133309",
]


def load_pdebench_2d_ns(
    file_index: int = 0,
    trajectory_index: int = 0,
    downsample_space: int = 4,
    downsample_time: int = 10,
    n_observations: int = 200,
    noise_level: float = 0.02,
    seed: int = 42,
) -> Dict:
    """
    Загружает 2D Incompressible NS из PDEBench с downsample.
    Оригинальная структура: (batch=4, time=1000, y=512, x=512, dim=2)
    """
    if not HAS_H5PY:
        raise ImportError("pip install h5py")

    url = PDEBENCH_NS_URLS[file_index % len(PDEBENCH_NS_URLS)]
    
    try:
        path = _download_file(url)
    except Exception as e:
        print(f"[warning] PDEBench загрузка не удалась: {e}")
        print("Переходим на локальный Taylor-Green vortex")
        return load_taylor_green(n_observations=n_observations,
                                 noise_level=noise_level, seed=seed)

    print(f"Чтение {path} (lazy loading) ...")

    with h5py.File(path, "r") as f:
        def _find_field(candidates):
            for c in candidates:
                if c in f:
                    return f[c]
                for key in f.keys():
                    if isinstance(f[key], h5py.Group) and c in f[key]:
                        return f[key][c]
            return None

        velocity_ds = _find_field(["velocity", "vel", "u_v"])
        
        if velocity_ds is None:
            u_ds = _find_field(["u", "vx", "velocity_x"])
            v_ds = _find_field(["v", "vy", "velocity_y"])
            if u_ds is None or v_ds is None:
                raise ValueError(f"Не найдены velocity/u/v. Доступно: {list(f.keys())}")
            
            u_full = u_ds[trajectory_index, ::downsample_time, 
                          ::downsample_space, ::downsample_space]
            v_full = v_ds[trajectory_index, ::downsample_time, 
                          ::downsample_space, ::downsample_space]
        else:
            print(f"Найден velocity: shape={velocity_ds.shape}")
            u_full = velocity_ds[trajectory_index, ::downsample_time, 
                                  ::downsample_space, ::downsample_space, 0]
            v_full = velocity_ds[trajectory_index, ::downsample_time, 
                                  ::downsample_space, ::downsample_space, 1]

        u = np.asarray(u_full, dtype=np.float32)
        v = np.asarray(v_full, dtype=np.float32)
        print(f"Загружено: u.shape = {u.shape}, dtype={u.dtype}")

        # Координатные сетки
        x_ds = _find_field(["x", "xgrid", "X"])
        y_ds = _find_field(["y", "ygrid", "Y"])
        t_ds = _find_field(["t", "time", "tgrid"])

        nt, ny, nx = u.shape
        
        # --- X сетка ---
        try:
            if x_ds is not None:
                x_raw = np.asarray(x_ds[::downsample_space]).ravel()
                if len(x_raw) == nx:
                    x_arr = x_raw.astype(np.float32)
                else:
                    x_arr = np.linspace(0, 1, nx, endpoint=False, dtype=np.float32)
            else:
                x_arr = np.linspace(0, 1, nx, endpoint=False, dtype=np.float32)
        except Exception:
            x_arr = np.linspace(0, 1, nx, endpoint=False, dtype=np.float32)
        
        # --- Y сетка ---
        try:
            if y_ds is not None:
                y_raw = np.asarray(y_ds[::downsample_space]).ravel()
                if len(y_raw) == ny:
                    y_arr = y_raw.astype(np.float32)
                else:
                    y_arr = np.linspace(0, 1, ny, endpoint=False, dtype=np.float32)
            else:
                y_arr = np.linspace(0, 1, ny, endpoint=False, dtype=np.float32)
        except Exception:
            y_arr = np.linspace(0, 1, ny, endpoint=False, dtype=np.float32)
        
        # --- T сетка (исправленный блок) ---
        try:
            if t_ds is not None:
                t_raw = np.asarray(t_ds[::downsample_time]).ravel()
                print(f"  t_raw: shape={t_raw.shape}, dtype={t_raw.dtype}, "
                      f"first={t_raw[0] if len(t_raw)>0 else 'N/A'}, "
                      f"last={t_raw[-1] if len(t_raw)>0 else 'N/A'}")
                
                if len(t_raw) == nt:
                    t_arr = t_raw.astype(np.float32)
                elif len(t_raw) == 1:
                    t_arr = np.linspace(0, float(t_raw[0]), nt, dtype=np.float32)
                elif len(t_raw) > nt:
                    t_arr = t_raw[:nt].astype(np.float32)
                elif len(t_raw) > 0:
                    t_arr = np.linspace(float(t_raw[0]), float(t_raw[-1]), 
                                         nt, dtype=np.float32)
                else:
                    t_arr = np.linspace(0, 1, nt, dtype=np.float32)
            else:
                t_arr = np.linspace(0, 1, nt, dtype=np.float32)
        except Exception as e:
            print(f"  [warning] не удалось распарсить t: {e}")
            t_arr = np.linspace(0, 1, nt, dtype=np.float32)

        # Завихренность из скоростей
        dx = float(x_arr[1] - x_arr[0]) if len(x_arr) > 1 else 1.0
        dy = float(y_arr[1] - y_arr[0]) if len(y_arr) > 1 else 1.0
        
        omega = np.zeros_like(u)
        for n in range(nt):
            dvdx = (np.roll(v[n], -1, axis=1) - np.roll(v[n], 1, axis=1)) / (2 * dx)
            dudy = (np.roll(u[n], -1, axis=0) - np.roll(u[n], 1, axis=0)) / (2 * dy)
            omega[n] = dvdx - dudy

        # Вязкость
        nu = 1e-3
        for key in f.keys():
            if isinstance(f[key], h5py.Group) and "nu" in f[key].attrs:
                nu = float(f[key].attrs["nu"])
                break
        if "nu" in f.attrs:
            nu = float(f.attrs["nu"])

    T = float(t_arr[-1]) if len(t_arr) > 0 else 1.0
    
    obs = _add_observations(u, v, x_arr, y_arr, t_arr,
                            n_observations, noise_level, seed)

    return {
        "x": x_arr, "y": y_arr, "t": t_arr,
        "u": u, "v": v, "omega": omega,
        "u0": u[0], "v0": v[0], "omega0": omega[0],
        "observations": obs,
        "params": {
            "nu": nu, "T": T, "L": float(x_arr[-1] - x_arr[0]) + dx,
            "source": f"PDEBench (file {file_index}, traj {trajectory_index})",
            "downsample": {"space": downsample_space, "time": downsample_time},
            "original_shape": "(4, 1000, 512, 512, 2)",
        },
        "grid": {
            "dx": dx, "dy": dy,
            "dt": float(t_arr[1] - t_arr[0]) if len(t_arr) > 1 else 1.0,
            "n_points": nx, "n_time": nt,
        },
    }


def load_taylor_green(
    n_points: int = 64, n_time: int = 40, T: float = 2.0,
    nu: float = 0.01, n_observations: int = 150,
    noise_level: float = 0.02, seed: int = 42,
) -> Dict:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    y = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    t = np.linspace(0, T, n_time)
    X, Y = np.meshgrid(x, y, indexing="xy")

    u = np.zeros((n_time, n_points, n_points))
    v = np.zeros_like(u)
    omega = np.zeros_like(u)
    for n, tt in enumerate(t):
        decay = np.exp(-2 * nu * tt)
        u[n] = np.sin(X) * np.cos(Y) * decay
        v[n] = -np.cos(X) * np.sin(Y) * decay
        omega[n] = -2 * np.sin(X) * np.sin(Y) * decay

    data = {
        "x": x, "y": y, "t": t,
        "u": u, "v": v, "omega": omega,
        "u0": u[0], "v0": v[0], "omega0": omega[0],
        "grid": {"dx": x[1] - x[0], "dy": y[1] - y[0],
                 "dt": t[1] - t[0], "n_points": n_points, "n_time": n_time},
        "params": {"nu": nu, "T": T, "L": 2 * np.pi,
                   "source": "Taylor-Green (analytic)"},
    }
    obs = _add_observations(u, v, x, y, t, n_observations, noise_level, seed)
    data["observations"] = obs
    return data


def _spectral_rk4_step(omega_hat, kx, ky, k2, nu, dt, forcing_hat=None):
    def rhs(wh):
        w = np.fft.ifft2(wh).real
        psi_hat = -wh / k2
        psi_hat[0, 0] = 0.0
        u = np.fft.ifft2(1j * ky * psi_hat).real
        v = np.fft.ifft2(-1j * kx * psi_hat).real
        dw_dx = np.fft.ifft2(1j * kx * wh).real
        dw_dy = np.fft.ifft2(1j * ky * wh).real
        nl_hat = np.fft.fft2(u * dw_dx + v * dw_dy)
        out = -nl_hat - nu * k2 * wh
        if forcing_hat is not None:
            out = out + forcing_hat
        return out
    k1 = rhs(omega_hat)
    k2_ = rhs(omega_hat + 0.5 * dt * k1)
    k3 = rhs(omega_hat + 0.5 * dt * k2_)
    k4 = rhs(omega_hat + dt * k3)
    return omega_hat + (dt / 6.0) * (k1 + 2 * k2_ + 2 * k3 + k4)


def load_kolmogorov(
    n_points: int = 64, n_time: int = 60, T: float = 5.0,
    nu: float = 1e-3, n_observations: int = 300,
    noise_level: float = 0.02, seed: int = 42,
) -> Dict:
    rng = np.random.default_rng(seed)
    L = 2 * np.pi
    x = np.linspace(0, L, n_points, endpoint=False)
    y = np.linspace(0, L, n_points, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="xy")

    omega0 = 0.1 * rng.standard_normal((n_points, n_points))
    omega0 -= omega0.mean()
    forcing = np.sin(4 * Y)

    dx, dy = x[1] - x[0], y[1] - y[0]
    t = np.linspace(0, T, n_time)

    kx = 2 * np.pi * np.fft.fftfreq(n_points, d=dx)[:, None]
    ky = 2 * np.pi * np.fft.fftfreq(n_points, d=dy)[None, :]
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0

    forcing_hat = np.fft.fft2(forcing)
    omega = np.zeros((n_time, n_points, n_points))
    omega[0] = omega0.copy()

    wh = np.fft.fft2(omega0)
    dt = T / n_time
    dt_sub = dt / 8

    for n in range(1, n_time):
        for _ in range(8):
            wh = _spectral_rk4_step(wh, kx, ky, k2, nu, dt_sub, forcing_hat)
        omega[n] = np.fft.ifft2(wh).real

    u = np.zeros_like(omega)
    v = np.zeros_like(omega)
    for n in range(n_time):
        wh_n = np.fft.fft2(omega[n])
        psi_hat = -wh_n / k2
        psi_hat[0, 0] = 0.0
        u[n] = np.fft.ifft2(1j * ky * psi_hat).real
        v[n] = np.fft.ifft2(-1j * kx * psi_hat).real

    data = {
        "x": x, "y": y, "t": t,
        "u": u, "v": v, "omega": omega,
        "u0": u[0], "v0": v[0], "omega0": omega[0],
        "grid": {"dx": dx, "dy": dy, "dt": t[1] - t[0],
                 "n_points": n_points, "n_time": n_time},
        "params": {"nu": nu, "T": T, "L": L,
                   "source": "Kolmogorov flow (forced)"},
    }
    obs = _add_observations(u, v, x, y, t, n_observations, noise_level, seed)
    data["observations"] = obs
    return data


def load_2d_ns_auto(
    mode: Literal["pdebench", "taylor_green", "kolmogorov"] = "pdebench",
    **kwargs,
) -> Dict:
    loaders = {
        "pdebench": load_pdebench_2d_ns,
        "taylor_green": load_taylor_green,
        "kolmogorov": load_kolmogorov,
    }
    return loaders[mode](**kwargs)