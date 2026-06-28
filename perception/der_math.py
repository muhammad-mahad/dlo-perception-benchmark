"""
B8: Bergou 2008 — Discrete Elastic Rods (DER) Math Reference
Paper: Bergou et al., "Discrete Elastic Rods" (SIGGRAPH 2008)

Key equations:
  1. Discrete curvature binormal κb_i [Bergou Eq. 1]
  2. Bishop frame parallel transport [Bergou App. A]
  3. Material curvature (κ1, κ2) [Bergou Eq. 2]
  4. Bending energy E_b [Bergou Eq. 4]
  5. Twist energy E_t [Bergou Eq. 5]
"""

import numpy as np


def discrete_curvature_binormal(pts):
    """[Bergou 2008, Eq. 1]: κb_i = 2(t_{i-1} × t_i)/(1 + t_{i-1}·t_i)"""
    if len(pts) < 3:
        return np.zeros((max(0, len(pts) - 2), 3))
    edges = np.diff(pts, axis=0)
    t = edges / (np.linalg.norm(edges, axis=1, keepdims=True) + 1e-9)
    kb = np.zeros((len(pts) - 2, 3))
    for i in range(len(pts) - 2):
        denom = 1.0 + np.dot(t[i], t[i + 1])
        kb[i] = 2.0 * np.cross(t[i], t[i + 1]) / max(denom, 1e-9)
    return kb


def bishop_frame_propagation(pts, u0=None):
    """Parallel transport Bishop frame along rod [Bergou 2008, App. A]."""
    edges = np.diff(pts, axis=0)
    t = edges / (np.linalg.norm(edges, axis=1, keepdims=True) + 1e-9)
    n_edges = len(t)
    u = np.zeros((n_edges, 3))
    if u0 is None:
        v = np.array([1.0, 0.0, 0.0]) if abs(t[0][0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u0 = v - np.dot(v, t[0]) * t[0]
        u0 /= np.linalg.norm(u0) + 1e-9
    u[0] = u0
    for i in range(1, n_edges):
        b = np.cross(t[i - 1], t[i])
        b_norm = np.linalg.norm(b)
        if b_norm < 1e-9:
            u[i] = u[i - 1]
        else:
            b /= b_norm
            c = np.clip(np.dot(t[i - 1], t[i]), -1, 1)
            s = np.sqrt(max(0, 1 - c**2))
            u[i] = u[i-1]*c + np.cross(b, u[i-1])*s + b*np.dot(b, u[i-1])*(1-c)
            u[i] -= np.dot(u[i], t[i]) * t[i]
            u[i] /= np.linalg.norm(u[i]) + 1e-9
    return u


def bending_energy_der(pts, kappa1_rest=None, kappa2_rest=None, EI=0.01):
    """[Bergou 2008, Eq. 4]: E_b = sum_i (EI/2) ||κ_i - κ̄_i||² / l_i"""
    kb = discrete_curvature_binormal(pts)
    lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    l_mid = (lens[:-1] + lens[1:]) / 2.0 + 1e-9
    kappa_sq = np.sum(kb**2, axis=1)
    if kappa1_rest is not None:
        kappa_rest_sq = np.array(kappa1_rest)**2 + np.array(kappa2_rest)**2
        kappa_sq = (np.sqrt(kappa_sq) - np.sqrt(kappa_rest_sq))**2
    return float(np.sum(EI / 2.0 * kappa_sq / l_mid))


def twist_energy_der(theta, theta_rest, lens, GJ=0.005):
    """[Bergou 2008, Eq. 5]: E_t = sum_i (GJ/2) (θ_i - θ̄_i)² / l_i"""
    return float(np.sum(GJ / 2.0 * (theta - theta_rest)**2 / (lens + 1e-9)))
