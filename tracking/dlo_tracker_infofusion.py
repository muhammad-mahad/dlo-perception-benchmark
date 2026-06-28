"""
dlo_tracker_infofusion.py — Multi-Camera Information-Form Kalman Fusion
========================================================================
Information-form (canonical) Kalman filter that fuses observations from
K cameras simultaneously in a single update step.

Per-keypoint state: [x, y, z] (3-dim, position only).
Info-form: Omega = P^{-1}, xi = P^{-1} x.

Predict:
  P_bar = F P F^T + Q   (F=I for constant position)
  Omega_bar = inv(P_bar)
  xi_bar = Omega_bar @ x

Update (multi-camera):
  For each camera k: Omega += H_k^T R_k^{-1} H_k, xi += H_k^T R_k^{-1} z_k
  x = inv(Omega) @ xi

Reference: Maybeck, "Stochastic Models, Estimation, and Control" (1979), Ch.4.
"""

import numpy as np


class InfoFusionTracker:
    """
    Multi-camera Information-form Kalman tracker for DLO keypoints.
    Per-keypoint 3-dim state (position only).
    """

    def __init__(self, n_keypoints: int = 20, dt: float = 0.033,
                 process_sigma: float = 0.01, obs_sigma: float = 0.005):
        self.n_kp = n_keypoints
        self.dt = dt
        self.sigma_process = process_sigma
        self.sigma_obs = obs_sigma

        # Per-keypoint information matrix (3×3) and vector (3,)
        self._Omega = None  # (N, 3, 3)
        self._xi = None     # (N, 3)
        self._initialized = False

    def init(self, keypoints_3d: np.ndarray):
        """Initialize from (N,3) keypoint positions."""
        N = self.n_kp
        P0 = np.eye(3) * 0.1
        Omega0 = np.linalg.inv(P0)
        self._Omega = np.stack([Omega0.copy() for _ in range(N)])
        self._xi = np.zeros((N, 3))
        for i in range(N):
            self._xi[i] = Omega0 @ keypoints_3d[i]
        self._initialized = True

    def predict(self):
        """Info-form prediction: F=I (constant position), Q=diag(sigma_process²)."""
        if not self._initialized:
            return
        N = self.n_kp
        Q = np.eye(3) * self.sigma_process ** 2
        # F = I => P_bar = P + Q
        for i in range(N):
            P_i = np.linalg.inv(self._Omega[i])
            x_i = P_i @ self._xi[i]
            P_bar = P_i + Q
            self._Omega[i] = np.linalg.inv(P_bar)
            self._xi[i] = self._Omega[i] @ x_i

    def update_multicam(self, observations: list) -> np.ndarray:
        """
        Fuse observations from multiple cameras.

        observations : list of (z_k, valid_k)
            z_k     : (N,3) observations from camera k
            valid_k : (N,) bool mask

        Returns (N,3) updated estimates.
        """
        if not self._initialized:
            raise RuntimeError("Call init() first.")

        N = self.n_kp
        R_inv = np.eye(3) / (self.sigma_obs ** 2)  # H=I, R=sigma²I

        for i in range(N):
            for z_k, valid_k in observations:
                if not valid_k[i]:
                    continue
                # H = I3: direct position observation
                self._Omega[i] += R_inv           # H^T R^{-1} H = R^{-1}
                self._xi[i] += R_inv @ z_k[i]    # H^T R^{-1} z

        return self.estimates

    def update(self, keypoints_3d: np.ndarray,
               valid_mask: np.ndarray) -> np.ndarray:
        """Single-camera update (convenience wrapper)."""
        return self.update_multicam([(keypoints_3d, valid_mask)])

    @property
    def estimates(self) -> np.ndarray:
        N = self.n_kp
        out = np.zeros((N, 3))
        for i in range(N):
            out[i] = np.linalg.solve(self._Omega[i], self._xi[i])
        return out

    @property
    def covariance_traces(self) -> np.ndarray:
        N = self.n_kp
        traces = np.zeros(N)
        for i in range(N):
            traces[i] = np.trace(np.linalg.inv(self._Omega[i]))
        return traces

    @property
    def initialized(self) -> bool:
        return self._initialized

    def reset(self):
        self._Omega = None
        self._xi = None
        self._initialized = False
