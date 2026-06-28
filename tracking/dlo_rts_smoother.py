"""
dlo_rts_smoother.py — Rauch-Tung-Striebel (RTS) Offline Smoother (T2.8)
=========================================================================
Offline backwards Kalman pass that gives the optimal smoothed estimate
given ALL observations in a recorded sequence.

Use case: post-hoc RMSE evaluation in the thesis — run the demo, record
keypoint estimates, then apply RTS to get the true best achievable RMSE
for comparison.  NOT used in the real-time pipeline.

Reference:
  Rauch, Tung, Striebel (1965). "Maximum likelihood estimates of linear
  dynamic systems." AIAA Journal 3(8).

Theory
------
Forward pass (standard Kalman) produces: x̂_t|t, P_t|t, x̂_t|t-1, P_t|t-1
Backward pass:
  G_t   = P_t|t  F^T  P_{t+1|t}^{-1}          (smoother gain)
  x̂_t|T = x̂_t|t + G_t (x̂_{t+1|T} - x̂_{t+1|t})
  P_t|T  = P_t|t  + G_t (P_{t+1|T} - P_{t+1|t}) G_t^T

CPU-only: NumPy + scipy.linalg.
"""

import numpy as np
from scipy.linalg import solve


def rts_smoother(filtered_means: np.ndarray,
                 filtered_covs:  np.ndarray,
                 predicted_means: np.ndarray,
                 predicted_covs:  np.ndarray,
                 F: np.ndarray) -> tuple:
    """
    Rauch-Tung-Striebel backward smoother.

    Parameters
    ----------
    filtered_means  : (T, n) — x̂_t|t  from forward Kalman pass
    filtered_covs   : (T, n, n) — P_t|t
    predicted_means : (T, n) — x̂_t|t-1  (one-step-ahead predictions)
    predicted_covs  : (T, n, n) — P_t|t-1
    F               : (n, n) — state transition matrix (constant)

    Returns
    -------
    smoothed_means : (T, n)
    smoothed_covs  : (T, n, n)
    """
    T, n = filtered_means.shape
    smoothed_means = filtered_means.copy()
    smoothed_covs  = filtered_covs.copy()

    for t in range(T - 2, -1, -1):
        P_pred = predicted_covs[t + 1]
        try:
            G = filtered_covs[t] @ F.T @ np.linalg.inv(P_pred)
        except np.linalg.LinAlgError:
            G = filtered_covs[t] @ F.T @ np.linalg.pinv(P_pred)

        smoothed_means[t] = (filtered_means[t]
                             + G @ (smoothed_means[t + 1] - predicted_means[t + 1]))
        smoothed_covs[t]  = (filtered_covs[t]
                             + G @ (smoothed_covs[t + 1] - P_pred) @ G.T)

    return smoothed_means, smoothed_covs


class DLORTSSmoother:
    """
    Convenience wrapper: run RTS smoother on a recorded sequence of
    DLO keypoint estimates from the standard constant-velocity Kalman.

    Usage (offline evaluation)
    --------------------------
    smoother = DLORTSSmoother(n_keypoints=20, dt=0.033)
    # Record one sequence:
    for frame in range(T):
        smoother.step(obs_kp[frame], valid[frame])
    smoothed = smoother.smooth()    # (T, N, 3) smoothed positions
    rmse = smoother.rmse(gt_kp)     # scalar RMSE vs ground truth (mm)
    """

    def __init__(self, n_keypoints=20, dt=0.033,
                 sigma_q=0.01, sigma_r=0.005):
        self.n_kp   = n_keypoints
        self.dt     = dt
        self.sq     = sigma_q
        self.sr     = sigma_r

        # Per-keypoint constant-velocity state: [x,y,z,vx,vy,vz]
        n = 6
        self.F = np.eye(n)
        self.F[:3, 3:] = np.eye(3) * dt
        self.Q = np.eye(n) * sigma_q**2
        self.H = np.zeros((3, n)); self.H[:3, :3] = np.eye(3)
        self.R = np.eye(3) * sigma_r**2

        self._reset_buffers()

    def _reset_buffers(self):
        self._filt_means  = [[] for _ in range(self.n_kp)]  # (T,6) per kp
        self._filt_covs   = [[] for _ in range(self.n_kp)]  # (T,6,6)
        self._pred_means  = [[] for _ in range(self.n_kp)]
        self._pred_covs   = [[] for _ in range(self.n_kp)]
        self._x           = [np.zeros(6) for _ in range(self.n_kp)]
        self._P           = [np.eye(6) * 0.1 for _ in range(self.n_kp)]
        self._initialized = [False] * self.n_kp
        self._T           = 0

    def step(self, keypoints_3d: np.ndarray, valid_mask: np.ndarray):
        """Process one frame. keypoints_3d: (N,3), valid_mask: (N,) bool."""
        for i in range(self.n_kp):
            obs_valid = valid_mask[i] and not np.isnan(keypoints_3d[i]).any()

            if not self._initialized[i]:
                if obs_valid:
                    self._x[i][:3] = keypoints_3d[i]
                    self._initialized[i] = True

            # Predict
            x_pred = self.F @ self._x[i]
            P_pred = self.F @ self._P[i] @ self.F.T + self.Q
            self._pred_means[i].append(x_pred.copy())
            self._pred_covs[i].append(P_pred.copy())

            # Update
            if obs_valid:
                y   = keypoints_3d[i] - self.H @ x_pred
                S   = self.H @ P_pred @ self.H.T + self.R
                K   = P_pred @ self.H.T @ np.linalg.inv(S)
                x_u = x_pred + K @ y
                P_u = (np.eye(6) - K @ self.H) @ P_pred
            else:
                x_u, P_u = x_pred, P_pred

            self._x[i] = x_u
            self._P[i] = P_u
            self._filt_means[i].append(x_u.copy())
            self._filt_covs[i].append(P_u.copy())

        self._T += 1

    def smooth(self) -> np.ndarray:
        """
        Run RTS backward pass.
        Returns (T, N, 3) smoothed keypoint positions.
        """
        T = self._T
        out = np.zeros((T, self.n_kp, 3))

        for i in range(self.n_kp):
            if not self._initialized[i]:
                continue
            fm = np.array(self._filt_means[i])   # (T,6)
            fc = np.array(self._filt_covs[i])    # (T,6,6)
            pm = np.array(self._pred_means[i])
            pc = np.array(self._pred_covs[i])

            sm, _ = rts_smoother(fm, fc, pm, pc, self.F)
            out[:, i, :] = sm[:, :3]

        return out

    def rmse(self, gt_kp: np.ndarray, smoothed: np.ndarray = None) -> float:
        """
        RMSE in mm vs ground truth.

        gt_kp    : (T, N, 3) ground truth
        smoothed : (T, N, 3) from smooth() — computed if None
        """
        if smoothed is None:
            smoothed = self.smooth()
        diff = smoothed - gt_kp
        return float(np.sqrt(np.nanmean(diff**2)) * 1000)

    def reset(self):
        self._reset_buffers()
