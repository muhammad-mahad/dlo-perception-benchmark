"""
dlo_tracker_v4.py — Phase 1v4 DLO Tracker
==========================================
Key features:
  T1.4  Hungarian assignment (scipy.optimize.linear_sum_assignment) replaces
        greedy nearest-neighbour — eliminates identity swaps at crossings.
        Cost matrix: 0.7 x Euclidean + 0.3 x geodesic arc-length mismatch.
        Reference: MultiDLO geodesic Hungarian, ICRA-RMDO 2023.
  T1.5  Arc-length occlusion fill (OcclusionHandlerV2) — correct spline
        parameterization along cable arc-length, not keypoint index.

CPU-only: NumPy + scipy
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import CubicSpline, interp1d


# ---------------------------------------------------------------------------
# Kalman filter per keypoint
# ---------------------------------------------------------------------------

class KalmanKeypoint:
    def __init__(self, init_pos, dt=0.033, process_noise=1e-3, obs_noise=5e-3):
        self.dt = dt
        self.F  = np.eye(6); self.F[0,3]=dt; self.F[1,4]=dt; self.F[2,5]=dt
        self.H  = np.zeros((3,6)); self.H[0,0]=self.H[1,1]=self.H[2,2]=1.0
        q = process_noise
        self.Q  = np.diag([q,q,q,q*10,q*10,q*10])
        self.R  = np.eye(3) * obs_noise**2
        self.x  = np.zeros(6); self.x[:3] = init_pos.copy()
        self.P  = np.eye(6) * 0.1

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:3].copy()

    def update(self, obs):
        R_inv = np.linalg.inv(self.R)
        Omega = np.linalg.inv(self.P)
        xi    = Omega @ self.x
        Omega += self.H.T @ R_inv @ self.H
        xi    += self.H.T @ R_inv @ obs
        self.P = np.linalg.inv(Omega)
        self.x = self.P @ xi
        return self.x[:3].copy()

    @property
    def position(self): return self.x[:3].copy()

    @property
    def covariance_trace(self): return float(np.trace(self.P[:3, :3]))


# ---------------------------------------------------------------------------
# T1.4 — DLO tracker with Hungarian assignment
# ---------------------------------------------------------------------------

class DLOTrackerV2:
    """
    Multi-keypoint DLO tracker with Hungarian optimal assignment.

    T1.4: linear_sum_assignment on geodesic-augmented cost matrix.
          Eliminates identity swaps that greedy nearest-neighbour produces
          at cable crossings and occlusion boundaries.
    """

    def __init__(self, n_keypoints=20, dt=0.033,
                 process_noise=1e-3, obs_noise=5e-3, max_assoc_dist=0.15):
        self.n_kp          = n_keypoints
        self.dt            = dt
        self.max_assoc_dist = max_assoc_dist
        self._pn = process_noise
        self._on = obs_noise
        self._filters     = []
        self._initialized = False

    def init(self, kps3d):
        self._filters = [KalmanKeypoint(kps3d[i], self.dt, self._pn, self._on)
                         for i in range(len(kps3d))]
        self._initialized = True

    def predict(self):
        return np.stack([f.predict() for f in self._filters])

    def update(self, detections, valid_mask=None):
        if not self._initialized:
            raise RuntimeError("Call init() before update().")

        if valid_mask is None:
            valid_mask = ~np.isnan(detections).any(axis=1)

        valid_dets    = detections[valid_mask]
        valid_orig_idx = np.where(valid_mask)[0]
        M_dets        = detections.shape[0]
        predicted     = np.stack([f.position for f in self._filters])
        N             = len(self._filters)
        updated       = predicted.copy()

        if len(valid_dets) == 0:
            return updated

        # Geodesic arc-length of predicted track
        segs   = np.linalg.norm(np.diff(predicted, axis=0), axis=1)
        s_track = np.concatenate([[0.], np.cumsum(segs)])
        L       = s_track[-1] if s_track[-1] > 1e-12 else 1.0

        denom = max(M_dets - 1, 1)
        euc   = np.linalg.norm(predicted[:,None,:] - valid_dets[None,:,:], axis=2)
        geo   = np.abs(s_track[:,None]/L - valid_orig_idx[None,:]/denom)
        cost  = 0.7*euc + 0.3*geo                    # (N, K)

        # T1.4 — Hungarian optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost)
        for i, j in zip(row_ind, col_ind):
            if euc[i, j] <= self.max_assoc_dist:
                updated[i] = self._filters[i].update(valid_dets[j])

        return updated

    def step(self, detections, valid_mask=None):
        self.predict()
        return self.update(detections, valid_mask)

    @property
    def estimates(self):
        return np.stack([f.position for f in self._filters])

    @property
    def covariance_traces(self):
        return np.array([f.covariance_trace for f in self._filters])

    @property
    def initialized(self): return self._initialized


# ---------------------------------------------------------------------------
# T1.5 — Arc-length occlusion handler
# ---------------------------------------------------------------------------

class OcclusionHandlerV2:
    """
    T1.5: Cubic spline occlusion fill parameterized by cable arc-length.
    Uses cumulative arc-length of visible keypoints as the spline parameter
    (not keypoint index), which is geometrically correct for curved cables.
    Reference: Caporali 2024 (acknowledged limitation of index-based approach).
    """

    def fill(self, keypoints, valid_mask=None, tracker_prediction=None):
        keypoints  = np.array(keypoints, dtype=float)
        N          = len(keypoints)
        if valid_mask is None:
            valid_mask = ~np.isnan(keypoints).any(axis=1)
        valid_mask = np.asarray(valid_mask, dtype=bool)
        filled     = keypoints.copy()

        n_valid   = valid_mask.sum()
        n_missing = (~valid_mask).sum()

        if n_missing == 0:
            return filled
        if n_valid == 0:
            return tracker_prediction.copy() if tracker_prediction is not None else filled
        if n_valid == 1:
            val = keypoints[valid_mask][0]
            filled[~valid_mask] = val
            return filled
        if n_valid == 2:
            return self._linear_fill(keypoints, valid_mask)

        vis_pts = keypoints[valid_mask]
        segs    = np.linalg.norm(np.diff(vis_pts, axis=0), axis=1)
        s_vis   = np.concatenate([[0.], np.cumsum(segs)])
        total   = s_vis[-1]

        if total < 1e-12:
            return self._linear_fill(keypoints, valid_mask)

        # Normalize visible arc-lengths to [0,1]
        s_hat_vis = s_vis / total                              # (K,)
        # Expected normalized arc-length for every keypoint by index
        s_hat_all = np.arange(N, dtype=float) / max(N-1, 1)  # (N,)

        try:
            cs_x = CubicSpline(s_hat_vis, vis_pts[:,0], extrapolate=True)
            cs_y = CubicSpline(s_hat_vis, vis_pts[:,1], extrapolate=True)
            cs_z = CubicSpline(s_hat_vis, vis_pts[:,2], extrapolate=True)
            miss = ~valid_mask
            s_m  = s_hat_all[miss]
            filled[miss, 0] = cs_x(s_m)
            filled[miss, 1] = cs_y(s_m)
            filled[miss, 2] = cs_z(s_m)
        except Exception:
            filled = self._linear_fill(keypoints, valid_mask)

        if tracker_prediction is not None and n_missing > N // 2:
            alpha = 0.3
            filled[~valid_mask] = ((1-alpha)*filled[~valid_mask]
                                   + alpha*tracker_prediction[~valid_mask])
        return filled

    def _linear_fill(self, keypoints, valid_mask):
        filled  = keypoints.copy()
        N       = len(keypoints)
        vis_idx = np.where(valid_mask)[0].astype(float)
        vis_pts = keypoints[valid_mask]
        all_idx = np.arange(N, dtype=float)
        for c in range(3):
            f = interp1d(vis_idx, vis_pts[:,c],
                         kind='linear', fill_value='extrapolate')
            filled[~valid_mask, c] = f(all_idx[~valid_mask])
        return filled
