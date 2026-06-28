"""
dlo_tracker_inextensible.py — Inextensibility-Constrained Kalman Wrapper
=========================================================================
Wraps any base tracker and enforces segment-length constraints as
pseudo-observations after the base tracker's update step.

For each segment i: pseudo-observation ||p_{i+1} - p_i|| = L0.
Linearized as: h_i ≈ L_cur + J_i @ (dx), where J_i is the Jacobian.

R_constraint ≈ sigma_constraint² * I (near-zero for hard constraint).

Usage:
    base = DLOTrackerV2(n_keypoints=20)
    tracker = InextensibleConstraintKalman(base, n_keypoints=20)
    tracker.init(keypoints_3d)
    for frame:
        result = tracker.update(obs, valid_mask)
"""

import numpy as np


class InextensibleConstraintKalman:
    """
    Inextensibility constraint wrapper for any DLO tracker.

    After the base tracker's update, applies N-1 segment-length
    pseudo-observations to enforce ||p_{i+1} - p_i|| ≈ L0.
    """

    def __init__(self, base_tracker, n_keypoints: int = 20,
                 L0: float = None, constraint_sigma: float = 0.005):
        """
        base_tracker      : any tracker with init/predict/update/estimates API
        n_keypoints       : number of keypoints
        L0                : rest segment length (default: 1.0/(N-1))
        constraint_sigma  : std dev for constraint pseudo-obs (smaller = harder)
        """
        self.base = base_tracker
        self.n_kp = n_keypoints
        self.L0 = L0 if L0 is not None else 1.0 / (n_keypoints - 1)
        self.sigma_c = constraint_sigma
        self.R_c = constraint_sigma ** 2  # scalar variance per constraint

        # Per-keypoint covariance (3×3) maintained by the wrapper
        self._P = None  # (N, 3, 3)
        self._initialized = False

    def init(self, keypoints_3d: np.ndarray):
        self.base.init(keypoints_3d)
        N = self.n_kp
        self._P = np.stack([np.eye(3) * 1.0 for _ in range(N)])
        self._initialized = True

    def predict(self):
        self.base.predict()
        # Process noise inflate P
        if self._P is not None:
            q = 1e-4
            self._P = self._P + np.eye(3) * q

    def update(self, keypoints_3d: np.ndarray,
               valid_mask: np.ndarray) -> np.ndarray:
        """
        1. Run base tracker update.
        2. Apply segment-length pseudo-observations.
        """
        if not self._initialized:
            raise RuntimeError("Call init() first.")

        # Step 1: base tracker update
        pts = self.base.update(keypoints_3d, valid_mask)

        # Step 2: enforce inextensibility per segment
        pts = self._apply_constraints(pts)
        return pts

    def _apply_constraints(self, pts: np.ndarray) -> np.ndarray:
        """
        Inextensibility constraint as soft geometric projection.
        For each segment (i, i+1): gently nudge endpoints toward rest length L0.
        Uses a soft blending (alpha) rather than Kalman gain to avoid instability.
        """
        N = self.n_kp
        corrected = pts.copy()

        # alpha controls constraint strength: 0=no correction, 1=full projection
        alpha = min(1.0, self.R_c / (self.R_c + 1e-4))
        alpha = 0.3  # gentle soft constraint

        for _ in range(2):  # 2 relaxation passes
            for i in range(N - 1):
                pi = corrected[i]
                pj = corrected[i + 1]
                diff = pj - pi
                L_cur = np.linalg.norm(diff) + 1e-9
                error = L_cur - self.L0

                if abs(error) < 1e-7:
                    continue

                # Direction
                d = diff / L_cur
                # Correction: split evenly between the two endpoints
                correction = alpha * error * d * 0.5
                corrected[i] = pi + correction
                corrected[i + 1] = pj - correction

        return corrected

    @property
    def estimates(self) -> np.ndarray:
        return self.base.estimates

    @property
    def covariance_traces(self) -> np.ndarray:
        if self._P is None:
            return np.zeros(self.n_kp)
        return np.array([np.trace(p) for p in self._P])

    @property
    def initialized(self) -> bool:
        return self._initialized

    def reset(self):
        self._P = None
        self._initialized = False
        if hasattr(self.base, 'reset'):
            self.base.reset()
