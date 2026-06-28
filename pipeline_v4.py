"""
pipeline_v4.py — Phase 1v4 Unified Perception + Tracking Pipeline
==================================================================
Wires together all Task 1 upgrades:
  T1.1  Multi-view DLT triangulation (DLOPerceptionV2.perceive_multiview)
  T1.2  RT-DLO topology graph in extract_skeleton
  T1.4  Hungarian assignment (DLOTrackerV2)
  T1.5  Arc-length occlusion fill (OcclusionHandlerV2)

Public API:
  pipeline.get_dlo_state(rgb, depth)                   -> (N,3) cam-frame
  pipeline.get_dlo_state_multiview(cam_data, cam_params) -> (N,3) world-frame
"""

import sys, os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "perception"))
sys.path.insert(0, os.path.join(_HERE, "tracking"))

from dlo_perception_v4 import DLOPerceptionV2, triangulate_multiview, _build_projection
from dlo_tracker_v4    import DLOTrackerV2, OcclusionHandlerV2

try:
    from scipy.signal import savgol_filter as _savgol
    _HAS_SAVGOL = True
except ImportError:
    _HAS_SAVGOL = False


class PipelineV2:
    def __init__(self, n_keypoints=20, dt=0.033, smooth=False,
                 savgol_window=7, savgol_poly=2, use_epipolar=False):
        self.n_kp        = n_keypoints
        self.dt          = dt
        self._perception = DLOPerceptionV2(n_keypoints=n_keypoints,
                                           use_epipolar=use_epipolar)
        self._tracker    = DLOTrackerV2(n_keypoints=n_keypoints, dt=dt)
        self._occlusion  = OcclusionHandlerV2()
        self._last_good  = None
        self._frame      = 0
        # T2.7 — Savitzky-Golay smoothing
        self._smooth        = smooth and _HAS_SAVGOL
        self._savgol_win    = savgol_window
        self._savgol_poly   = savgol_poly
        self._kp_history    = []   # list of (N,3) arrays for SG window

    def set_intrinsics(self, K):
        self._perception.set_intrinsics(K)

    def _run_tracker(self, raw_kp, valid):
        if not self._tracker.initialized:
            pts = raw_kp if valid.all() else self._occlusion.fill(raw_kp, valid)
            if not np.isnan(pts).any():
                self._tracker.init(pts)
            tracked = pts
        else:
            self._tracker.predict()
            tracked = self._tracker.update(raw_kp, valid)

        pred = self._tracker.estimates if self._tracker.initialized else None
        filled = self._occlusion.fill(tracked, ~np.isnan(tracked).any(axis=1), pred)
        if not np.isnan(filled).any():
            self._last_good = filled.copy()
        self._frame += 1

        # T2.7 — Savitzky-Golay temporal smoothing (optional)
        if self._smooth and _HAS_SAVGOL and not np.isnan(filled).any():
            self._kp_history.append(filled.copy())
            if len(self._kp_history) > self._savgol_win:
                self._kp_history.pop(0)
            if len(self._kp_history) >= self._savgol_win:
                hist = np.stack(self._kp_history, axis=0)  # (W, N, 3)
                for c in range(3):
                    hist[:, :, c] = _savgol(hist[:, :, c], self._savgol_win,
                                            self._savgol_poly, axis=0)
                filled = hist[-1]  # smoothed current frame

        return filled

    def get_dlo_state(self, rgb, depth):
        result = self._perception.perceive(rgb, depth)
        if result['success']:
            raw_kp = result['keypoints_3d']
            valid  = ~np.isnan(raw_kp).any(axis=1)
        else:
            raw_kp = (self._last_good.copy() if self._last_good is not None
                      else np.full((self.n_kp, 3), np.nan))
            valid  = np.zeros(self.n_kp, dtype=bool)
        return self._run_tracker(raw_kp, valid)

    def get_dlo_state_multiview(self, cam_data, cam_params):
        """
        cam_data   : {cam_name: (rgb, depth)}
        cam_params : {cam_name: {'K': dict, 'R': (3,3), 't': (3,)}}
        Returns (N,3) world-frame keypoints.
        """
        names = list(cam_data.keys())
        rgbs  = [cam_data[n][0] for n in names]
        Ks    = [cam_params[n]['K'] for n in names]
        Rs    = [cam_params[n]['R'] for n in names]
        ts    = [cam_params[n]['t'] for n in names]
        depth0 = cam_data[names[0]][1]

        mv = self._perception.perceive_multiview(rgbs, Ks, Rs, ts, depth0)
        kps3d = mv['keypoints_3d']

        if kps3d is None:
            raw_kp = (self._last_good.copy() if self._last_good is not None
                      else np.full((self.n_kp, 3), np.nan))
            valid = np.zeros(self.n_kp, dtype=bool)
        else:
            raw_kp = kps3d
            valid  = ~np.isnan(raw_kp).any(axis=1)

        return self._run_tracker(raw_kp, valid)

    def reset(self):
        self._tracker     = DLOTrackerV2(n_keypoints=self.n_kp, dt=self.dt)
        self._last_good   = None
        self._frame       = 0
        self._kp_history  = []

    @property
    def perception(self): return self._perception

    @property
    def tracker(self): return self._tracker
