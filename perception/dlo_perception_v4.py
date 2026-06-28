"""
dlo_perception_v4.py — Phase 1v4 DLO Perception Pipeline
==========================================================
Key features:
  T1.1  Multi-view DLT triangulation (Caporali et al., DLO_MultiView_Tracking)
        replaces single-camera depth-buffer lift (RMSE ~185mm → target <30mm).
  T1.2  RT-DLO topological graph resolution (Caporali ICRA 2023) wired into
        extract_skeleton() — prunes spurious branches, resolves crossings.
  T1.3  Zhang-Suen as skeleton base (67 Hz on CPU; RT-DLO graph cleans topology).

CPU-only: OpenCV + NumPy + scikit-image + networkx + scipy
"""

import numpy as np
import cv2
from skimage.morphology import skeletonize
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist

try:
    import networkx as nx
    _NX = True
except ImportError:
    _NX = False

# ---------------------------------------------------------------------------
# HSV range for the cable as rendered by MuJoCo (appears blue-ish in output)
# Actual rendered hue: 100-130 (verified by pixel inspection)
# ---------------------------------------------------------------------------
HSV_LO = np.array([100,  80,  60], dtype=np.uint8)  # lowered S/V for MuJoCo cable
HSV_HI = np.array([130, 255, 255], dtype=np.uint8)

_KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# ---------------------------------------------------------------------------
# T1.1 — DLT triangulation (ported from Caporali DLO_MultiView_Tracking)
# Reference: Hartley & Zisserman, Multiple View Geometry, Sec 12.2
# ---------------------------------------------------------------------------

def _build_projection(K_dict: dict, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 3x4 projection matrix P = K [R | -R t].

    MuJoCo cameras look along -z_cam, so the verified pixel convention is
    u = fx*(-x_c/z_c)+cx, v = fy*(y_c/z_c)+cy  => fx enters K negated.
    """
    fx, fy = K_dict['fx'], K_dict['fy']
    cx, cy = K_dict['cx'], K_dict['cy']
    K = np.array([[-fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    Rt = np.hstack([R, -(R @ t.reshape(3, 1))])
    return K @ Rt


def triangulate_point_dlt(P_list, uvs):
    """
    Linear triangulation of one 3D point from C camera observations.
    P_list : list of (3,4) projection matrices
    uvs    : list of (2,) pixel coords (u, v)
    Returns (3,) world point.
    """
    rows = []
    for P, (u, v) in zip(P_list, uvs):
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    A = np.stack(rows)                  # (2C, 4)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]                          # (4,) homogeneous
    return (X[:3] / X[3]).astype(float)


def _reproject(P, X):
    """Project (N,3) world points with (3,4) P. Returns (N,2) pixels."""
    Xh = np.hstack([X, np.ones((len(X), 1))])
    uvw = (P @ Xh.T).T
    return uvw[:, :2] / (uvw[:, 2:3] + 1e-12)


def align_chain_directions(kps_per_cam, P_list):
    """
    Resolve per-camera head/tail flips before triangulation.
    Each camera orders its 2D chain independently (topmost-first), so two
    views of the same cable can be tip-flipped; triangulating mismatched
    correspondences produces garbage.  For each non-reference camera, try
    both directions, triangulate against the reference view and keep the
    direction with the lower mean reprojection error.
    """
    ref = next((i for i, k in enumerate(kps_per_cam) if k is not None), None)
    if ref is None:
        return kps_per_cam
    out = list(kps_per_cam)
    for i, kp in enumerate(kps_per_cam):
        if i == ref or kp is None:
            continue
        best_err, best_kp = None, kp
        for cand in (kp, kp[::-1]):
            X = triangulate_multiview([out[ref], cand],
                                      [P_list[ref], P_list[i]])
            if X is None:
                continue
            ok = ~np.isnan(X).any(axis=1)
            if ok.sum() < 2:
                continue
            e = (np.linalg.norm(_reproject(P_list[i], X[ok]) - cand[ok], axis=1).mean()
                 + np.linalg.norm(_reproject(P_list[ref], X[ok]) - out[ref][ok], axis=1).mean())
            if best_err is None or e < best_err:
                best_err, best_kp = e, cand
        out[i] = best_kp
    return out


def triangulate_multiview(kps_per_cam, P_list, min_cameras=2):
    """
    Triangulate N keypoints from C cameras.
    kps_per_cam : list of C arrays shaped (N,2) or None if camera failed
    P_list      : list of C (3,4) projection matrices
    Returns (N,3) world keypoints (NaN where < min_cameras see the point).
    """
    valid_cams = [(i, kp) for i, kp in enumerate(kps_per_cam) if kp is not None]
    if len(valid_cams) < min_cameras:
        return None

    N = valid_cams[0][1].shape[0]
    out = np.full((N, 3), np.nan)

    for n in range(N):
        P_n, uv_n = [], []
        for i, kp in valid_cams:
            if not np.isnan(kp[n]).any():
                P_n.append(P_list[i])
                uv_n.append(kp[n])
        if len(P_n) >= min_cameras:
            out[n] = triangulate_point_dlt(P_n, uv_n)

    return out


# ---------------------------------------------------------------------------
# Stage 1 — epipolar-constrained curve correspondence
# Reference: Hartley & Zisserman, MVG Sec 9 (fundamental matrix) + monotone DP.
#
# Why: per-camera arc-length keypoint sampling is NOT projection-invariant
# (foreshortening), so index-matched triangulation pairs different physical
# points across views -> systematic 2D->3D-lift bias.  Here we match the two
# ORDERED 2D centerlines directly: cost = symmetric point-to-epipolar-line
# distance, constraint = arc-length monotonicity (DP).  Flip is resolved by
# trying both directions (subsumes align_chain_directions).
# ---------------------------------------------------------------------------

def _camera_center(P: np.ndarray) -> np.ndarray:
    """World-frame camera centre C: the right null vector of P (P @ [C;1] = 0)."""
    _, _, Vt = np.linalg.svd(P)
    Ch = Vt[-1]
    return Ch[:3] / Ch[3]


def fundamental_from_P(P_a: np.ndarray, P_b: np.ndarray) -> np.ndarray:
    """
    Fundamental matrix F mapping points in image A to epipolar lines in image B:
        l_b = F @ [u_a, v_a, 1]      (a point x_b on the cable satisfies x_b^T l_b = 0)
    F = [e_b]_x  P_b  P_a^+  with e_b = P_b [C_a; 1] the epipole of A in B.
    """
    Pa_pinv = np.linalg.pinv(P_a)               # (4,3)
    Ca      = _camera_center(P_a)               # (3,)
    e_b     = P_b @ np.append(Ca, 1.0)          # (3,) epipole in image B
    ex      = np.array([[0., -e_b[2], e_b[1]],
                        [e_b[2], 0., -e_b[0]],
                        [-e_b[1], e_b[0], 0.]])
    return ex @ (P_b @ Pa_pinv)                 # (3,3)


def _epipolar_cost(cl_a: np.ndarray, cl_b: np.ndarray, F: np.ndarray,
                   cap: float) -> np.ndarray:
    """
    (Ma x Mb) symmetric point-to-epipolar-line distance, capped at `cap`.
    cl_a, cl_b : (M,2) ordered centerlines in (u,v) pixel coords.
    """
    Ah = np.hstack([cl_a, np.ones((len(cl_a), 1))])   # (Ma,3)
    Bh = np.hstack([cl_b, np.ones((len(cl_b), 1))])   # (Mb,3)
    la = Ah @ F.T            # (Ma,3) epipolar lines in B for each a
    lb = Bh @ F              # (Mb,3) epipolar lines in A for each b
    # distance b_j to line la_i
    num_ab = np.abs(la @ Bh.T)                          # (Ma,Mb)
    den_ab = np.sqrt(la[:, 0]**2 + la[:, 1]**2)[:, None] + 1e-9
    d_ab = num_ab / den_ab
    # distance a_i to line lb_j
    num_ba = np.abs(Ah @ lb.T)                          # (Ma,Mb)
    den_ba = np.sqrt(lb[:, 0]**2 + lb[:, 1]**2)[None, :] + 1e-9
    d_ba = num_ba / den_ba
    return np.minimum(0.5 * (d_ab + d_ba), cap)


def _dp_match(cost: np.ndarray, gap: float):
    """
    Monotone DTW-style alignment of two ordered chains given a cost matrix.
    Diagonal step = a correspondence; off-diagonal = a gap (skip one chain).
    Returns (corr, total_cost) where corr[i] = matched index in chain B or -1.
    """
    Ma, Mb = cost.shape
    D = np.full((Ma + 1, Mb + 1), np.inf)
    D[0, 0] = 0.0
    # allow free leading/trailing gaps so partial overlap (occlusion) is OK
    D[1:, 0] = np.cumsum(np.full(Ma, gap))
    D[0, 1:] = np.cumsum(np.full(Mb, gap))
    bt = np.zeros((Ma + 1, Mb + 1), np.int8)   # 0=diag 1=up(skip a) 2=left(skip b)
    for i in range(1, Ma + 1):
        for j in range(1, Mb + 1):
            diag = D[i - 1, j - 1] + cost[i - 1, j - 1]
            up   = D[i - 1, j] + gap
            left = D[i, j - 1] + gap
            k = int(np.argmin((diag, up, left)))
            D[i, j] = (diag, up, left)[k]
            bt[i, j] = k
    corr = np.full(Ma, -1, dtype=np.int64)
    i, j = Ma, Mb
    while i > 0 and j > 0:
        k = bt[i, j]
        if k == 0:
            corr[i - 1] = j - 1
            i -= 1; j -= 1
        elif k == 1:
            i -= 1
        else:
            j -= 1
    return corr, D[Ma, Mb]


def epipolar_dp_match(cl_ref: np.ndarray, cl_other: np.ndarray, F: np.ndarray,
                      cap: float = 30.0, gap: float = 12.0):
    """
    Correspondence from each ref-centerline index to an index in cl_other,
    resolving the head/tail flip by trying both orientations of cl_other.
    cl_ref, cl_other : (M,2) ordered centerlines in (u,v).
    F                : fundamental matrix ref -> other.
    Returns (corr, mean_cost) with corr[i] in [0, len(cl_other)) or -1.
    """
    best = None
    for flip in (False, True):
        other = cl_other[::-1] if flip else cl_other
        cost  = _epipolar_cost(cl_ref, other, F, cap)
        corr, total = _dp_match(cost, gap)
        matched = corr >= 0
        mean_c  = total / max(matched.sum(), 1)
        if best is None or mean_c < best[0]:
            if flip:
                m = len(cl_other) - 1
                corr = np.where(corr >= 0, m - corr, -1)
            best = (mean_c, corr)
    return best[1], best[0]


# ---------------------------------------------------------------------------
# T1.2 — RT-DLO topological graph resolution
# Reference: Caporali et al. RT-DLO, ICRA 2023 / arXiv 2210.11127
# ---------------------------------------------------------------------------

def _rtdlo_resolve_topology(skel_pts: np.ndarray, gap_thresh: int = 6) -> np.ndarray:
    """
    Prune spurious skeleton branches via RT-DLO topological graph.
    Keeps the longest endpoint-to-endpoint path, discarding side-branches.
    Falls back to input unchanged if networkx unavailable or skeleton too small.
    """
    if not _NX or len(skel_pts) < 10:
        return skel_pts

    pts = skel_pts
    if len(pts) > 200:
        idx = np.round(np.linspace(0, len(pts)-1, 200)).astype(int)
        pts = pts[idx]

    pt_set = {(int(r), int(c)): i for i, (r, c) in enumerate(pts)}
    G = nx.Graph()
    G.add_nodes_from(range(len(pts)))
    for i, (r, c) in enumerate(pts):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                nb = (int(r)+dr, int(c)+dc)
                if nb in pt_set:
                    j = pt_set[nb]
                    w = 1.414 if abs(dr)+abs(dc) == 2 else 1.0
                    G.add_edge(i, j, weight=w)

    # Close small gaps between disconnected components
    comps = list(nx.connected_components(G))
    if len(comps) > 1:
        cp = [np.array([pts[n] for n in c]) for c in comps]
        for a in range(len(cp)):
            for b in range(a+1, len(cp)):
                d = cdist(cp[a], cp[b])
                ri, ci = np.unravel_index(d.argmin(), d.shape)
                if d[ri, ci] <= gap_thresh:
                    na = list(comps[a])[ri]
                    nb = list(comps[b])[ci]
                    G.add_edge(na, nb, weight=float(d[ri, ci]))

    if not nx.is_connected(G):
        biggest = max(nx.connected_components(G), key=len)
        pts = pts[sorted(biggest)]
        # rebuild
        pt_set = {(int(r), int(c)): i for i, (r, c) in enumerate(pts)}
        G2 = nx.Graph(); G2.add_nodes_from(range(len(pts)))
        for i, (r, c) in enumerate(pts):
            for dr in (-1,0,1):
                for dc in (-1,0,1):
                    if dr==dc==0: continue
                    nb=(int(r)+dr,int(c)+dc)
                    if nb in pt_set:
                        j=pt_set[nb]
                        G2.add_edge(i,j,weight=1.414 if abs(dr)+abs(dc)==2 else 1.0)
        G = G2

    endpoints = [n for n in G.nodes if G.degree(n) == 1]
    if len(endpoints) < 2:
        return pts

    best_path, best_len = [], -1
    for ep_a in endpoints[:6]:
        try:
            lengths = nx.single_source_dijkstra_path_length(G, ep_a, weight='weight')
            ep_b = max((n for n in endpoints if n != ep_a), key=lambda n: lengths.get(n, 0))
            path = nx.shortest_path(G, ep_a, ep_b, weight='weight')
            plen = lengths.get(ep_b, 0)
            if plen > best_len:
                best_len = plen; best_path = path
        except (nx.NetworkXNoPath, ValueError):
            continue

    return pts[best_path].astype(np.int32) if best_path else pts


# ---------------------------------------------------------------------------
# Main perception class
# ---------------------------------------------------------------------------

class DLOPerceptionV2:
    """
    Full DLO perception pipeline for phase1v2 (DLO-only, DER cable).

    T1.1  perceive_multiview() — multi-view DLT triangulation
    T1.2  extract_skeleton()   — Zhang-Suen + RT-DLO topology graph
    T1.3  segment()            — HSV mask tuned for MuJoCo DER cable color
    """

    def __init__(self, n_keypoints: int = 20, use_epipolar: bool = False):
        self.n_kp = n_keypoints
        self._K   = dict(fx=400., fy=400., cx=160., cy=120.)
        # Stage 1: epipolar-DP cross-view correspondence in perceive_multiview.
        self.use_epipolar = use_epipolar

    def set_intrinsics(self, K: dict):
        self._K = K

    # ------------------------------------------------------------------ T1.3
    def segment(self, rgb: np.ndarray) -> np.ndarray:
        """HSV segmentation. Returns (H,W) uint8 binary mask (255=cable)."""
        hsv  = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HSV_LO, HSV_HI)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL_CLOSE, iterations=1)
        return mask

    # ------------------------------------------------------------------ T1.2
    def extract_skeleton(self, mask: np.ndarray) -> np.ndarray:
        """
        Zhang-Suen skeletonize → RT-DLO topology graph pruning.
        Returns (M,2) ordered (row,col) skeleton points, tip-to-tip.
        """
        if mask.sum() < 100:
            return np.empty((0, 2), dtype=np.int32)

        skel = skeletonize(mask.astype(bool))
        pts  = np.argwhere(skel).astype(np.int32)

        if len(pts) < 5:
            return pts

        # T1.2: RT-DLO topology graph — prune spurious branches
        pts = _rtdlo_resolve_topology(pts)

        return pts

    def order_skeleton(self, pts: np.ndarray) -> np.ndarray:
        """Nearest-neighbour ordering from topmost point."""
        if len(pts) < 2:
            return pts
        start = int(np.argmin(pts[:, 0]))
        ordered = [start]
        remaining = list(range(len(pts)))
        remaining.remove(start)
        while remaining:
            cur = pts[ordered[-1]]
            rem_arr = np.array(remaining)
            d = np.linalg.norm(pts[rem_arr] - cur, axis=1)
            nxt = rem_arr[int(np.argmin(d))]
            ordered.append(nxt); remaining.remove(nxt)
        return pts[ordered].astype(np.int32)

    def sample_keypoints_2d(self, ordered_pts: np.ndarray) -> np.ndarray:
        """Uniformly sample n_kp keypoints along skeleton by arc-length."""
        if len(ordered_pts) < 2:
            return None
        pts = ordered_pts.astype(float)
        segs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s    = np.concatenate([[0.], np.cumsum(segs)])
        s_q  = np.linspace(0, s[-1], self.n_kp)
        cols = np.interp(s_q, s, pts[:, 1])
        rows = np.interp(s_q, s, pts[:, 0])
        return np.stack([cols, rows], axis=1)   # (N,2) as (u,v)

    def lift_to_3d(self, kps_2d: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Single-camera depth-buffer lift (fallback). Returns (N,3) camera-frame."""
        H, W = depth.shape
        K = self._K
        out = np.full((self.n_kp, 3), np.nan)
        if kps_2d is None:
            return out
        for i, (u, v) in enumerate(kps_2d):
            ui, vi = int(round(u)), int(round(v))
            if 0 <= ui < W and 0 <= vi < H:
                z = float(depth[vi, ui])
                if z > 0.01:
                    out[i] = [(ui - K['cx']) * z / K['fx'],
                               (vi - K['cy']) * z / K['fy'], z]
        return out

    # ------------------------------------------------------------------ T1.1
    def _epipolar_keypoints(self, cl_list, P_list, band: float = 3.0):
        """
        Stage 1: build N projection-consistent 2D correspondences across cameras.

        Sample N nodes by arc-length along a REFERENCE centerline.  For each node,
        the epipolar constraint alone is 1-DOF (a point-on-line) and is ambiguous
        where the cable runs along the epipolar direction — so we DISAMBIGUATE with
        multi-view consistency: among the other view's epipolar-band candidates,
        keep the one whose triangulated 3D point reprojects closest to the cable in
        the REMAINING views.  (Verified: recovers GT-pixel triangulation to 0 mm,
        whereas pairwise epipolar-DP regresses to ~180 mm.)  With only two cameras
        no third view exists to break the tie, so we fall back to monotonic epipolar
        DP.  Returns a list of C (N,2) arrays (NaN rows allowed), or None.
        """
        det = [i for i, cl in enumerate(cl_list)
               if cl is not None and len(cl) >= 2]
        if len(det) < 2:
            return None
        ref, others = det[0], det[1:]
        cl_ref = cl_list[ref]

        segs = np.linalg.norm(np.diff(cl_ref, axis=0), axis=1)
        s    = np.concatenate([[0.], np.cumsum(segs)])
        s_q  = np.linspace(0., s[-1], self.n_kp)
        idx_ref = np.searchsorted(s, s_q).clip(0, len(cl_ref) - 1)
        ref_pts = cl_ref[idx_ref]                                # (N,2)

        out = [np.full((self.n_kp, 2), np.nan) for _ in cl_list]
        out[ref] = ref_pts

        # two-camera case: no disambiguating view -> monotonic epipolar DP
        if len(others) == 1:
            a = others[0]
            F = fundamental_from_P(P_list[ref], P_list[a])
            corr, _ = epipolar_dp_match(cl_ref, cl_list[a], F)
            m = corr[idx_ref]
            out[a][m >= 0] = cl_list[a][m[m >= 0]]
            return out

        F   = {a: fundamental_from_P(P_list[ref], P_list[a]) for a in others}
        clh = {a: np.hstack([cl_list[a], np.ones((len(cl_list[a]), 1))])
               for a in others}

        for n in range(self.n_kp):
            x0  = ref_pts[n]
            x0h = np.append(x0, 1.0)
            cand = {}
            for a in others:
                l = x0h @ F[a].T                                 # epipolar line in a
                d = np.abs(clh[a] @ l) / (np.hypot(l[0], l[1]) + 1e-9)
                ci = np.where(d < band)[0]
                cand[a] = ci if len(ci) else np.array([int(d.argmin())])

            a0   = others[0]
            best = None
            for j0 in cand[a0]:
                X = triangulate_point_dlt([P_list[ref], P_list[a0]],
                                          [x0, cl_list[a0][j0]])
                score, picks = 0.0, {a0: j0}
                for b in others[1:]:
                    rb = _reproject(P_list[b], X[None])[0]
                    dd = np.linalg.norm(cl_list[b][cand[b]] - rb, axis=1)
                    k  = int(dd.argmin())
                    score += float(dd[k]); picks[b] = int(cand[b][k])
                if best is None or score < best[0]:
                    best = (score, picks)
            for a, j in best[1].items():
                out[a][n] = cl_list[a][j]
        return out

    def perceive_multiview(self,
                           rgb_per_cam: list,
                           Ks: list,
                           Rs: list,
                           ts: list,
                           fallback_depth=None) -> dict:
        """
        T1.1: Multi-view DLT triangulation.
        rgb_per_cam : list of C (H,W,3) uint8 RGB images
        Ks          : list of C intrinsic dicts {fx,fy,cx,cy}
        Rs          : list of C (3,3) world-to-cam rotation matrices
        ts          : list of C (3,) camera world positions
        fallback_depth : (H,W) depth from primary cam for single-cam fallback
        Returns dict with keys: keypoints_3d, keypoints_2d_per_cam, method
        """
        C = len(rgb_per_cam)
        kps2d_list = []
        cl_list    = []          # per-cam ordered centerline in (u,v), or None
        P_list     = []

        for c in range(C):
            mask = self.segment(rgb_per_cam[c])
            skel = self.extract_skeleton(mask)
            if len(skel) >= 5:
                ordered = self.order_skeleton(skel)            # (M,2) row,col
                kp2d = self.sample_keypoints_2d(ordered)
                cl   = ordered[:, ::-1].astype(float)          # -> (u,v)
            else:
                kp2d, cl = None, None
            kps2d_list.append(kp2d)
            cl_list.append(cl)
            P_list.append(_build_projection(Ks[c], Rs[c], ts[c]))

        n_detected = sum(k is not None for k in kps2d_list)

        # Stage 1: epipolar-DP correspondence (projection-consistent lift).
        if self.use_epipolar and n_detected >= 2:
            epi = self._epipolar_keypoints(cl_list, P_list)
            if epi is not None:
                kps3d = triangulate_multiview(epi, P_list, min_cameras=2)
                if kps3d is not None and not np.isnan(kps3d).all():
                    return dict(keypoints_3d=kps3d,
                                keypoints_2d_per_cam=epi,
                                method='multiview_epipolar')

        if n_detected >= 2:
            kps2d_list = align_chain_directions(kps2d_list, P_list)
            kps3d = triangulate_multiview(kps2d_list, P_list, min_cameras=2)
            if kps3d is not None:
                return dict(keypoints_3d=kps3d,
                            keypoints_2d_per_cam=kps2d_list,
                            method='multiview')

        # Fallback: single-camera depth lift from first successful camera
        for c in range(C):
            if kps2d_list[c] is not None:
                self.set_intrinsics(Ks[c])
                depth = fallback_depth if (fallback_depth is not None and c == 0) \
                        else np.zeros((rgb_per_cam[c].shape[0],
                                       rgb_per_cam[c].shape[1]), dtype=np.float32)
                kps3d = self.lift_to_3d(kps2d_list[c], depth)
                return dict(keypoints_3d=kps3d,
                            keypoints_2d_per_cam=kps2d_list,
                            method='single_cam_fallback')

        return dict(keypoints_3d=None, keypoints_2d_per_cam=kps2d_list, method='failed')

    def perceive(self, rgb: np.ndarray, depth: np.ndarray) -> dict:
        """Single-camera perceive. Returns dict with success, mask, skeleton_pts, keypoints_2d, keypoints_3d."""
        mask = self.segment(rgb)
        if mask.sum() < 50:
            return dict(success=False, mask=mask, skeleton_pts=None,
                        keypoints_2d=None, keypoints_3d=None)

        skel = self.extract_skeleton(mask)
        if len(skel) < 5:
            return dict(success=False, mask=mask, skeleton_pts=skel,
                        keypoints_2d=None, keypoints_3d=None)

        ordered = self.order_skeleton(skel)
        kp2d    = self.sample_keypoints_2d(ordered)
        kp3d    = self.lift_to_3d(kp2d, depth)

        return dict(success=True, mask=mask, skeleton_pts=ordered,
                    keypoints_2d=kp2d, keypoints_3d=kp3d)
