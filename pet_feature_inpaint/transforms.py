"""Geometric transforms painting-crop -> photo-crop for PetFeatureMatch, fitted to point matches.

All coordinates are normalised to [-1, 1] (x, y). A transform T maps a painting position to the photo
position that is sampled there (grid_sample convention), i.e. we fit T(pa) ~ pb.

Global models (batched closed-form fits, so robust estimators can try thousands of minimal samples):
    rigid (3 DOF), similarity (4), affine (6), homography (8), poly2 (12), poly3 (20)
Robust estimators: ransac (MSAC score + local LS refit), lmeds, huber (IRLS), tukey (IRLS)
Non-rigid models on top of a global one (fitted on its inliers, Tukey-reweighted):
    tps   - thin-plate spline, smoothing parameter lam (0 = exact interpolation)
    ffd   - cubic B-spline free-form deformation on a uniform control grid, 2nd-difference penalty

Every fitted transform is a callable: T(x) with x (..., 2) -> (..., 2)."""
import math

import torch

# ------------------------------------------------------------------ helpers


def _wn(w, shape):
    if w is None:
        return torch.ones(shape)
    return w.float().clamp(min=0)


def ident_pts(n):
    u = torch.linspace(-1, 1, n)
    return torch.stack(torch.meshgrid(u, u, indexing="ij")[::-1], -1)  # (n, n, 2) x, y


def jac_det(T, pts, eps=1e-3):
    """Jacobian determinant of T at pts (m, 2) by central differences."""
    ex = torch.tensor([eps, 0.0])
    ey = torch.tensor([0.0, eps])
    dx = (T(pts + ex) - T(pts - ex)) / (2 * eps)
    dy = (T(pts + ey) - T(pts - ey)) / (2 * eps)
    return dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]


# ------------------------------------------------------------------ global models (batched)
def _monomials(x, order):
    X, Y = x[..., 0], x[..., 1]
    return torch.stack([X ** i * Y ** (d - i) for d in range(order + 1) for i in range(d, -1, -1)], -1)


class Global:
    """Batched fit: pa, pb (B, n, 2), w (B, n) -> params (B, ...). apply(params, x (B, m, 2))."""
    name, k, dof = "", 0, 0

    def fit(self, pa, pb, w):
        raise NotImplementedError

    def apply(self, P, x):
        raise NotImplementedError

    def make(self, P):
        """Single param set -> callable on (..., 2)."""
        def T(x):
            sh = x.shape
            return self.apply(P[None], x.reshape(1, -1, 2))[0].reshape(sh)
        T.model = self.name
        return T


class Euclid(Global):
    """Rigid (scale=False) or similarity (scale=True): weighted Umeyama / Kabsch."""

    def __init__(self, scale):
        self.scale = scale
        self.name = "similarity" if scale else "rigid"
        self.k, self.dof = 2, (4 if scale else 3)

    def fit(self, pa, pb, w):
        w = w / w.sum(1, keepdim=True).clamp(min=1e-9)
        ma = (w[..., None] * pa).sum(1, keepdim=True)
        mb = (w[..., None] * pb).sum(1, keepdim=True)
        A, B = pa - ma, pb - mb
        C = (B * w[..., None]).transpose(1, 2) @ A  # B x 2 x 2
        U, S, Vt = torch.linalg.svd(C)
        d = torch.sign(torch.linalg.det(U @ Vt))
        d = torch.where(d == 0, torch.ones_like(d), d)
        D = torch.diag_embed(torch.stack([torch.ones_like(d), d], -1))
        R = U @ D @ Vt
        if self.scale:
            va = (w * (A ** 2).sum(-1)).sum(1).clamp(min=1e-9)
            s = (S * torch.diagonal(D, dim1=1, dim2=2)).sum(-1) / va
        else:
            s = torch.ones(len(R))
        L = s[:, None, None] * R
        t = mb[:, 0] - (L @ ma[:, 0, :, None])[..., 0]
        return torch.cat([L, t[..., None]], 2)

    def apply(self, P, x):
        return x @ P[:, :, :2].transpose(1, 2) + P[:, None, :, 2]


class Affine(Euclid):
    name, k, dof = "affine", 3, 6

    def __init__(self):
        pass

    def fit(self, pa, pb, w):
        X = torch.cat([pa, torch.ones_like(pa[..., :1])], -1)
        sw = w.sqrt()[..., None]
        sol = torch.linalg.lstsq(X * sw, pb * sw, driver="gelsd" if X.device.type == "cpu" else None).solution
        return sol.transpose(1, 2)  # B x 2 x 3


class Poly(Global):
    def __init__(self, order):
        self.order = order
        self.name = "poly%d" % order
        self.m = (order + 1) * (order + 2) // 2
        self.k, self.dof = self.m, 2 * self.m

    def fit(self, pa, pb, w):
        X = _monomials(pa, self.order)
        sw = w.sqrt()[..., None]
        return torch.linalg.lstsq(X * sw, pb * sw, driver="gelsd").solution  # B x m x 2

    def apply(self, P, x):
        return _monomials(x, self.order) @ P


class Homography(Global):
    name, k, dof = "homography", 4, 8

    @staticmethod
    def _norm(p, w):
        ws = w / w.sum(1, keepdim=True).clamp(min=1e-9)
        m = (ws[..., None] * p).sum(1, keepdim=True)
        d = (ws * (p - m).norm(dim=-1)).sum(1).clamp(min=1e-6)
        s = math.sqrt(2) / d
        T = torch.zeros(len(p), 3, 3)
        T[:, 0, 0] = s
        T[:, 1, 1] = s
        T[:, 0, 2] = -s * m[:, 0, 0]
        T[:, 1, 2] = -s * m[:, 0, 1]
        T[:, 2, 2] = 1
        return (p - m) * s[:, None, None], T

    def fit(self, pa, pb, w):
        a, Ta = self._norm(pa, w)
        b, Tb = self._norm(pb, w)
        x, y = a[..., 0], a[..., 1]
        u, v = b[..., 0], b[..., 1]
        o, z = torch.ones_like(x), torch.zeros_like(x)
        r1 = torch.stack([x, y, o, z, z, z, -u * x, -u * y, -u], -1)
        r2 = torch.stack([z, z, z, x, y, o, -v * x, -v * y, -v], -1)
        sw = w.sqrt()[..., None]
        M = torch.cat([r1 * sw, r2 * sw], 1)  # B x 2n x 9
        if M.shape[1] < 9:  # minimal sample: pad so the SVD returns the null vector
            M = torch.cat([M, torch.zeros(len(M), 9 - M.shape[1], 9)], 1)
        _, _, Vh = torch.linalg.svd(M)
        Hn = Vh[:, -1].reshape(-1, 3, 3)
        H = torch.linalg.inv(Tb) @ Hn @ Ta
        return H / H[:, 2:3, 2:3].where(H[:, 2:3, 2:3].abs() > 1e-9, torch.ones_like(H[:, 2:3, 2:3]))

    def apply(self, P, x):
        X = torch.cat([x, torch.ones_like(x[..., :1])], -1) @ P.transpose(1, 2)
        z = X[..., 2:3]
        z = torch.where(z.abs() < 1e-6, torch.full_like(z, 1e-6), z)
        return X[..., :2] / z


GLOBAL = {"rigid": Euclid(False), "similarity": Euclid(True), "affine": Affine(), "homography": Homography(),
          "poly2": Poly(2), "poly3": Poly(3)}


# ------------------------------------------------------------------ validity (no folds, sane scale)
def _valid(model, P, check, det_lo=0.3, det_hi=3.5):
    """Batched: Jacobian determinant of every candidate on the check points stays in [lo, hi] (no fold,
    no extreme stretch where the mask can land). Homographies must not put the horizon line inside."""
    B = len(P)
    if check is None or len(check) == 0:
        return torch.ones(B, dtype=torch.bool)
    c = check[None].expand(B, -1, -1)
    eps = 1e-3
    ex, ey = torch.tensor([eps, 0.0]), torch.tensor([0.0, eps])
    dx = (model.apply(P, c + ex) - model.apply(P, c - ex)) / (2 * eps)
    dy = (model.apply(P, c + ey) - model.apply(P, c - ey)) / (2 * eps)
    det = dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]
    ok = (det > det_lo).all(1) & (det < det_hi).all(1) & torch.isfinite(det).all(1)
    if isinstance(model, Homography):
        z = torch.cat([c, torch.ones_like(c[..., :1])], -1) @ P[:, 2:3, :].transpose(1, 2)
        ok &= (z[..., 0] > 0.2).all(1)
    return ok


def _degenerate(pa):
    """Batched minimal samples: any three points (nearly) collinear -> True."""
    k = pa.shape[1]
    bad = torch.zeros(len(pa), dtype=torch.bool)
    if k < 3:
        return (pa[:, 0] - pa[:, 1]).norm(dim=-1) < 0.05
    import itertools
    for i, j, l in itertools.combinations(range(min(k, 6)), 3):
        a = pa[:, j] - pa[:, i]
        b = pa[:, l] - pa[:, i]
        bad |= (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).abs() / 2 < 2e-3
    return bad


def _res(model, P, pa, pb):
    return (model.apply(P, pa[None].expand(len(P), -1, -1)) - pb[None]).norm(dim=-1)  # B x n


def _ls(model, pa, pb, w):
    return model.fit(pa[None], pb[None], w[None])


# ------------------------------------------------------------------ robust estimators
def ransac(model, pa, pb, w=None, thr=0.05, iters=1500, check=None, seed=0, lmeds=False):
    """RANSAC (MSAC score) or LMedS over minimal samples; final weighted LS on the inliers, one LO step.
    Returns (params, inlier mask) or (None, None)."""
    n = len(pa)
    w = _wn(w, n)
    if n < max(model.k + 2, 6):
        return None, None
    g = torch.Generator().manual_seed(seed)
    idx = torch.argsort(torch.rand(iters, n, generator=g), 1)[:, :model.k]
    sa, sb = pa[idx], pb[idx]
    keep = ~_degenerate(sa)
    sa, sb = sa[keep], sb[keep]
    if len(sa) == 0:
        return None, None
    best_P, best_s = None, None
    for c0 in range(0, len(sa), 500):  # chunks keep the memory small
        P = model.fit(sa[c0:c0 + 500], sb[c0:c0 + 500], torch.ones(sa[c0:c0 + 500].shape[:2]))
        fin = torch.isfinite(P.reshape(len(P), -1)).all(1)
        ok = fin & _valid(model, torch.nan_to_num(P), check)
        r = _res(model, torch.nan_to_num(P), pa, pb)
        if lmeds:
            s = -(r ** 2).median(1).values  # higher is better
        else:
            s = -(r.clamp(max=thr) ** 2 * w).sum(1)  # MSAC
        s[~ok] = -float("inf")
        i = s.argmax()
        if best_s is None or s[i] > best_s:
            best_s, best_P = s[i], P[i:i + 1]
    if best_P is None or not torch.isfinite(best_s):
        return None, None
    r = _res(model, best_P, pa, pb)[0]
    if lmeds:
        sigma = 1.4826 * (1 + 5 / max(1, n - model.k)) * (r ** 2).median().sqrt()
        thr = max(2.5 * float(sigma), 1e-3)
    for _ in range(2):  # local optimisation: LS on the inliers, re-select
        inl = r < thr
        if inl.sum() < model.k + 2:
            return None, None
        P = _ls(model, pa[inl], pb[inl], w[inl])
        if not _valid(model, P, check)[0]:
            break
        best_P = P
        r = _res(model, best_P, pa, pb)[0]
    inl = r < thr
    if inl.sum() < max(6, 0.15 * n):
        return None, None
    return best_P[0], inl


def lmeds(model, pa, pb, w=None, thr=None, iters=1500, check=None, seed=0):
    return ransac(model, pa, pb, w, thr=thr or 0.05, iters=iters, check=check, seed=seed, lmeds=True)


def irls(model, pa, pb, w=None, kind="huber", iters=30, check=None, init=None, thr=0.05):
    """Huber / Tukey M-estimator by iteratively reweighted least squares. Scale from the MAD.
    Tukey starts from Huber (or `init`); its weights reach 0, so gross mismatches drop out."""
    n = len(pa)
    w0 = _wn(w, n)
    if n < model.k + 2:
        return None, None
    P = init[None] if init is not None else _ls(model, pa, pb, w0)
    if kind == "tukey" and init is None:
        P, _ = irls(model, pa, pb, w, "huber", iters, None)
        if P is None:
            return None, None
        P = P[None]
    for _ in range(iters):
        r = _res(model, P, pa, pb)[0]
        sigma = max(1.4826 * float(r.median()), 1e-4)
        if kind == "huber":
            c = 1.345 * sigma
            rw = torch.where(r <= c, torch.ones_like(r), c / r.clamp(min=1e-9))
        else:
            c = 4.685 * sigma
            rw = ((1 - (r / c) ** 2).clamp(min=0)) ** 2
        if (rw > 0).sum() < model.k + 2:
            break
        Pn = _ls(model, pa, pb, w0 * rw)
        if not torch.isfinite(Pn).all():
            break
        done = (model.apply(Pn, pa[None]) - model.apply(P, pa[None])).norm(dim=-1).max() < 1e-5
        P = Pn
        if done:
            break
    if not _valid(model, P, check)[0]:
        return None, None
    r = _res(model, P, pa, pb)[0]
    inl = r < thr
    if inl.sum() < max(6, 0.15 * n):
        return None, None
    return P[0], inl


ESTIMATORS = ["ransac", "lmeds", "huber", "tukey"]


def fit_global(name, est, pa, pb, w=None, thr=0.05, check=None, seed=0):
    """-> (callable T, inlier mask) or (None, None)."""
    model = GLOBAL[name]
    if est == "ransac":
        P, inl = ransac(model, pa, pb, w, thr=thr, check=check, seed=seed)
    elif est == "lmeds":
        P, inl = lmeds(model, pa, pb, w, thr=thr, check=check, seed=seed)
    elif est in ("huber", "tukey"):
        P, inl = irls(model, pa, pb, w, est, check=check, thr=thr)
    elif est == "ransac+tukey":
        P0, _ = ransac(model, pa, pb, w, thr=thr, check=check, seed=seed)
        P, inl = (None, None) if P0 is None else irls(model, pa, pb, w, "tukey", check=check, init=P0, thr=thr)
    else:
        raise ValueError(est)
    if P is None:
        return None, None
    return model.make(P), inl


# ------------------------------------------------------------------ non-rigid on top of a global transform
def _tukey_w(r):
    sigma = max(1.4826 * float(r.median()), 1e-4)
    c = 4.685 * sigma
    return ((1 - (r / c) ** 2).clamp(min=0)) ** 2


def _tps_U(d2):
    return 0.5 * d2 * torch.log(d2.clamp(min=1e-12))  # r^2 log r


def fit_tps(T0, pa, pb, w=None, lam=0.01, max_ctrl=200, robust=2, seed=0, keep=None):
    """Smoothing thin-plate spline on the residual of T0 (T = T0 + tps). lam is relative to the mean
    kernel magnitude; 0 interpolates every control point exactly. keep: indices that are always control
    points and are never down-weighted by the robust passes (landmark anchors)."""
    n = len(pa)
    w = _wn(w, n)
    if n > max_ctrl:  # farthest-point subsample: even coverage, bounded cost
        g = torch.Generator().manual_seed(seed)
        sel = [int(torch.randint(n, (1,), generator=g))] if keep is None or len(keep) == 0 else [int(i) for i in keep]
        d = torch.stack([(pa - pa[i]).norm(dim=1) for i in sel]).min(0).values
        for _ in range(max_ctrl - len(sel)):
            i = int(d.argmax())
            sel.append(i)
            d = torch.minimum(d, (pa - pa[i]).norm(dim=1))
        sel = torch.tensor(sel)
    else:
        sel = torch.arange(n)
    pinned = torch.zeros(len(sel), dtype=torch.bool)
    if keep is not None and len(keep):
        pinned = torch.isin(sel, torch.as_tensor(keep))
    c, e0, wc = pa[sel], pb[sel] - T0(pa[sel]), w[sel]
    K = _tps_U(torch.cdist(c, c) ** 2)
    m = len(c)
    Pm = torch.cat([torch.ones(m, 1), c], 1)
    scale = K.abs().mean().clamp(min=1e-6)
    rw = torch.ones(m)
    for it in range(robust + 1):
        ww = (wc * rw).clamp(min=1e-3)
        A = torch.zeros(m + 3, m + 3)
        A[:m, :m] = K + lam * scale * torch.diag(1 / ww)  # low-weight points are smoothed more
        A[:m, m:] = Pm
        A[m:, :m] = Pm.T
        rhs = torch.cat([e0, torch.zeros(3, 2)])
        sol = torch.linalg.lstsq(A, rhs).solution
        Wk, a = sol[:m], sol[m:]
        if it < robust and lam > 0:
            fitv = K @ Wk + Pm @ a
            rw = _tukey_w((fitv - e0).norm(dim=1))
            rw[pinned] = 1.0

    def T(x):
        sh = x.shape
        x2 = x.reshape(-1, 2)
        out = T0(x2)
        for i in range(0, len(x2), 65536):
            xi = x2[i:i + 65536]
            U = _tps_U(torch.cdist(xi, c) ** 2)
            out[i:i + 65536] = out[i:i + 65536] + U @ Wk + torch.cat([torch.ones(len(xi), 1), xi], 1) @ a
        return out.reshape(sh)
    T.model = "tps"
    return T


def _bspline_basis(t):
    """Cubic B-spline weights for fractional position t (n,) -> (n, 4)."""
    t2, t3 = t * t, t * t * t
    return torch.stack([(1 - t) ** 3 / 6, (3 * t3 - 6 * t2 + 4) / 6, (-3 * t3 + 3 * t2 + 3 * t + 1) / 6, t3 / 6], -1)


def _ffd_design(x, G):
    """Sparse-ish dense design matrix (n, (G+3)^2): control grid of G intervals over [-1, 1] plus the
    border ring a cubic spline needs."""
    u = (x[:, 0].clamp(-1, 1) + 1) / 2 * G
    v = (x[:, 1].clamp(-1, 1) + 1) / 2 * G
    iu = u.floor().clamp(max=G - 1)
    iv = v.floor().clamp(max=G - 1)
    bu, bv = _bspline_basis(u - iu), _bspline_basis(v - iv)
    N = G + 3
    D = torch.zeros(len(x), N * N)
    for a in range(4):
        for b in range(4):
            col = (iv.long() + b) * N + (iu.long() + a)
            D.scatter_add_(1, col[:, None], (bu[:, a] * bv[:, b])[:, None])
    return D


def fit_ffd(T0, pa, pb, w=None, grid=6, lam=0.1, robust=2):
    """B-spline free-form deformation on the residual of T0; penalty = squared 2nd differences of the
    control displacements (bending), lam relative to the data term."""
    n = len(pa)
    w = _wn(w, n)
    e0 = pb - T0(pa)
    D = _ffd_design(pa, grid)
    N = grid + 3
    # 2nd-difference operator in x and y on the control grid
    rows = []
    for j in range(N):
        for i in range(N):
            for di, dj in ((1, 0), (0, 1)):
                if 0 < i < N - 1 if di else 0 < j < N - 1:
                    r = torch.zeros(N * N)
                    r[j * N + i] = -2
                    r[(j - dj) * N + (i - di)] = 1
                    r[(j + dj) * N + (i + di)] = 1
                    rows.append(r)
    L = torch.stack(rows)
    LtL = L.T @ L
    rw = torch.ones(n)
    for it in range(robust + 1):
        ww = w * rw
        A = (D * ww[:, None]).T @ D + lam * ww.sum() / N ** 2 * LtL + 1e-6 * torch.eye(N * N)
        C = torch.linalg.solve(A, (D * ww[:, None]).T @ e0)  # (N*N, 2)
        if it < robust:
            rw = _tukey_w((D @ C - e0).norm(dim=1))

    def T(x):
        sh = x.shape
        x2 = x.reshape(-1, 2)
        out = T0(x2)
        for i in range(0, len(x2), 65536):
            out[i:i + 65536] = out[i:i + 65536] + _ffd_design(x2[i:i + 65536], grid) @ C
        return out.reshape(sh)
    T.model = "ffd"
    return T


def grid_of(T, n):
    """Sampling grid (1, n, n, 2) for F.grid_sample."""
    return T(ident_pts(n).reshape(-1, 2)).reshape(1, n, n, 2)
