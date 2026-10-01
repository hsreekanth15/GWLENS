"""
Amplification factor for an ARBITRARY lens potential psi(x1, x2) with the rotated-tail method.

    F(w, y) = w/(2 pi i) Int d^2x exp{ i w [ |x - y|^2/2 - psi(x) ] }

What you supply
---------------
  psi(x1, x2)   the lens potential. It is called with real arrays (head of the radial
                integral) and with COMPLEX arrays (rotated tail), so write it with analytic
                numpy operations only:  + - * / **, np.log, np.sqrt, np.exp, np.arctan,
                np.arccos, np.arcsinh, ...  Do NOT use np.abs, np.hypot, np.arctan2,
                np.maximum/minimum, np.where, np.real or if/else on the coordinates: they
                break the analytic continuation. If a formula needs complex intermediates on
                the real plane (e.g. arccos of numbers > 1), cast to complex inside; the
                solver uses the real part on the real plane.
  singular      the points where psi is not smooth: point masses, SIS / NFW centres, and
                (recommended) the centres of very small cores. Nothing else is needed.
  q_inf         optional: kappa + |gamma| of any external sheet / shear (the limit of
                (d psi/dr)/r at large r). Estimated automatically if omitted. Must be < 1.

    lens = Lens(my_psi, singular=[(x1a, x2a), (x1b, x2b)])
    F = amplification(w, y, lens)
    F, err = amplification(w, y, lens, estimate_error=True)     # self-consistency check

Ready-made components (they add with +):  point_mass, sis, cis, nfw, external_shear.

Method
------
Partition of unity 1 = chi_0(x) + sum_k chi_k(x), with chi_k a C-infinity bump (Planck taper)
that equals 1 within s0*rho_k of singular point x_k and 0 beyond rho_k. rho_k is smaller than
the distance to any other singular point (default 0.8 of it, at most 1), so no patch contains a
second singularity; patches may overlap, since chi_0 = 1 - sum chi_k need not be positive.

  * patch k      Int chi_k e^{iwT} over the disc |x - x_k| < rho_k, in polar coordinates about
                 x_k: the singularity is at the polar origin, where the radial grid resolves it
                 exactly as in the 1D code. Finite domain: no tail.
  * background   Int chi_0 e^{iwT} in polar coordinates about a centre c (default: the singular
                 point nearest the source, which then needs no patch). chi_0 vanishes near every
                 other singular point, so the integrand is smooth there. Beyond the split radius
                 r_b, chi_0 = 1 and the radial tail is rotated to u = u_b + i t exactly as in 1D.

Radial integrals: Gauss-Legendre on pieces holding <= one cycle of the phase. Angular integrals:
trapezoid rule (periodic integrand), with the number of angles chosen per radial band (octaves in
r) from the local angular bandwidth, so the strongly oscillating region near a point mass costs
few angles.
"""
import warnings
import numpy as np
from scipy.special import roots_legendre, roots_laguerre, expit

_GL, _LAG = {}, {}


def _gl(n):
    if n not in _GL:
        _GL[n] = roots_legendre(n)
    return _GL[n]


def _lag(n):
    if n not in _LAG:
        _LAG[n] = roots_laguerre(n)
    return _LAG[n]


def _unique_points(pts, tol=1e-12):
    out = []
    for p in pts:
        p = (float(p[0]), float(p[1]))
        if not any(np.hypot(p[0] - q[0], p[1] - q[1]) < tol for q in out):
            out.append(p)
    return out


# ================================================================== lens objects
class Lens:
    """A lens potential psi(x1, x2) plus the list of its singular points.

    psi       callable psi(x1, x2) -> array; analytic operations only (see module docstring)
    singular  iterable of (x1, x2) points where psi is not smooth
    q_inf     kappa + |gamma| of external sheet / shear (None: estimate automatically)
    """

    def __init__(self, psi, singular=(), q_inf=None):
        self._terms = [psi]
        self.singular = _unique_points(singular)
        self.q_inf = q_inf
        self._checked = False

    def psi(self, x1, x2):
        out = self._terms[0](x1, x2)
        for f in self._terms[1:]:
            out = out + f(x1, x2)
        return out

    __call__ = psi

    def __add__(self, other):
        new = Lens.__new__(Lens)
        new._terms = self._terms + other._terms
        new.singular = _unique_points(self.singular + other.singular)
        new.q_inf = None if (self.q_inf is None or other.q_inf is None) else self.q_inf + other.q_inf
        new._checked = False
        return new

    def grad(self, x1, x2, h=1e-20):
        """(d psi/d x1, d psi/d x2) at real points by the complex step (exact to rounding)."""
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)
        g1 = np.imag(self.psi(x1 + 1j * h, x2 + 0j)) / h
        g2 = np.imag(self.psi(x1 + 0j, x2 + 1j * h)) / h
        return g1, g2

    def check_analytic(self, n=400, seed=0, verbose=False):
        """Compare complex-step and finite-difference gradients at random points.
        A mismatch means psi uses a non-analytic operation (abs, hypot, arctan2, where, ...),
        which would make the rotated tail wrong. Returns the worst relative mismatch."""
        rng = np.random.default_rng(seed)
        P = np.array(self.singular) if self.singular else np.zeros((1, 2))
        c = P.mean(axis=0)
        R = 2.0 + np.max(np.hypot(*(P - c).T))
        x = c + rng.uniform(-R, R, size=(n, 2))
        dmin = np.min(np.hypot(x[:, None, 0] - P[None, :, 0], x[:, None, 1] - P[None, :, 1]), axis=1)
        x = x[dmin > 0.05]
        with np.errstate(all="ignore"):
            try:
                g1, g2 = self.grad(x[:, 0], x[:, 1])
            except TypeError as err:
                raise TypeError("psi(x1, x2) must accept complex arrays (it is evaluated on the "
                                f"rotated tail); use analytic numpy operations only. ({err})") from None
            h = 1e-6
            f = lambda a, b: np.real(self.psi(a + 0j, b + 0j))
            d1 = (f(x[:, 0] + h, x[:, 1]) - f(x[:, 0] - h, x[:, 1])) / (2 * h)
            d2 = (f(x[:, 0], x[:, 1] + h) - f(x[:, 0], x[:, 1] - h)) / (2 * h)
            im = np.imag(self.psi(x[:, 0] + 0j, x[:, 1] + 0j))
            re = np.real(self.psi(x[:, 0] + 0j, x[:, 1] + 0j))
        mism = np.max(np.hypot(g1 - d1, g2 - d2) / (1 + np.hypot(d1, d2)))
        imag = np.max(np.abs(im) / (1 + np.abs(re)))
        bad = (not np.isfinite(mism)) or mism > 1e-5 or imag > 1e-10
        if bad:
            warnings.warn(
                f"psi does not look analytic (gradient mismatch {mism:.1e}, imaginary part on the "
                f"real plane {imag:.1e}). Use only analytic numpy operations (no abs/hypot/"
                "arctan2/where/real/if-else on x); otherwise the rotated tail is invalid.")
        if verbose:
            print(f"check_analytic: gradient mismatch {mism:.1e}, Im psi on real plane {imag:.1e}")
        self._checked = True
        return mism


# ------------------------------------------------------------------ ready-made components
def point_mass(m=1.0, x0=(0.0, 0.0)):
    """psi = m ln|x - x0|."""
    a, b, m = float(x0[0]), float(x0[1]), float(m)
    return Lens(lambda x1, x2: 0.5 * m * np.log((x1 - a)**2 + (x2 - b)**2), singular=[(a, b)], q_inf=0.0)


def sis(psi0=1.0, x0=(0.0, 0.0)):
    """psi = psi0 |x - x0|."""
    a, b, p = float(x0[0]), float(x0[1]), float(psi0)
    return Lens(lambda x1, x2: p * np.sqrt((x1 - a)**2 + (x2 - b)**2), singular=[(a, b)], q_inf=0.0)


def cis(psi0=1.0, rc=0.05, x0=(0.0, 0.0)):
    """Cored isothermal sphere (GLoW convention): psi0 [s + rc ln(2 rc/(s + rc))], s = sqrt(R^2 + rc^2).
    The centre is listed as a singular point so that a small core is resolved like a cusp."""
    a, b, p, rc = float(x0[0]), float(x0[1]), float(psi0), float(rc)

    def f(x1, x2):
        R2 = (x1 - a)**2 + (x2 - b)**2
        if rc == 0:
            return p * np.sqrt(R2)
        s = np.sqrt(R2 + rc * rc)
        return p * (s + rc * np.log(2 * rc / (s + rc)))
    return Lens(f, singular=[(a, b)], q_inf=0.0)


def nfw(k=1.0, psi0=1.0, x0=(0.0, 0.0)):
    """NFW: 0.5 psi0 [ln^2(v/2) + arccos^2(1/v)], v = k |x - x0|  (k = xi0 / r_s)."""
    a, b, p, k = float(x0[0]), float(x0[1]), float(psi0), float(k)

    def f(x1, x2):
        v = k * np.sqrt((x1 - a)**2 + (x2 - b)**2 + 0j)       # complex: arccos(1/v) for v < 1
        return 0.5 * p * (np.log(v / 2)**2 + np.arccos(1 / v)**2)
    return Lens(f, singular=[(a, b)], q_inf=0.0)


def external_shear(gamma1=0.0, gamma2=0.0, kappa=0.0):
    """psi = kappa |x|^2/2 + [gamma1 (x1^2 - x2^2) + 2 gamma2 x1 x2]/2."""
    g1, g2, k = float(gamma1), float(gamma2), float(kappa)
    f = lambda x1, x2: 0.5 * (k * (x1 * x1 + x2 * x2) + g1 * (x1 * x1 - x2 * x2) + 2 * g2 * x1 * x2)
    return Lens(f, singular=[], q_inf=k + np.hypot(g1, g2))


# ================================================================== helpers
def _step(s, s0, b=2.5):
    """C-infinity step: 1 for s <= s0, 0 for s >= 1.
    Planck taper 1/(1 + exp(b [1/(1-t) - 1/t])), t = (s - s0)/(1 - s0); b = 2.5 minimises the
    number of Fourier modes (hence trapezoid angles) needed to resolve it to ~1e-12."""
    t = np.clip((np.asarray(s, dtype=float) - s0) / (1.0 - s0), 0.0, 1.0)
    inner = (t > 0) & (t < 1)
    tt = np.where(inner, t, 0.5)
    z = b * (1.0 / (1.0 - tt) - 1.0 / tt)
    return np.where(t <= 0, 1.0, np.where(t >= 1, 0.0, expit(-z)))


def _estimate_q_inf(lens, c):
    th = np.linspace(0, 2 * np.pi, 257)[:-1]
    q = -np.inf
    for R in (1e3, 1e4):
        g1, g2 = lens.grad(c[0] + R * np.cos(th), c[1] + R * np.sin(th))
        q = max(q, np.max((np.cos(th) * g1 + np.sin(th) * g2) / R))
    return float(q)


def _polar_integral(w, lens, c, yc, ub, ngl, nlag, tail, theta_factor, max_chunk,
                    weight=None, radial_weight=None, extra_r=(), ang_req=None, info=None):
    """Int_0^{2pi} dth Int_0^{ub} du  W  e^{i w phi}   (+ rotated tail u_b + i t if `tail`)

    phi = u - r n.yc - psi(c + r n),  r = sqrt(2u),  n = (cos th, sin th)
    W   = weight(x1, x2) (position dependent, background) or radial_weight(r) (patch), or 1.
    """
    c = np.asarray(c, dtype=float)
    yabs = float(np.hypot(*yc))
    rmax = np.sqrt(2 * ub)
    umin = 1e-14 / max(w, 1.0)

    def eval_phase(r, th2d):
        x1 = c[0] + r * np.cos(th2d)
        x2 = c[1] + r * np.sin(th2d)
        P = np.real(lens.psi(x1, x2))
        return x1, x2, P

    # ---------- radial pieces: cut by the worst-angle accumulated phase
    n_lin = int(max(3000, 8 * w * ub / (2 * np.pi)))
    ug = np.unique(np.concatenate([np.geomspace(umin, ub, 4000), np.linspace(umin, ub, n_lin)]))
    xg = np.sqrt(2 * ug)
    thp = np.linspace(0, 2 * np.pi, 129)[:-1]
    dV = np.zeros(ug.size - 1)
    step = max(1, int(2_000_000 // ug.size))
    with np.errstate(all="ignore"):
        for s in range(0, thp.size, step):
            t = thp[s:s + step, None]
            x1, x2, P = eval_phase(xg[None, :], t)
            ph = w * (ug[None, :] - xg[None, :] * (yc[0] * np.cos(t) + yc[1] * np.sin(t)) - P)
            d = np.abs(np.diff(ph, axis=1))
            ok = np.isfinite(d)
            if weight is not None:
                m = weight(x1, x2) != 0
                ok &= m[:, 1:] & m[:, :-1]
            dV = np.maximum(dV, np.max(np.where(ok, d, 0.0), axis=0))
    V = np.concatenate([[0.0], np.cumsum(dV)])
    cuts = np.interp(np.arange(0, V[-1], 2 * np.pi), V, ug)
    near0 = ub * np.logspace(-12, -1, 23)
    rb = rmax * 2.0 ** -np.arange(12, 0, -1)                       # band boundaries (ascending)
    extra_u = np.asarray(extra_r, dtype=float) ** 2 / 2
    edges = np.concatenate([[0.0, umin, ub], near0, cuts, np.linspace(0, ub, 5), rb**2 / 2, extra_u])
    edges = np.unique(edges[(edges >= 0) & (edges <= ub)])
    GX, GW = _gl(ngl)
    mid, half = (edges[1:] + edges[:-1]) / 2, (edges[1:] - edges[:-1]) / 2
    uu = (mid[:, None] + half[:, None] * GX[None, :]).ravel()
    wu = (half[:, None] * GW[None, :]).ravel()
    xu = np.sqrt(2 * uu)
    if radial_weight is not None:
        wu = wu * radial_weight(xu)
        keep = wu != 0
        uu, wu, xu = uu[keep], wu[keep], xu[keep]
    band = np.searchsorted(rb, xu)                                  # 0 (innermost) .. 12 (outer)
    r_lo_all = np.concatenate([[0.0], rb])
    r_hi_all = np.concatenate([rb, [rmax]])

    # ---------- angular bandwidth per band
    th_b = np.linspace(0, 2 * np.pi, 257)[:-1]

    def max_dpsi_dth(r_lo, r_hi):
        rs = np.linspace(max(r_lo, 1e-6 * rmax), r_hi, 8)
        R, T = np.meshgrid(rs, th_b)
        x1, x2 = c[0] + R * np.cos(T), c[1] + R * np.sin(T)
        with np.errstate(all="ignore"):
            g1, g2 = lens.grad(x1, x2)
            d = np.abs(R * (-np.sin(T) * g1 + np.cos(T) * g2))
            ok = np.isfinite(d)
            if weight is not None:
                ok &= weight(x1, x2) != 0
        return float(np.max(np.where(ok, d, 0.0)))

    def n_angles(r_lo, r_hi, r_eff=None):
        D = max_dpsi_dth(r_lo, r_hi)
        n = 1.3 * w * ((r_eff if r_eff is not None else r_hi) * yabs + D) + 32
        if ang_req is not None:
            n += ang_req(r_lo, r_hi)
        n *= theta_factor
        return int(2 * np.ceil(n / 2)), D

    total = 0.0 + 0.0j
    n_eval = 0
    per_band = []
    for b in np.unique(band):
        sel = band == b
        ub_nodes, wb_nodes, xb_nodes = uu[sel], wu[sel], xu[sel]
        N, _ = n_angles(r_lo_all[b], r_hi_all[b])
        th = 2 * np.pi * np.arange(N) / N
        chunk = max(1, int(max_chunk // ub_nodes.size))
        acc = 0.0 + 0.0j
        with np.errstate(all="ignore"):
            for s in range(0, N, chunk):
                t = th[s:s + chunk, None]
                x1, x2, P = eval_phase(xb_nodes[None, :], t)
                ph = w * (ub_nodes[None, :] - xb_nodes[None, :] * (yc[0] * np.cos(t) + yc[1] * np.sin(t)) - P)
                val = wb_nodes[None, :] * np.exp(1j * ph)
                if weight is not None:
                    W = weight(x1, x2)
                    val = np.where(W != 0, W * val, 0.0)
                acc += np.sum(val)
        total += acc * (2 * np.pi / N)
        n_eval += N * ub_nodes.size
        per_band.append((float(r_lo_all[b]), float(r_hi_all[b]), int(ub_nodes.size), N))

    # ---------- rotated tail
    n_tail = 0
    if tail:
        LX, LW = _lag(nlag)
        # decay rate per direction a(th) = w [1 - (n.yc + d psi/dr)/r_b]; angles from the bandwidth
        th0 = np.linspace(0, 2 * np.pi, 257)[:-1]
        g1, g2 = lens.grad(c[0] + rmax * np.cos(th0), c[1] + rmax * np.sin(th0))
        a0 = w * (1 - (np.cos(th0) * (yc[0] + g1) + np.sin(th0) * (yc[1] + g2)) / rmax)
        r_far = abs(np.sqrt(2 * (ub + 1j * 40 / np.min(a0))))
        n_tail, _ = n_angles(rmax / 2, rmax, r_eff=r_far)
        th = 2 * np.pi * np.arange(n_tail) / n_tail
        g1, g2 = lens.grad(c[0] + rmax * np.cos(th), c[1] + rmax * np.sin(th))
        a = w * (1 - (np.cos(th) * (yc[0] + g1) + np.sin(th) * (yc[1] + g2)) / rmax)
        if np.any(a <= 0):
            raise RuntimeError("tail does not decay in some direction; increase xb_min.")
        chunk = max(1, int(max_chunk // nlag))
        acc = 0.0 + 0.0j
        for s in range(0, n_tail, chunk):
            t, at = th[s:s + chunk, None], a[s:s + chunk, None]
            ut = ub + 1j * LX[None, :] / at
            xt = np.sqrt(2 * ut)
            x1, x2 = c[0] + xt * np.cos(t), c[1] + xt * np.sin(t)
            g = (np.exp(-1j * w * (xt * (yc[0] * np.cos(t) + yc[1] * np.sin(t)) + lens.psi(x1, x2)))
                 * np.exp(-(w / at - 1) * LX[None, :]))
            acc += np.sum(1j * np.exp(1j * w * ub) / at[:, 0] * np.sum(LW[None, :] * g, axis=1))
        total += acc * (2 * np.pi / n_tail)
        n_eval += n_tail * nlag

    if info is not None:
        info.update(n_eval=n_eval, bands=per_band, n_theta_tail=n_tail, n_radial=int(uu.size))
    return total


# ================================================================== main routine
def amplification(w, y, lens, center=None, patch_frac=0.8, patch_max=1.0, s0=0.3,
                  ngl=20, nlag=40, theta_factor=1.0, xb_min=0.0, xb_base=2.0, n_transition=6, pou_density=16.0,
                  estimate_error=False, return_info=False, max_chunk=3_000_000):
    """Amplification factor F(w, y) for a Lens.

    center          polar centre of the background integral (default: the singular point
                    nearest the source; that point then needs no patch)
    patch_frac      patch radius = min(patch_max, patch_frac * distance to the nearest other
                    singular point); must be < 1 (patches may overlap, but none may reach
                    another singular point)
    s0              the bump chi_k equals 1 for |x - x_k| < s0 rho_k
    ngl, nlag       Gauss-Legendre nodes per radial piece / Gauss-Laguerre nodes on the tail
    theta_factor    multiplies every angular node count (convergence test: 2)
    xb_min          lower bound on the split radius (convergence test: move it out)
    xb_base         smallest split radius tried before the criterion (n.y + dpsi/dr)/r <= 1/2
    n_transition    radial pieces across each bump's transition zone
    pou_density     angles that fall across one bump transition, seen from the centre
                    (16 resolves the bump to ~1e-13)
    estimate_error  also return max(|F - F(theta_factor x2)|, |F - F(split x1.5)|)
    return_info     also return a dict describing the grids
    """
    w = float(w)
    y = np.asarray(y, dtype=float)
    if not lens._checked:
        lens.check_analytic()
    S = np.array(lens.singular, dtype=float).reshape(-1, 2)

    # ---- centre of the background integral
    if center is None:
        c = S[np.argmin(np.hypot(*(S - y).T))] if len(S) else np.zeros(2)
    else:
        c = np.asarray(center, dtype=float)
    yc = y - c

    # ---- patches (every singular point except one sitting at the centre)
    patches = []
    for k, p in enumerate(S):
        if np.hypot(*(p - c)) < 1e-12:
            continue
        others = [np.hypot(*(p - q)) for j, q in enumerate(S) if j != k]
        rho = min(patch_max, patch_frac * min(others)) if others else patch_max
        patches.append((p, rho))

    def chi0(x1, x2):
        out = np.ones(np.broadcast(x1, x2).shape)
        for p, rho in patches:
            out = out - _step(np.sqrt((x1 - p[0])**2 + (x2 - p[1])**2) / rho, s0)
        return out

    # ---- split radius of the background: (n.yc + d psi/dr)/r <= thr in every direction
    q = lens.q_inf if lens.q_inf is not None else _estimate_q_inf(lens, c)
    if q >= 1:
        raise ValueError(f"kappa + |gamma| = {q:.3f} >= 1: the tail cannot be rotated upward in every direction.")
    thr = max(0.5, 0.5 * (1 + q))
    th = np.linspace(0, 2 * np.pi, 257)[:-1]
    reach = max([np.hypot(*(p - c)) + rho for p, rho in patches], default=0.0)
    xb = max(float(xb_base), 2 * np.hypot(*yc), 1.25 * reach, float(xb_min))
    for _ in range(400):
        g1, g2 = lens.grad(c[0] + xb * np.cos(th), c[1] + xb * np.sin(th))
        if np.max((np.cos(th) * (yc[0] + g1) + np.sin(th) * (yc[1] + g2)) / xb) <= thr:
            break
        xb *= 1.1
    else:
        raise RuntimeError("could not find a split radius; is the potential growing faster than r^2/2?")
    ub = max(xb**2 / 2, 20 / w)

    # ---- background: chi_0 e^{iwT} about c, with extra radial pieces / angles across each bump
    extra_r, dists = [], []
    for p, rho in patches:
        d = np.hypot(*(p - c))
        n = int(np.ceil(2 * rho / ((1 - s0) * rho / n_transition)))
        extra_r.append(np.linspace(max(0.0, d - rho), d + rho, n + 1))
        dists.append((d, rho))

    def ang_req(r_lo, r_hi):
        n = 0.0
        for d, rho in dists:
            if d - rho < r_hi and d + rho > r_lo:
                n = max(n, pou_density * 2 * np.pi * min(r_hi, d + rho) / ((1 - s0) * rho))
        return n

    info_bg = {}
    bg = _polar_integral(w, lens, c, yc, ub, ngl, nlag, True, theta_factor, max_chunk,
                         weight=chi0 if patches else None,
                         extra_r=np.concatenate(extra_r) if extra_r else (),
                         ang_req=ang_req if patches else None, info=info_bg)
    total = np.exp(0.5j * w * (yc @ yc)) * bg

    # ---- patches: chi_k e^{iwT} about x_k (finite disc, no tail)
    info_p = []
    for p, rho in patches:
        yk = y - p
        ip = {}
        val = _polar_integral(w, lens, p, yk, rho**2 / 2, ngl, nlag, False, theta_factor, max_chunk,
                              radial_weight=lambda r, rho=rho: _step(r / rho, s0),
                              extra_r=np.linspace(s0 * rho, rho, n_transition + 1), info=ip)
        total += np.exp(0.5j * w * (yk @ yk)) * val
        info_p.append(dict(center=tuple(p), rho=rho, **ip))

    F = w / (2j * np.pi) * total

    out = [F]
    if estimate_error:
        kw = dict(center=c, patch_frac=patch_frac, patch_max=patch_max, s0=s0, ngl=ngl, nlag=nlag,
                  n_transition=n_transition, pou_density=pou_density, max_chunk=max_chunk)
        F2 = amplification(w, y, lens, theta_factor=2 * theta_factor, xb_min=xb_min, **kw)
        F3 = amplification(w, y, lens, theta_factor=theta_factor, xb_min=1.5 * np.sqrt(2 * ub), **kw)
        out.append(max(abs(F2 - F), abs(F3 - F)))
    if return_info:
        out.append(dict(center=tuple(c), x_b=float(np.sqrt(2 * ub)), q_inf=q, background=info_bg,
                        patches=info_p,
                        n_eval=info_bg["n_eval"] + sum(i["n_eval"] for i in info_p)))
    return out[0] if len(out) == 1 else tuple(out)


# ================================================================== example
if __name__ == "__main__":
    # equal-mass binary: two point masses m = 0.5 at (-0.5, 0) and (+0.5, 0)
    lens = point_mass(0.5, (-0.5, 0.0)) + point_mass(0.5, (0.5, 0.0))
    y = (0.1, 0.2)
    for w in (1.0, 10.0, 100.0):
        F, err = amplification(w, y, lens, estimate_error=True)
        print(f"w = {w:6.1f}   F = {F.real:+.10f} {F.imag:+.10f}i   |F| = {abs(F):.10f}   (self-check {err:.1e})")