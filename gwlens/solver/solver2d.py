"""
Amplification factor for an ARBITRARY lens potential psi(x1, x2) with the rotated-tail method.
Version 2: fast frequency sweeps.

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
    Fs = amplification_many(ws, y, lens)            # many frequencies: the fast path
    F  = amplification(w, y, lens)                  # one frequency
    solver = Solver2D(lens, y, wmin, wmax); solver(w); solver.many(ws)   # keep the solver

Ready-made components (they add with +):  point_mass, sis, cis, nfw, external_shear.

Method (unchanged from version 1)
------
Partition of unity 1 = chi_0(x) + sum_k chi_k(x), with chi_k a C-infinity bump (Planck taper)
that equals 1 within s0*rho_k of singular point x_k and 0 beyond rho_k (rho_k < distance to any
other singular point). Patch k: Int chi_k e^{iwT} on the disc about x_k in polar coordinates
centred on x_k (no tail). Background: Int chi_0 e^{iwT} in polar coordinates about a centre c
(default: the singular point nearest the source); beyond r_b the radial tail is rotated to
u_b + i t. Radial: Gauss-Legendre on pieces cut by the accumulated phase. Angular: trapezoid
rule, number of angles per radial octave from the local angular bandwidth.

What makes version 2 fast
-------------------------
 1. Nothing in the head depends on w except the factor e^{i w phi}: the phase of every node,
        phi_j = u_j - r_j n_j.(y - c) - psi(x_j) + |y - c|^2/2,
    and its weight W_j (Gauss-Legendre x trapezoid x partition-of-unity) are computed ONCE,
    on a grid built for the largest frequency (it then resolves every smaller one), so
        head(w) = sum_j W_j exp(i w phi_j).
    For many frequencies this is a type-3 non-uniform FFT (FINUFFT, if installed): the whole
    sweep costs about as much as one evaluation at w_max.
 2. Pieces hold up to 8 cycles of the phase with 32 Gauss-Legendre nodes (4 per cycle
    instead of 20), at the same ~1e-13 accuracy.
 3. The tail, per direction, is i e^{i w Phi_0} S(1/w) / (w alpha), where S is a smooth
    function of 1/w; S is tabulated at Chebyshev points in 1/w once and interpolated, so
    psi is not re-evaluated per frequency.
 4. Grid probing, bandwidths and partition-of-unity weights are computed once per (lens, y).
"""
import warnings
import numpy as np
from scipy.special import roots_legendre, roots_laguerre, expit
try:
    import finufft                      # optional: fast many-frequency sums
except ImportError:
    finufft = None

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
class Potential:
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
        new = Potential.__new__(Potential)
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
    return Potential(lambda x1, x2: 0.5 * m * np.log((x1 - a)**2 + (x2 - b)**2), singular=[(a, b)], q_inf=0.0)


def sis(psi0=1.0, x0=(0.0, 0.0)):
    """psi = psi0 |x - x0|."""
    a, b, p = float(x0[0]), float(x0[1]), float(psi0)
    return Potential(lambda x1, x2: p * np.sqrt((x1 - a)**2 + (x2 - b)**2), singular=[(a, b)], q_inf=0.0)


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
    return Potential(f, singular=[(a, b)], q_inf=0.0)


def nfw(k=1.0, psi0=1.0, x0=(0.0, 0.0)):
    """NFW: 0.5 psi0 [ln^2(v/2) + arccos^2(1/v)], v = k |x - x0|  (k = xi0 / r_s)."""
    a, b, p, k = float(x0[0]), float(x0[1]), float(psi0), float(k)

    def f(x1, x2):
        v = k * np.sqrt((x1 - a)**2 + (x2 - b)**2 + 0j)       # complex: arccos(1/v) for v < 1
        return 0.5 * p * (np.log(v / 2)**2 + np.arccos(1 / v)**2)
    return Potential(f, singular=[(a, b)], q_inf=0.0)


def external_shear(gamma1=0.0, gamma2=0.0, kappa=0.0):
    """psi = kappa |x|^2/2 + [gamma1 (x1^2 - x2^2) + 2 gamma2 x1 x2]/2."""
    g1, g2, k = float(gamma1), float(gamma2), float(kappa)
    f = lambda x1, x2: 0.5 * (k * (x1 * x1 + x2 * x2) + g1 * (x1 * x1 - x2 * x2) + 2 * g2 * x1 * x2)
    return Potential(f, singular=[], q_inf=k + np.hypot(g1, g2))


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



# ================================================================== grid building blocks
def _probe_V1(lens, c, yc, u_floor, u_lin, u_hi, wmax, weight=None, n_ang=64):
    """Worst-direction accumulated phase per unit w, V1(u), on a probe grid in [u_floor, u_hi].
    Dense (>= 3 points per cycle at wmax) up to u_lin, geometric everywhere."""
    n_lin = int(max(1500, 3 * wmax * u_lin / (2 * np.pi)))
    ug = np.unique(np.concatenate([np.linspace(u_floor, u_lin, n_lin),
                                   np.geomspace(u_floor, u_hi, 3000)]))
    xg = np.sqrt(2 * ug)
    th = (np.arange(n_ang) + 0.5) * 2 * np.pi / n_ang
    dV = np.zeros(ug.size - 1)
    step = max(1, int(1_000_000 // ug.size))
    with np.errstate(all="ignore"):
        for s in range(0, n_ang, step):
            t = th[s:s + step, None]
            x1, x2 = c[0] + xg[None, :] * np.cos(t), c[1] + xg[None, :] * np.sin(t)
            ph = ug[None, :] - xg[None, :] * (yc[0] * np.cos(t) + yc[1] * np.sin(t)) - np.real(lens.psi(x1, x2))
            d = np.abs(np.diff(ph, axis=1))
            ok = np.isfinite(d)
            if weight is not None:
                m = weight(x1, x2) != 0
                ok &= m[:, 1:] & m[:, :-1]
            dV = np.maximum(dV, np.max(np.where(ok, d, 0.0), axis=0))
    return ug, np.concatenate([[0.0], np.cumsum(dV)])


def _max_dpsi_dth(lens, c, r_lo, r_hi, weight=None, n_ang=256, n_r=8):
    """max over the annulus of |d psi / d theta| (angular derivative about c), masked by weight != 0."""
    rs = np.linspace(max(r_lo, 1e-6 * r_hi), r_hi, n_r)
    R, T = np.meshgrid(rs, (np.arange(n_ang) + 0.5) * 2 * np.pi / n_ang)
    x1, x2 = c[0] + R * np.cos(T), c[1] + R * np.sin(T)
    with np.errstate(all="ignore"):
        g1, g2 = lens.grad(x1, x2)
        d = np.abs(R * (-np.sin(T) * g1 + np.cos(T) * g2))
        ok = np.isfinite(d)
        if weight is not None:
            ok &= weight(x1, x2) != 0
    return float(np.max(np.where(ok, d, 0.0)))


def _even(n):
    return int(2 * np.ceil(n / 2))


# ================================================================== the solver
class Solver2D:
    """Everything that does not depend on w, for one (lens, y); evaluate for any w in [wmin, wmax].

    Parameters (defaults are the validated ones)
      center        polar centre of the background (default: singular point nearest y)
      patch_frac    patch radius = min(patch_max, patch_frac * distance to nearest other singular point)
      s0            chi_k = 1 for |x - x_k| < s0 rho_k
      cycles, ngl   phase cycles per radial piece (at wmax) and Gauss-Legendre nodes per piece
      nlag          Gauss-Laguerre nodes on the rotated tail
      n_cheb        Chebyshev points in 1/w for the tail table (None: automatic)
      theta_factor  multiplies every angular node count (convergence test: 2)
      bw_factor, bw_margin   angles = bw_factor * bandwidth + bw_margin
      xb_min        lower bound on the split radius (convergence test: move it out)
      tail_mode     'auto' | 'chebyshev' | 'direct'
      nufft_eps     FINUFFT tolerance for many-frequency sums
    """

    def __init__(self, lens, y, wmin, wmax, center=None, patch_frac=0.8, patch_max=1.0, s0=0.3,
                 cycles=8, ngl=32, nlag=30, n_cheb=None, theta_factor=1.0, bw_factor=1.3, bw_margin=32,
                 xb_min=0.0, xb_base=2.0, n_transition=3, pou_density=16.0, tail_mode="auto",
                 nufft_eps=1e-14, max_chunk=2_000_000):
        if not lens._checked:
            lens.check_analytic()
        self.lens = lens
        self.y = y = np.asarray(y, dtype=float)
        self.wmin, self.wmax = float(wmin), float(wmax)
        if not (0 < self.wmin <= self.wmax):
            raise ValueError("need 0 < wmin <= wmax")
        self.cycles, self.ngl, self.nlag = cycles, ngl, nlag
        self.theta_factor, self.bw_factor, self.bw_margin = theta_factor, bw_factor, bw_margin
        self.nufft_eps, self.max_chunk = nufft_eps, max_chunk
        self.u_floor = 1e-15 / max(self.wmax, 1.0)
        S = np.array(lens.singular, dtype=float).reshape(-1, 2)

        # ---- centre and patches
        if center is None:
            c = S[np.argmin(np.hypot(*(S - y).T))] if len(S) else np.zeros(2)
        else:
            c = np.asarray(center, dtype=float)
        self.c, self.yc = c, y - c
        patches = []
        for k, p in enumerate(S):
            if np.hypot(*(p - c)) < 1e-12:
                continue
            others = [np.hypot(*(p - q)) for j, q in enumerate(S) if j != k]
            rho = min(patch_max, patch_frac * min(others)) if others else patch_max
            patches.append((p, rho))
        self.patches = patches

        def chi0(x1, x2):
            out = np.ones(np.broadcast(x1, x2).shape)
            for p, rho in patches:
                out = out - _step(np.sqrt((x1 - p[0])**2 + (x2 - p[1])**2) / rho, s0)
            return out
        self._chi0 = chi0 if patches else None

        # ---- split radius: (n.yc + d psi/dr)/r <= thr in every direction
        q = lens.q_inf if lens.q_inf is not None else _estimate_q_inf(lens, c)
        if q >= 1:
            raise ValueError(f"kappa + |gamma| = {q:.3f} >= 1: the tail cannot be rotated upward in every direction.")
        thr = max(0.5, 0.5 * (1 + q))
        th = np.linspace(0, 2 * np.pi, 257)[:-1]
        reach = max([np.hypot(*(p - c)) + rho for p, rho in patches], default=0.0)
        xb = max(float(xb_base), 2 * np.hypot(*self.yc), 1.25 * reach, float(xb_min))
        for _ in range(400):
            g1, g2 = lens.grad(c[0] + xb * np.cos(th), c[1] + xb * np.sin(th))
            if np.max((np.cos(th) * (self.yc[0] + g1) + np.sin(th) * (self.yc[1] + g2)) / xb) <= thr:
                break
            xb *= 1.1
        else:
            raise RuntimeError("could not find a split radius; is the potential growing faster than r^2/2?")
        self.q_inf, self.xb0, self.ub0 = q, xb, xb * xb / 2
        self.w_switch = 20 / self.ub0                      # below this, u_b = 20/w > ub0
        self.ub_max = max(self.ub0, 20 / self.wmin)
        self.off_bg = 0.5 * (self.yc @ self.yc)

        # ---- background probe (also covers [ub0, ub_max] for small w)
        self.ug_bg, self.V1_bg = _probe_V1(lens, c, self.yc, self.u_floor, self.ub0, self.ub_max,
                                           self.wmax, weight=self._chi0)
        xbmax = np.sqrt(2 * self.ub_max)
        self.D_out = _max_dpsi_dth(lens, c, self.xb0 / 2, max(xbmax, self.xb0))

        # ---- head nodes, background
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

        def touches(r_lo, r_hi):
            return any(d - rho < r_hi and d + rho > r_lo for d, rho in dists)

        phis, Ws, self.info = [], [], dict(center=tuple(c), x_b=self.xb0, q_inf=q, regions=[])
        ph, W, inf = self._head_nodes(c, self.yc, self.ub0, self.ug_bg, self.V1_bg, self.off_bg,
                                      weight=self._chi0, touches=touches,
                                      extra_r=np.concatenate(extra_r) if extra_r else (),
                                      ang_req=ang_req if patches else None)
        phis.append(ph); Ws.append(W); self.info["regions"].append(dict(kind="background", **inf))

        # ---- head nodes, patches
        for p, rho in patches:
            yk = y - p
            ub = rho * rho / 2
            ug, V1 = _probe_V1(lens, p, yk, self.u_floor, ub, ub, self.wmax)
            ph, W, inf = self._head_nodes(p, yk, ub, ug, V1, 0.5 * (yk @ yk),
                                          radial_weight=lambda r, rho=rho: _step(r / rho, s0),
                                          extra_r=np.linspace(s0 * rho, rho, n_transition + 1))
            phis.append(ph); Ws.append(W)
            self.info["regions"].append(dict(kind="patch", center=tuple(p), rho=rho, **inf))
        self.phi = np.concatenate(phis)
        self.W = np.concatenate(Ws)
        self.info["n_head_nodes"] = int(self.phi.size)

        # ---- tail
        self.D_tail = max(self.info["regions"][0]["D_outer"], self.D_out)
        LX, LW = _lag(nlag)
        self.LX, self.LW = LX, LW
        w_lo = max(self.wmin, self.w_switch)
        if tail_mode == "auto":
            tail_mode = "chebyshev" if self.wmax > 1.5 * w_lo else "direct"
        self.tail_mode = tail_mode
        if tail_mode == "chebyshev" and self.wmax > w_lo:
            self._build_tail_table(w_lo, n_cheb)
        else:
            self.tail_mode = "direct"
        self.info["tail_mode"] = self.tail_mode

    # ------------------------------------------------------------ head nodes of one region
    def _head_nodes(self, c, yc, ub, ug, V1, offset, weight=None, touches=None,
                    radial_weight=None, extra_r=(), ang_req=None):
        lens, wmax, K = self.lens, self.wmax, self.cycles
        yabs = float(np.hypot(*yc))
        rmax = np.sqrt(2 * ub)
        V1_ub = np.interp(ub, ug, V1)
        cuts = np.interp(np.arange(0, V1_ub, 2 * np.pi * K / wmax), V1, ug)
        geo = np.geomspace(self.u_floor, ub, int(np.ceil(np.log(ub / self.u_floor))) + 1)
        rb = rmax * 2.0 ** -np.arange(12, 0, -1)                       # radial octaves
        extra_u = np.asarray(extra_r, dtype=float) ** 2 / 2
        edges = np.concatenate([[0.0, ub], geo, cuts, np.linspace(0, ub, 5), rb**2 / 2, extra_u])
        edges = np.unique(edges[(edges >= 0) & (edges <= ub)])
        GX, GW = _gl(self.ngl)
        mid, half = (edges[1:] + edges[:-1]) / 2, (edges[1:] - edges[:-1]) / 2
        uu = (mid[:, None] + half[:, None] * GX[None, :]).ravel()
        wu = (half[:, None] * GW[None, :]).ravel()
        xu = np.sqrt(2 * uu)
        if radial_weight is not None:
            wu = wu * radial_weight(xu)
            keep = wu != 0
            uu, wu, xu = uu[keep], wu[keep], xu[keep]
        band = np.searchsorted(rb, xu)
        r_lo_all, r_hi_all = np.concatenate([[0.0], rb]), np.concatenate([rb, [rmax]])

        phis, Ws, n_eval, bands, D_outer = [], [], 0, [], 0.0
        for b in np.unique(band):
            sel = band == b
            ub_n, wb_n, xb_n = uu[sel], wu[sel], xu[sel]
            r_lo, r_hi = r_lo_all[b], r_hi_all[b]
            D = _max_dpsi_dth(lens, c, r_lo, r_hi, weight)
            if b == band.max():
                D_outer = D
            n = self.bw_factor * wmax * (r_hi * yabs + D) + self.bw_margin
            if ang_req is not None:
                n += ang_req(r_lo, r_hi)
            N = _even(n * self.theta_factor)
            th = 2 * np.pi * np.arange(N) / N
            use_w = weight is not None and (touches is None or touches(r_lo, r_hi))
            chunk = max(1, int(self.max_chunk // ub_n.size))
            with np.errstate(all="ignore"):
                for s in range(0, N, chunk):
                    t = th[s:s + chunk, None]
                    ct, st = np.cos(t), np.sin(t)
                    x1, x2 = c[0] + xb_n[None, :] * ct, c[1] + xb_n[None, :] * st
                    ph = ub_n[None, :] - xb_n[None, :] * (yc[0] * ct + yc[1] * st) - np.real(lens.psi(x1, x2)) + offset
                    Wt = np.broadcast_to(wb_n[None, :] * (2 * np.pi / N), ph.shape)
                    if use_w:
                        chi = weight(x1, x2)
                        m = chi != 0
                        phis.append(ph[m]); Ws.append((Wt * chi)[m])
                    else:
                        phis.append(ph.ravel()); Ws.append(Wt.ravel().copy())
            n_eval += N * ub_n.size
            bands.append((float(r_lo), float(r_hi), int(ub_n.size), N))
        return (np.concatenate(phis), np.concatenate(Ws),
                dict(n_eval=n_eval, bands=bands, n_radial=int(uu.size), D_outer=D_outer))

    # ------------------------------------------------------------ tail pieces
    def _tail_angles(self, w, xb):
        """Directions and decay rates for the rotated tail at radius xb for frequency w."""
        c, yc = self.c, self.yc
        th0 = np.linspace(0, 2 * np.pi, 257)[:-1]
        g1, g2 = self.lens.grad(c[0] + xb * np.cos(th0), c[1] + xb * np.sin(th0))
        al0 = 1 - (np.cos(th0) * (yc[0] + g1) + np.sin(th0) * (yc[1] + g2)) / xb
        r_far = abs(np.sqrt(2 * (xb * xb / 2 + 1j * 40 / (w * np.min(al0)))))
        n = self.bw_factor * w * (r_far * np.hypot(*yc) + self.D_tail) + self.bw_margin
        N = _even(n * self.theta_factor)
        th = 2 * np.pi * np.arange(N) / N
        g1, g2 = self.lens.grad(c[0] + xb * np.cos(th), c[1] + xb * np.sin(th))
        al = 1 - (np.cos(th) * (yc[0] + g1) + np.sin(th) * (yc[1] + g2)) / xb
        if np.any(al <= 0):
            raise RuntimeError("tail does not decay in some direction; increase xb_min.")
        return th, al

    def _phase_on_tail(self, th, t, ub):
        """Phi(t) = u - r n.yc - psi(c + r n) + |yc|^2/2 at u = ub + i t (complex)."""
        c, yc = self.c, self.yc
        r = np.sqrt(2 * (ub + 1j * t))
        ct, st = np.cos(th), np.sin(th)
        return ub + 1j * t - r * (yc[0] * ct + yc[1] * st) - self.lens.psi(c[0] + r * ct, c[1] + r * st) + self.off_bg

    def _tail_direct(self, w, ub):
        """sum over directions (2 pi/N) * i e^{iw Phi_0} Int_0^inf e^{iw[Phi(t) - Phi_0]} dt, Gauss-Laguerre."""
        xb = np.sqrt(2 * ub)
        th, al = self._tail_angles(w, xb)
        a = w * al
        Phi0 = self._phase_on_tail(th, 0.0, ub).real
        chunk = max(1, int(self.max_chunk // self.nlag))
        acc = 0.0 + 0.0j
        for s in range(0, th.size, chunk):
            t = self.LX[None, :] / a[s:s + chunk, None]
            Phi = self._phase_on_tail(th[s:s + chunk, None], t, ub)
            S = np.sum(self.LW[None, :] * np.exp(1j * w * (Phi - Phi0[s:s + chunk, None]) + self.LX[None, :]), axis=1)
            acc += np.sum(1j * np.exp(1j * w * Phi0[s:s + chunk]) / a[s:s + chunk] * S)
        return acc * (2 * np.pi / th.size)

    def _segment_direct(self, w, ub):
        """Background head on [ub0, ub] (beyond every patch, chi_0 = 1), for small w."""
        lens, c, yc = self.lens, self.c, self.yc
        ug, V1 = self.ug_bg, self.V1_bg
        lo, hi = np.interp([self.ub0, ub], ug, V1)
        cuts = np.interp(np.arange(lo, hi, 2 * np.pi * self.cycles / w), V1, ug)
        geo = np.geomspace(self.ub0, ub, int(np.ceil(np.log(ub / self.ub0))) + 2)   # r = sqrt(2u) varies smoothly per piece
        edges = np.unique(np.concatenate([cuts, geo, np.linspace(self.ub0, ub, 5)]))
        edges = edges[(edges >= self.ub0) & (edges <= ub)]
        GX, GW = _gl(self.ngl)
        mid, half = (edges[1:] + edges[:-1]) / 2, (edges[1:] - edges[:-1]) / 2
        uu = (mid[:, None] + half[:, None] * GX[None, :]).ravel()
        wu = (half[:, None] * GW[None, :]).ravel()
        xu = np.sqrt(2 * uu)
        N = _even((self.bw_factor * w * (np.sqrt(2 * ub) * np.hypot(*yc) + self.D_out) + self.bw_margin)
                  * self.theta_factor)
        th = 2 * np.pi * np.arange(N) / N
        chunk = max(1, int(self.max_chunk // uu.size))
        acc = 0.0 + 0.0j
        for s in range(0, N, chunk):
            t = th[s:s + chunk, None]
            ct, st = np.cos(t), np.sin(t)
            ph = uu[None, :] - xu[None, :] * (yc[0] * ct + yc[1] * st) \
                - np.real(lens.psi(c[0] + xu[None, :] * ct, c[1] + xu[None, :] * st)) + self.off_bg
            acc += np.sum(wu[None, :] * np.exp(1j * w * ph))
        return acc * (2 * np.pi / N)

    def _build_tail_table(self, w_lo, n_cheb):
        """Tabulate S(theta, 1/w) at Chebyshev points in 1/w on [1/wmax, 1/w_lo] (u_b = ub0 fixed)."""
        th, al = self._tail_angles(self.wmax, self.xb0)
        self.t_th, self.t_al = th, al
        self.t_Phi0 = self._phase_on_tail(th, 0.0, self.ub0).real
        self.eps_lo, self.eps_hi = 1 / self.wmax, 1 / w_lo
        if n_cheb is None:
            n_cheb = int(np.clip(14 + np.sqrt(self.wmax / w_lo), 20, 32))      # 20 already reaches rounding level
        k = np.arange(n_cheb)
        xk = np.cos(np.pi * (k + 0.5) / n_cheb)                        # Chebyshev points, first kind
        eps = 0.5 * (self.eps_hi + self.eps_lo) + 0.5 * (self.eps_hi - self.eps_lo) * xk
        vals = np.empty((th.size, n_cheb), dtype=complex)
        chunk = max(1, int(self.max_chunk // self.nlag))
        for j, e in enumerate(eps):
            w = 1 / e
            for s in range(0, th.size, chunk):
                t = self.LX[None, :] / (w * al[s:s + chunk, None])
                Phi = self._phase_on_tail(th[s:s + chunk, None], t, self.ub0)
                vals[s:s + chunk, j] = np.sum(
                    self.LW[None, :] * np.exp(1j * w * (Phi - self.t_Phi0[s:s + chunk, None]) + self.LX[None, :]), axis=1)
        # Chebyshev coefficients by the discrete cosine transform
        Tm = np.cos(np.outer(np.arange(n_cheb), np.pi * (k + 0.5) / n_cheb))   # (m, k)
        coef = (2.0 / n_cheb) * vals @ Tm.T
        coef[:, 0] *= 0.5
        self.t_coef = coef
        self.info.update(n_theta_tail=int(th.size), n_cheb=n_cheb)
        # self-check: compare the interpolant with a direct evaluation at the hardest frequency
        # in the middle of the range (largest Chebyshev gap is near eps_hi)
        w_chk = 1 / (0.5 * (self.eps_hi + self.eps_lo) + 0.5 * (self.eps_hi - self.eps_lo) * np.cos(np.pi / (2 * n_cheb)) * 0.999)
        self.info["tail_table_check"] = float(abs(self._tail_cheb(np.array([w_chk]))[0] - self._tail_fixed_direct(w_chk)))

    def _tail_fixed_direct(self, w):
        """Direct tail with the table's directions (for the self-check)."""
        th, al, Phi0 = self.t_th, self.t_al, self.t_Phi0
        t = self.LX[None, :] / (w * al[:, None])
        Phi = self._phase_on_tail(th[:, None], t, self.ub0)
        S = np.sum(self.LW[None, :] * np.exp(1j * w * (Phi - Phi0[:, None]) + self.LX[None, :]), axis=1)
        return np.sum(1j * np.exp(1j * w * Phi0) / (w * al) * S) * (2 * np.pi / th.size)

    def _tail_cheb(self, ws):
        ws = np.asarray(ws, dtype=float)
        x = (2 / ws - (self.eps_hi + self.eps_lo)) / (self.eps_hi - self.eps_lo)
        n = self.t_coef.shape[1]
        Tm = np.empty((n, ws.size))
        Tm[0] = 1.0
        if n > 1:
            Tm[1] = x
        for m in range(2, n):
            Tm[m] = 2 * x * Tm[m - 1] - Tm[m - 2]
        out = np.empty(ws.size, dtype=complex)
        step = max(1, int(self.max_chunk // self.t_th.size))
        for s in range(0, ws.size, step):
            wv = ws[s:s + step]
            Sv = self.t_coef @ Tm[:, s:s + step]                               # (theta, w)
            E = np.exp(1j * np.outer(self.t_Phi0, wv)) / (self.t_al[:, None] * wv[None, :])
            out[s:s + step] = 1j * np.sum(E * Sv, axis=0) * (2 * np.pi / self.t_th.size)
        return out

    # ------------------------------------------------------------ evaluation
    def _head(self, ws):
        ws = np.asarray(ws, dtype=float)
        if finufft is not None and ws.size >= 4:
            return finufft.nufft1d3(self.phi, self.W.astype(complex), ws, isign=1, eps=self.nufft_eps)
        if ws.size > 20 and not getattr(Solver2D, "_warned_nufft", False):
            Solver2D._warned_nufft = True
            warnings.warn("finufft is not installed: frequency sweeps use direct sums and are ~100x slower. "
                          "Install it with  pip install finufft")
        out = np.empty(ws.size, dtype=complex)
        step = 1_000_000
        for i, w in enumerate(ws):
            acc = 0.0 + 0.0j
            for s in range(0, self.phi.size, step):
                acc += np.sum(self.W[s:s + step] * np.exp(1j * w * self.phi[s:s + step]))
            out[i] = acc
        return out

    def many(self, ws):
        """F at an array of frequencies (each in [wmin, wmax]; w = 0 gives 1)."""
        ws = np.atleast_1d(np.asarray(ws, dtype=float))
        F = np.ones(ws.size, dtype=complex)
        pos = ws > 0
        wp = ws[pos]
        if np.any(wp < self.wmin * (1 - 1e-12)) or np.any(wp > self.wmax * (1 + 1e-12)):
            raise ValueError(f"frequencies must lie in [{self.wmin}, {self.wmax}] for this solver")
        total = self._head(wp)
        low = wp < self.w_switch
        for i in np.nonzero(low)[0]:                       # small w: u_b = 20/w > ub0
            ub = 20 / wp[i]
            total[i] += self._segment_direct(wp[i], ub) + self._tail_direct(wp[i], ub)
        hi = ~low
        if np.any(hi):
            if self.tail_mode == "chebyshev":
                total[hi] += self._tail_cheb(wp[hi])
            else:
                total[hi] += np.array([self._tail_direct(w, self.ub0) for w in wp[hi]])
        F[pos] = wp / (2j * np.pi) * total
        return F

    def __call__(self, w):
        return self.many([w])[0]


# ================================================================== convenience functions
def amplification_many(ws, y, lens, estimate_error=False, **kw):
    """F on an array of frequencies (the fast path: one grid, one non-uniform FFT).
    estimate_error=True also returns max |F - F(angles x2)|, |F - F(split x1.5)| over the array."""
    ws = np.atleast_1d(np.asarray(ws, dtype=float))
    wp = ws[ws > 0]
    if wp.size == 0:
        return np.ones(ws.size, dtype=complex)
    s = Solver2D(lens, y, wp.min(), wp.max(), **kw)
    F = s.many(ws)
    if not estimate_error:
        return F
    kw2 = dict(kw); kw2["theta_factor"] = 2 * kw.get("theta_factor", 1.0)
    kw3 = dict(kw); kw3["xb_min"] = 1.5 * s.xb0
    F2 = Solver2D(lens, y, wp.min(), wp.max(), **kw2).many(ws)
    F3 = Solver2D(lens, y, wp.min(), wp.max(), **kw3).many(ws)
    return F, np.maximum(np.abs(F2 - F), np.abs(F3 - F))


def amplification(w, y, lens, estimate_error=False, return_info=False, **kw):
    """F at one frequency. Options as for Solver2D (center, patch_frac, s0, theta_factor, xb_min, ...)."""
    w = float(w)
    if w == 0:
        return 1.0 + 0.0j
    s = Solver2D(lens, y, w, w, **kw)
    F = s(w)
    out = [F]
    if estimate_error:
        kw2 = dict(kw); kw2["theta_factor"] = 2 * kw.get("theta_factor", 1.0)
        kw3 = dict(kw); kw3["xb_min"] = 1.5 * np.sqrt(2 * max(s.ub0, 20 / w))
        out.append(max(abs(Solver2D(lens, y, w, w, **kw2)(w) - F), abs(Solver2D(lens, y, w, w, **kw3)(w) - F)))
    if return_info:
        out.append(dict(s.info, n_eval=s.info["n_head_nodes"]))
    return out[0] if len(out) == 1 else tuple(out)


