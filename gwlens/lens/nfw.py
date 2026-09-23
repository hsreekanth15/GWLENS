import numpy as np
import time
import multiprocessing as mpr
import h5py
import os
import configparser
import logging
from .base import Lens
from gwlens.lens import solver
from importlib.metadata import version, PackageNotFoundError   
from datetime import datetime, timezone                        
from scipy.optimize import fsolve,brentq
from scipy.special import jv
from scipy.special import factorial as sp_fac
from astropy.constants import c,G,M_sun
from joblib import Parallel, delayed
from mpmath import mp,mpf, mpc, sqrt, exp, besselj, quad,log, atan,atanh
from mpi4py import MPI



# Setup logger (you can customize the path or pass it in)
logger = logging.getLogger("nfw_interpolator")
logger.setLevel(logging.INFO)
if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


mp.dps = 200  # set precision to 50 decimal places (arbitrary precision)
eps = 1e-3  # relative accuracy tolerance


class NFW1(Lens):
    """
    Navarro Frenk White Class, a class inherited from the base class Lens
    """
    def potential(self,k, x):
        u = k * x
        return np.where(
            u <= 1,
            0.5 * (np.log(np.maximum(u, 1e-300) / 2)**2
                - np.arctanh(np.sqrt(np.maximum(1 - u**2, 0)))**2),
            0.5 * (np.log(u / 2)**2
                + np.arctan(np.sqrt(np.maximum(u**2 - 1, 0)))**2)
        )
    def potential1(self,k, x):
        u  = np.maximum(k * x, 1e-300)

        a  = np.log(u / 2.0)                                     # ln(u/2), always negative for u<2
        
        # u <= 1 : factor  a² − b² = (a−b)(a+b)  to avoid catastrophic cancellation
        s     = np.sqrt(np.maximum(1.0 - u**2, 0.0))
        b     = np.arctanh(s)                                    # arctanh(√(1−u²))
        leq1  = 0.5 * (a - b) * (a + b)                         # stable for all u ≤ 1

        # u > 1 : no cancellation issue (both terms are positive and additive)
        t     = np.sqrt(np.maximum(u**2 - 1.0, 0.0))
        gt1   = 0.5 * (a**2 + np.arctan(t)**2)

        return np.where(u <= 1.0, leq1, gt1)
    def grad_potential(self, x1, x2, rs):
        """
        Gradient of the NFW lensing potential (= deflection angle vector).

        For a circularly-symmetric lens:
            ∇ψ = α(r)/r · (x1, x2)

        where the scalar deflection angle is:
            α(r) = (1/r) · [ln(u/2) + h(u)],   u = rs·r

        The returned components are:
            ∂ψ/∂x1 = α(r)/r · x1
            ∂ψ/∂x2 = α(r)/r · x2
        """
        r = np.sqrt(x1**2 + x2**2)
        u = rs * r

        h = self._h(u)                             # atanh / atan branch

        # α(r)/r — avoids a separate division by r
        alpha_over_r = (np.log(u / 2.0) + h) / r**2

        return alpha_over_r * x1, alpha_over_r * x2
    

    def deflection(self, x1, x2, rs):
        """
        NFW deflection angle vector  α = ∇ψ.

        For a circularly-symmetric lens the deflection points radially:

            α₁ = α(r)/r · x1,   α₂ = α(r)/r · x2

        where the scalar deflection magnitude is:

            α(r) = (1/r) · [ln(u/2) + h(u)],   u = rs · r

        This is identical to grad_potential — the deflection angle IS
        the gradient of the lensing potential.

        Parameters
        ----------
        x1, x2 : float  – image-plane coordinates
        rs     : float  – dimensionless NFW scale radius

        Returns
        -------
        alpha1, alpha2 : float  – deflection angle components
        """
        return self.grad_potential(x1, x2, rs)  
    
    def Time_Delay(self,y1, y2,rs, Ml):

        """
        
        Computes the physical timedelay between images

        Returns: timedelay between signals in seconds

        """

        prefactor = 12*G*Ml*M_sun/c**3

        sol = self.solve_images(y1,y2,rs)
        td = []
        for img in sol:
            dimensionlenss_td = img["dimensionless_time_delay"]
            img["timedelay"] = prefactor*dimensionlenss_td
            td.append(img["timedelay"])

        return td
 

    def Fermat_Potential(self,x1,x2,y1,y2,rs):
        x = np.sqrt(x1**2 + x2**2)
        
        return 0.5* ( (x1-y1)**2 + (x2-y2)**2) - self.potential(rs,x)

    def _h(self, u):
        """
        Stable auxiliary function for NFW lensing.

        Handles removable singularity at u = 1.
        """

        if np.isscalar(u):
            if abs(u - 1.0) < 1e-7:
                return 1.0

            if u < 1.0:
                s = np.sqrt(max(1.0 - u*u, 0.0))

                if s < 1e-12:
                    return 1.0  # limit

                return np.arctanh(s) / s

            else:
                t = np.sqrt(max(u*u - 1.0, 0.0))

                if t < 1e-12:
                    return 1.0  # symmetric limit

                return np.arctan(t) / t

        else:
            # vector-safe version
            u = np.asarray(u)

            out = np.ones_like(u)

            mask_low  = u < 1.0
            mask_high = u > 1.0

            # u < 1
            ul = u[mask_low]
            s = np.sqrt(np.maximum(1.0 - ul**2, 0.0))
            safe = s > 1e-12
            out[mask_low] = 1.0
            out[mask_low][safe] = np.arctanh(s[safe]) / s[safe]

            # u > 1
            uh = u[mask_high]
            t = np.sqrt(np.maximum(uh**2 - 1.0, 0.0))
            safe = t > 1e-12
            out[mask_high] = 1.0
            out[mask_high][safe] = np.arctan(t[safe]) / t[safe]

            return out
    def _convergence(self, r, rs):
        """
        NFW convergence (surface-mass density in units of Σ_cr):
            κ(r) = (rs²/2) · [1/(u²-1) + h(u)/(1-u²)]   u < 1
            κ(r) = rs² / 6                                 u = 1  (analytic limit)
            κ(r) = (rs²/2) · [1/(u²-1) - h(u)/(u²-1)]   u > 1

        Derived by applying the 2-D Poisson equation to the NFW
        potential: κ = (1/2r) d(r·α)/dr.
        """
        u = rs * r

        if abs(u - 1.0) < 1e-7:              # analytic limit  u → 1
            return rs**2 / 6.0

        elif u < 1.0:
            # 1/(u²-1) is negative here; atanh term compensates → κ > 0
            return (rs**2 / 2.0) * (
                1.0 / (u**2 - 1.0)
                + np.arctanh(np.sqrt(1.0 - u**2)) / (1.0 - u**2)**1.5
            )
        else:
            return (rs**2 / 2.0) * (
                1.0 / (u**2 - 1.0)
                - np.arctan(np.sqrt(u**2 - 1.0)) / (u**2 - 1.0)**1.5
            )
    def _mean_convergence(self, r, rs):
        """
        Mean convergence within projected radius r:
            κ̄(r) = α(r)/r = (1/r²) · [ln(u/2) + h(u)]

        This equals twice the enclosed projected mass divided by π r²
        (in units of Σ_cr), and is directly the deflection angle α = κ̄·r.
        """
        u = rs * r
        return (1.0 / r**2) * (np.log(u / 2.0) + self._h(u))


    def Hessian(self, x1, x2, rs):
        """
        Second derivatives of the NFW lensing potential ψ(x1, x2).

        For a circularly-symmetric lens the Hessian decomposes as:
            ψ_xx = κ + γ cos2φ
            ψ_yy = κ − γ cos2φ        cos2φ = (x1²−x2²)/r²
            ψ_xy =     γ sin2φ         sin2φ = 2 x1 x2 / r²

        where the shear magnitude is  γ = κ̄(r) − κ(r).

        Parameters
        ----------
        x1, x2 : float  – lens-plane coordinates
        rs      : float  – dimensionless NFW scale radius

        Returns
        -------
        psi_xx, psi_xy, psi_yy : float
        """
        r = np.sqrt(x1**2 + x2**2)

        kappa   = self._convergence(r, rs)        # local convergence
        kappa_bar = self._mean_convergence(r, rs) # mean convergence
        gamma   = kappa_bar - kappa               # shear magnitude

        r2 = r**2
        cos2phi = (x1**2 - x2**2) / r2           # = cos(2φ)
        sin2phi = 2.0 * x1 * x2 / r2             # = sin(2φ)

        psi_xx = kappa + gamma * cos2phi
        psi_yy = kappa - gamma * cos2phi
        psi_xy = gamma * sin2phi

        return psi_xx, psi_xy, psi_yy
    def image_type(self, x1, x2, rs):
        """
        Classify an image position using the eigenvalues of the lensing
        Jacobian  A = I − ∇∇ψ.

        Morse index n:
            n = 0   → minimum   (both eigenvalues of A > 0, det A > 0, tr A > 0)
            n = 1/2 → saddle    (det A < 0  ↔  one eigenvalue negative)
            n = 1   → maximum   (both eigenvalues < 0, det A > 0, tr A < 0)
                      (rare for NFW since there is no central core)
        """
        psi_xx, psi_xy, psi_yy = self.Hessian(x1, x2, rs)

        A = np.array([
            [1.0 - psi_xx, -psi_xy],
            [-psi_xy,       1.0 - psi_yy]
        ])
        eigvals = np.linalg.eigvalsh(A)   # always real for symmetric A

        if eigvals[0] * eigvals[1] < 0.0:
            return {"type": "saddle",   "Morse": 0.5}
        elif eigvals[0] > 0.0:
            return {"type": "minimum",  "Morse": 0}
        else:
            return {"type": "maximum",  "Morse": 1}
    
    def _nfw_guesses(self, y1, y2, rs):
        """
        Build a set of starting points for fsolve that covers where NFW
        images actually form.

        NFW images lie:
          • Along the source azimuth φ_s  (same side, larger r)
          • Opposite  azimuth φ_s + π     (counter-image, smaller r)
          • Near the tangential critical curve  r ~ 1/rs  (Einstein ring scale)

        Strategy: radial grid in both directions × angular perturbations.
        """
        y     = np.hypot(y1, y2)
        phi_s = np.arctan2(y2, y1) if y > 1e-8 else 0.0

        # Characteristic radii:  0.2 … 5 × θ_E  (θ_E ~ 1/rs in these units)
        r_scale = 1.0 / rs if rs > 0 else 1.0
        radii   = r_scale * np.array([0.1, 0.3, 0.6, 1.0, 1.5, 2.5, 4.0])

        guesses = []

        # Along and opposite the source direction (NFW main images)
        for r in radii:
            for dphi in [0.0, np.pi]:
                phi = phi_s + dphi
                guesses.append([r * np.cos(phi), r * np.sin(phi)])

        # Tangential arc: ring at r_scale with 8 equally-spaced angles
        for phi in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            guesses.append([r_scale * np.cos(phi), r_scale * np.sin(phi)])

        # Slight off-axis nudges (helps for nearly-aligned configurations)
        if y > 1e-8:
            for sign in [+1.0, -1.0]:
                guesses.append([y1 + sign * 0.5, y2])
                guesses.append([y1,               y2 + sign * 0.5])

        return guesses
    

    def geometric_optics_amplification(self, w, y1, y2, rs):
        """
        True geometric optics amplification for NFW lens.

        Implements:
            F_geo(w) = Σ μ_i exp(i w Δt_i - i π n_i)

        where:
            μ_i  = magnification
            Δt_i = dimensionless time delay (Fermat potential difference)
            n_i  = Morse index
        """

        sol = self.solve_images(y1, y2, rs)

        if len(sol) == 0:
            return 0.0 + 0.0j

        # Fermat potentials
        fermat = np.array([img["fermat_potential"] for img in sol])
        phi_min = np.min(fermat)

        F_geo = 0.0 + 0.0j

        for img in sol:

            mu = img["magnification"]
            phi = img["fermat_potential"]
            n   = img["Morse"]

            dt = phi - phi_min

            phase = np.exp(1j * w * dt - 1j * np.pi * n)

            F_geo += mu * phase

        return F_geo
    
    def geometric_optics_grid(self, w_values, y, rs):
        """
        Computes F_geo(w) for fixed y over full frequency grid.
        """

        w_values = np.asarray(w_values)

        F = np.zeros_like(w_values, dtype=np.complex128)

        for i, w in enumerate(w_values):
            F[i] = self.geometric_optics_amplification(w, y, 0.0, rs)

        return F
    
    def wave_geo_deviation_map(self, w_vals, y_vals, rs):
        """
        Returns:
            deviation(w,y), wave region mask (>10%)
        """

        F_wave = np.zeros((len(y_vals), len(w_vals)))
        F_geo  = np.zeros((len(y_vals), len(w_vals)))

        for i, y in enumerate(y_vals):

            # precompute geometric optics once per y
            geo_y = np.array([
                self.geometric_optics_amplification(w, y, 0.0, rs)
                for w in w_vals
            ])

            F_geo[i, :] = np.abs(geo_y)

            # wave optics
            Psi_nfw = self

            It = self.It_SingleIntegral_C(Psi_nfw, y, p_prec={'tmax': 1e5})
            Fw = self.Fw_FFT_C(It, p_prec={'wmin': w_vals.min(), 'wmax': w_vals.max()})

            F_wave[i, :] = np.abs(Fw.eval_Fw(w_vals))

        # deviation
        deviation = np.abs(F_wave - F_geo) / (F_geo + 1e-30)

        wave_region = deviation > 0.10

        return F_wave, F_geo, deviation, wave_region

    def magnification(self, x1, x2,rs):

        psi_xx, psi_xy, psi_yy = self.Hessian(x1, x2,rs)

        A = np.array([
            [1 - psi_xx, -psi_xy],
            [-psi_xy, 1 - psi_yy]
        ])

        detA = np.linalg.det(A)

        return 1.0 / detA
    from scipy.optimize import brentq

    from scipy.optimize import brentq

    def solve_images(self, y1, y2, rs, tol=1e-10):
        """
        Robust NFW image solver.

        Returns the physical image solutions for a circular NFW lens.

        Maximum number of images:
            • 1 outside caustic
            • 3 inside caustic
        """

        y = np.hypot(y1, y2)

        if y > 1e-14:
            phi_s = np.arctan2(y2, y1)
        else:
            phi_s = 0.0

        # ============================================================
        # Stable scalar deflection α(r)
        # ============================================================

        def alpha(r):

            r = np.maximum(r, 1e-14)

            u = rs * r

            if abs(u - 1.0) < 1e-10:
                h = 1.0

            elif u < 1.0:

                s = np.sqrt(1.0 - u*u)

                if s < 1e-14:
                    h = 1.0
                else:
                    h = np.arctanh(s) / s

            else:

                t = np.sqrt(u*u - 1.0)

                if t < 1e-14:
                    h = 1.0
                else:
                    h = np.arctan(t) / t

            return (np.log(u / 2.0) + h) / r

        # ============================================================
        # Radial equations
        # ============================================================

        def f_same(r):
            return r - alpha(r) - y

        def f_counter(r):
            return alpha(r) - r - y

        # ============================================================
        # Root search interval
        # ============================================================

        rmin = 1e-8
        rmax = max(30.0 / rs, 30.0)

        grid = np.logspace(np.log10(rmin),
                        np.log10(rmax),
                        4000)

        roots = []

        # ============================================================
        # Find bracketed roots
        # ============================================================

        for side, func in [("same", f_same),
                        ("counter", f_counter)]:

            vals = np.array([func(r) for r in grid])

            for i in range(len(grid)-1):

                r1 = grid[i]
                r2 = grid[i+1]

                f1 = vals[i]
                f2 = vals[i+1]

                if not np.isfinite(f1) or not np.isfinite(f2):
                    continue

                # Require TRUE sign crossing
                if f1 * f2 > 0:
                    continue

                try:

                    rsol = brentq(func,
                                r1,
                                r2,
                                xtol=1e-14,
                                rtol=1e-14)

                except Exception:
                    continue

                # verify root
                if abs(func(rsol)) > 1e-8:
                    continue

                roots.append((side, rsol))

        # ============================================================
        # Remove duplicates robustly
        # ============================================================

        cleaned = []

        roots.sort(key=lambda z: z[1])

        for side, r in roots:

            duplicate = False

            for side2, r2 in cleaned:

                if side != side2:
                    continue

                # relative clustering
                rel = abs(r - r2) / max(abs(r2), 1e-10)

                if rel < 1e-5:
                    duplicate = True
                    break

            if not duplicate:
                cleaned.append((side, r))

        # ============================================================
        # NFW cannot produce >3 physical images
        # ============================================================

        if len(cleaned) > 3:

            # keep only best roots
            scored = []

            for side, r in cleaned:

                if side == "same":
                    resid = abs(f_same(r))
                else:
                    resid = abs(f_counter(r))

                scored.append((resid, side, r))

            scored.sort(key=lambda z: z[0])

            cleaned = [(s, r) for _, s, r in scored[:3]]

        # ============================================================
        # Convert back to image plane
        # ============================================================

        results = []

        for side, r in cleaned:

            if side == "same":

                x1 = r * np.cos(phi_s)
                x2 = r * np.sin(phi_s)

            else:

                x1 = -r * np.cos(phi_s)
                x2 = -r * np.sin(phi_s)

            # Verify full 2D equation
            gx, gy = self.grad_potential(x1, x2, rs)

            residual = np.sqrt(
                (x1 - gx - y1)**2 +
                (x2 - gy - y2)**2
            )

            if residual > 1e-7:
                continue

            itype = self.image_type(x1, x2, rs)

            try:
                mu = self.magnification(x1, x2, rs)
            except Exception:
                mu = np.inf

            results.append({

                "position": (x1, x2),

                "magnification": mu,

                "type": itype["type"],

                "Morse": itype["Morse"],

                "method": "numeric"

            })

        # ============================================================
        # Fermat potential
        # ============================================================

        for img in results:

            x1, x2 = img["position"]

            phi = self.Fermat_Potential(
                x1, x2,
                y1, y2,
                rs
            )

            img["fermat_potential"] = phi

        # ============================================================
        # Time delays
        # ============================================================

        if results:

            phi_min = min(
                img["fermat_potential"]
                for img in results
            )

            for img in results:

                img["dimensionless_time_delay"] = (
                    img["fermat_potential"] - phi_min
                )

        # ============================================================
        # Sort by arrival time
        # ============================================================

        results.sort(
            key=lambda z: z["dimensionless_time_delay"]
        )

        return results
    
    def solve_images_old(self, y1, y2, rs, method="numeric", tol=1e-6):
        """
        Solve for NFW image positions, magnifications, Morse indices,
        and dimensionless time delays.

        Parameters
        ----------
        y1, y2 : float  – source position (lens-plane units)
        rs     : float  – dimensionless NFW normalization parameter (like in GLoW paper)
        method : str    – only "numeric" is meaningful for NFW
                          (no closed-form image positions exist)
        tol    : float  – duplicate-removal threshold in the image plane

        Returns
        -------
        results : list of dict, each containing:
            position               – (x1, x2)
            magnification          – float (signed)
            type                   – "minimum" / "saddle" / "maximum"
            Morse                  – 0 / 0.5 / 1
            fermat_potential       – φ(x) = (x-y)²/2 − ψ(x)
            dimensionless_time_delay – φ(x) − φ_min
            method                 – "numeric"
        """
        results = []

        # ── lens equation: x − ∇ψ(x) = y ───────────────────────────────
        def lens_eq(x):
            gx, gy = self.grad_potential(x[0], x[1], rs)  # <-- rs threaded
            return [x[0] - gx - y1,
                    x[1] - gy - y2]

        # ── collect unique, well-converged roots ─────────────────────────
        found = []       # list of accepted solution vectors

        for guess in self._nfw_guesses(y1, y2, rs):
            try:
                sol, _, ier, _ = fsolve(lens_eq, guess, full_output=True)
            except Exception:
                continue

            if ier != 1:                               # fsolve did not converge
                continue

            residual = np.max(np.abs(lens_eq(sol)))
            if residual > tol:                # converged to a non-root
                continue

            if any(np.linalg.norm(sol - s) < 1e-4 for s in found):  # duplicate
                continue

            found.append(sol)

        # ── classify each image ──────────────────────────────────────────
        for sol in found:
            x1, x2 = sol
            itype = self.image_type(x1, x2, rs)       # <-- rs threaded
            results.append({
                "position":      (x1, x2),
                "magnification": self.magnification(x1, x2, rs),  # <-- rs
                "type":          itype["type"],
                "Morse":         itype["Morse"],
                "method":        "numeric",
            })

        # ── Fermat potential & relative time delays ──────────────────────
        for img in results:
            x1, x2 = img["position"]
            img["fermat_potential"] = self.Fermat_Potential(x1, x2, y1, y2, rs)  # <-- rs

        if results:
            phi_min = min(img["fermat_potential"] for img in results)
            for img in results:
                img["dimensionless_time_delay"] = img["fermat_potential"] - phi_min

        return results
    
    def F_eps_NFW(self,w, y1,k,y2=0, N=5000, eps=0.002):

        """
        Docstring for F_eps_NFW
        
        :param w: Dimensionless frequency
        :param y1: Impact parameter 
        :param y2: Impact parameter (set to 0 )
        :param k: Normalization constant , consistent with GLoW NFW normalization
        :param N: Grid resolution
        :param eps: regulaization
        """ 
        y   = np.sqrt(y1**2 + y2**2)
        w_c = w * (1.0 + 1j * eps)
        L   = max(y, 0.0) + max(3.5, 4.0 / np.sqrt(w * eps), 2.5 * k)
        r   = np.linspace(0, L, N + 1)[1:]
        dr  = r[1] - r[0]

        psi       = self.potential1(k, r)
        phi_r     = 0.5 * (r**2 + y**2) - psi
        J0        = jv(0, w_c * y * r)
        integrand = np.exp(1j * w_c * phi_r) * J0 * r
        return (w / 1j) * np.sum(integrand) * dr
    
    




    def filon_weights(self,theta, K=16):
        """
        Filon–Lagrange quadrature weights (A0, A1, A2) for one panel [x0, x0+2h].

        For the composite rule ∫ f(t) e^{iωt} dt, all panels share the same
        weights on a uniform grid (θ = ωh is fixed).

        Derived from the Lagrange quadratic interpolant on {0, h, 2h}:
            Aⱼ = ∫₀² Lⱼ(u) e^{iθu} du

        which reduces to A0 = ½(I₂−3I₁+2I₀), A1 = −(I₂−2I₁), A2 = ½(I₂−I₁)
        via the moments Iₙ = ∫₀² uⁿ e^{iθu} du.

        At θ=0 → Simpson weights (1/3, 4/3, 1/3).  Exact for quadratic f.

        Parameters
        ----------
        theta : float  – ωh, phase accumulation per half-panel
        K     : int    – Taylor truncation order (used when |θ| < 0.5)
        """
        if abs(theta) < 0.5:
            # Stable Taylor series: Iₙ = Σₖ (iθ)ᵏ/k! · 2^{n+k+1}/(n+k+1)
            # avoids catastrophic cancellation in the direct formulas near θ=0
            k   = np.arange(K, dtype=float)
            qk  = (1j * theta)**k / sp_fac(k, exact=False)   # (iθ)ᵏ/k!
            p2  = 2.0**k
            I0  = np.sum(qk * 2.0 * p2 / (k + 1.0))          # n=0: 2^{k+1}/(k+1)
            I1  = np.sum(qk * 4.0 * p2 / (k + 2.0))          # n=1: 2^{k+2}/(k+2)
            I2  = np.sum(qk * 8.0 * p2 / (k + 3.0))          # n=2: 2^{k+3}/(k+3)
        else:
            # Direct analytic recurrence (IBP on ∫₀² uⁿ e^{iθu} du)
            P  = np.exp(2j * theta)
            q  = 1j * theta
            I0 = (P - 1.0) / q
            I1 = 2.0 * P / q - I0 / q
            I2 = 4.0 * P / q - 2.0 * I1 / q

        A0 = 0.5 * (I2 - 3.0*I1 + 2.0*I0)
        A1 = -(I2 - 2.0*I1)
        A2 = 0.5 * (I2 - I1)
        return A0, A1, A2

    def F_filon_NFW(self, w, y, rs, N=None, T=None, tail_order=1):
        """
        NFW wave-optics amplification F(w) via Filon quadrature.

        After t = r²/2 the Kirchhoff integral becomes:
            F = (w/i) exp(iwy²/2) ∫₀ᵀ f(t) e^{iwt} dt
        with f(t) = J₀(wy√(2t)) exp(-iw ψ(√(2t))).

        Filon treats e^{iwt} analytically (exact for any panel width)
        and approximates f as a quadratic per panel — so accuracy does
        not degrade with increasing w.  The required N scales as wy/rs
        rather than w·L/π as in F_eps_NFW.

        Parameters
        ----------
        w          : float  – dimensionless frequency
        y          : float  – impact parameter
        rs         : float  – NFW scale parameter (= k in potential1)
        N          : int    – number of panels, must be even (default: adaptive)
        T          : float  – upper limit in t-space = r_max²/2 (default: adaptive)
        tail_order : int    – asymptotic tail-correction order: 0, 1, or 2

        Returns
        -------
        F : complex
        """
        # ── Upper limit in t-domain ───────────────────────────────────
        if T is None:
            r_max = max(y, 0.0) + max(5.0 / rs, 5.0)
            T = 0.5 * r_max**2

        # ── Grid size: resolve f(t), NOT e^{iwt} ─────────────────────
        # f oscillates at frequency ~ max(wy·rs, w·rs²/4) per unit t.
        # Filon handles e^{iwt} analytically, so N can be much smaller
        # than for direct quadrature (no need to resolve the w·t phase).
        if N is None:
            freq_bessel = w * max(y, 1e-8) * rs        # J₀ oscillation in t
            freq_psi    = 0.25 * w * rs**2             # ψ oscillation in t
            freq        = max(freq_bessel, freq_psi, 1.0)
            N = max(256, 2 * int(10.0 * T * freq / np.pi + 1))
        elif N % 2:
            N += 1

        h     = T / N
        theta = w * h                          # phase per half-panel

        # ── Grid and amplitude f(t) ───────────────────────────────────
        t    = np.linspace(0.0, T, N + 1)     # (N+1,) nodes
        r    = np.sqrt(2.0 * t)
        r[0] = 0.0                             # avoid √0 in potential

        psi       = self.potential1(rs, np.maximum(r, 1e-300))
        psi[0]    = 0.0                        # ψ(0) = 0 exactly

        J0        = jv(0, w * y * r)          # real w — no complex Bessel
        J0[0]     = 1.0

        f = J0 * np.exp(-1j * w * psi)

        # ── Composite Filon rule ──────────────────────────────────────
        # Weights are the same for every panel (uniform grid → fixed θ)
        A0, A1, A2 = self.filon_weights(theta)

        g      = f * np.exp(1j * w * t)       # gⱼ = fⱼ · e^{iw·tⱼ}
        g_even = g[0::2]                       # j = 0, 2, …, N  (len N/2+1)
        g_odd  = g[1::2]                       # j = 1, 3, …, N-1 (len N/2)

        # Node contributions:
        #   left endpoint  t₀         → A0
        #   right endpoint t_N         → A2
        #   interior even  t_{2k}     → A0 + A2  (shared between two panels)
        #   all odd nodes  t_{2k+1}   → A1       (panel midpoints)
        S = (  A0          * g_even[0]
             + A2          * g_even[-1]
             + (A0 + A2)   * g_even[1:-1].sum()
             + A1          * g_odd.sum()       )

        integral = h * S

        # ── Asymptotic tail correction ∫_T^∞ f(t) e^{iwt} dt ─────────
        # Integration by parts:
        #   ∫_T^∞ f e^{iwt} dt = -f(T)/(iw)·e^{iwT} + f'(T)/(iw)²·e^{iwT} + …
        if tail_order >= 1:
            rT   = np.sqrt(2.0 * T)
            psiT = self.potential1(rs, np.array([rT]))[0]
            J0T  = float(jv(0, w * y * rT))
            fT   = J0T * np.exp(-1j * w * psiT)
            eiT  = np.exp(1j * w * T)

            integral += (-fT / (1j * w)) * eiT

            if tail_order >= 2:
                # f'(T) w.r.t. t:
                #   dJ₀/dt  = -J₁(wyr)·wy/r  (r = √(2t), dr/dt = 1/r)
                #   dψ/dt   = ψ'(r)/r = α(r)/r
                J1T      = float(jv(1, w * y * rT))
                alpha_rT = (np.log(rs * rT / 2.0) + self._h(rs * rT)) / rT
                dfT_dt   = (  (-w * y * J1T / rT)
                             - (1j * w * alpha_rT / rT) * J0T
                            ) * np.exp(-1j * w * psiT)
                integral += (dfT_dt / (1j * w)**2) * eiT

        return (w / 1j) * np.exp(0.5j * w * y**2) * integral
                    
    
    def Amplification(self,w,y,rs,method="numeric"):
        """
        Computes geometric optics amplification factor for Point Mass Lens

        Parameters:
        
        w: Dimensionless frequency
        y: Impact parameter

        type: Nature of the image: minimum, saddle, both

        """

        
        y1 = y
        y2 = 0

        sol = self.solve_images(y1,y2,rs)

        mu = []
        td = []
        n = []
        
        for img in sol:
            mag = img["magnification"]
            dimensionless_td = img["dimensionless_time_delay"]
            MorseIndex = img["Morse"]
            mu.append(mag)
            td.append(dimensionless_td)
            n.append(MorseIndex)

        #magnification = self.Analytical_Magnification(y,type=type)


        return solver.GO(w,td,mu,n)
    
    @staticmethod
    def Dimensionless_Frequency(lensmass,f):

        """
        Computes the dimensionss frequency mentioned in Takahashi 2003

        Parameters:

        lensmass: The mass of the lens in solar units
        f : Frequency of the gravitational wave source in units of Hz

        Returns: The dimensionless frequency 
        
        """
    
        const = 24*np.pi*G/c**3

        w = const*lensmass*M_sun*f

        return w

def psi(x,rs):
    """
    NFW lensing potential defined as per Villarrubia-Rojo et. al. 2024
    The constant k = xi_0/r_s

    To avoid replacing all the r_s with k. I have used k = rs in this form of lensing potential. This is only for testing purpose and hence will be replaced later.


    """

    u = rs*x
    if u <= 1:

        return 0.5*(  (log(u/2))**2 - (atanh(sqrt(1-u**2)))**2 )
    else:

        return 0.5*( (log(u/2))**2 + (atan(sqrt(u**2 -1)))**2   )


def psi_2(x, rs):
    if x == 0:
        return 0
    elif x < 1:
        return rs/2 * ((log(x/2))**2 - (atanh(sqrt(1 - x**2)))**2)
    else:
        return rs/2 * ((log(x/2))**2 +  (atan(sqrt(x**2 - 1)))**2)

#def psi(x,rs):
#
#    bterm = mpf('2.01')   

#    return rs*( (log(x/2*bterm))**2 - (atanh(sqrt(1 - ((x**2) /bterm**2)**2)))**2)

def func(x, w, y,rs):
    # x: real integration variable (mpf)
    # w, y, amp, core, p: mpc or mpf
    sqrt2x = sqrt(2 * x)
    arg_bessel = w * y * sqrt2x
    bessel_val = besselj(0, arg_bessel)
    psi_val = psi(sqrt2x, rs)
    return bessel_val * exp(-1j * w * psi_val)

def func2(x, w, y, rs):
    return func(x, w, y, rs) * exp(1j * w * x)

def dfunc(x, w, y, rs):
    sqrt2x = sqrt(2 * x)
    psi_val = psi(sqrt2x, rs)
    J1 = besselj(1, w * y * sqrt2x)
    prefactor = -w * y / sqrt2x
    return prefactor * J1 * exp(-1j * w * psi_val) - (1j * w / (2 * x)) * func(x, w, y,rs)

def ddfunc(x, w, y,rs):
    sqrt2x = sqrt(2 * x)
    psi_val = psi(sqrt2x, rs)
    #denom = (sqrt2x**2 + core**2)**(1 - p/2)
    #dpsi = amp * p * sqrt2x / denom
    
    # derivative squared term:
    #d2psi = amp * p * (1 - p / 2) * (2 * x)**(-0.5) * (1 - 2 * x / (2 * x + core**2)) / (2 * x + core**2)**(1 - p/2)
    
    term1 = (w * y) / (2 * x * sqrt2x) * (2 + 1j * w) * besselj(1, w * y * sqrt2x) * exp(-1j * w * psi_val)
    term2 = -1 / (2 * x) * (w**2 * y**2 - 1j * w / x) * func(x, w, y, rs)
    term3 = -1j * w / (2 * x) * dfunc(x, w, y,rs)
    return term1 + term2 + term3

def NFW(w, y, rs):

    # Convert inputs to mpmath types
    w = mpf(w)        # real number
    y = mpf(y)        # real number
    rs = mpf(rs)      # real number
    a = mpf('0.00001')

    if w <= 1:
        b = mpf('100') / w
    elif 1 < w < 400000:
        b = mpf('1000') / w
    elif 1000 < w < 500000:
        b = mpf('100000') / w
    elif 500000 < w < 10000000:
        b = mpf('1000000') / w
    else:
        b = mpf('100000000') / w
    
    # mpmath.quad with complex integrand
    zz = quad(lambda x: func2(x, w, y,rs), [a, b], error=True, maxdegree=10)
    zz_val = zz[0]  # integral value
    
    # Add tail correction terms (at b)
    tail = (-func(b, w, y, rs) / (1j * w) * exp(1j * w * b)
            - dfunc(b, w, y, rs) / (w**2) * exp(1j * w * b)
            + ddfunc(b, w, y, rs) / (1j * w**3) * exp(1j * w * b))
    
    zz_val += tail
        
    return -1j * w * exp(0.5 * 1j * w * y**2) * zz_val


def _worker(args):
    i, j, w, y, rs = args
    val = NFW(w, y, rs)
    return i, j, val.real, val.imag



def compute_nfw_lens_grid(w_values, y_values, rs , hdf5_filename,num_processes=None):
    """
    Computes the Point(w, y) complex amplification factor over a grid
    and stores the result in an HDF5 file.

    Parameters:
    - w_values (1D np.array): dimensionless frequency array
    - y_values (1D np.array): impact parameter array
    - hdf5_filename (str): path to output HDF5 file
    - num_processes (int or None): number of CPU cores to use (default: all)
    """

    #Read the config file
    config = configparser.ConfigParser()
    config.read('parameters.ini')

    # Setup logging
    logger = logging.getLogger("compute_logger")
    logger.setLevel(logging.INFO)


    #Store result
    output_folder = config['Output']['outdir']

    # Subfolder for interpolation outputs
    log_folder = os.path.join(output_folder, 'logs')
    os.makedirs(log_folder, exist_ok=True)  # Create if it doesn't exist

    log_file = os.path.join(log_folder,'interpolation.log')


    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    W_len = len(w_values)
    Y_len = len(y_values)
    total_points = W_len * Y_len

    w_values = [mpc(w) for w in w_values]
    y_values = [mpc(y) for y in y_values]

    # Create task list
    tasks = [(i, j, w_values[i], y_values[j], rs ) for i in range(W_len) for j in range(Y_len)]

    logger.info(f" Output file: {hdf5_filename}")
    logger.info(f" Grid size: {W_len} × {Y_len} → {total_points} points")
    logger.info(f"Launching parallel computation with {num_processes} tasks")

    start_time = time.time()
    w_array = np.array([float(w) for w in w_values], dtype='f8')
    y_array = np.array([float(y) for y in y_values], dtype='f8')
    with h5py.File(hdf5_filename, "w") as h5file:
        h5file.create_dataset("w_values", data=w_array)
        h5file.create_dataset("y_values", data=y_array)
        dset = h5file.create_dataset(
            "results",
            shape=(W_len, Y_len, 2),
            dtype="f8",
            chunks=(1, Y_len, 2)  # optimize for row-wise access
        )

        with mpr.Pool(processes=num_processes or mpr.cpu_count()) as pool:
            for count, (i, j, re, im) in enumerate(pool.imap_unordered(_worker, tasks, chunksize=100), 1):
                dset[i, j, 0] = re
                dset[i, j, 1] = im

                if count % 1000 == 0 or count == total_points:
                    logger.info(f" {count}/{total_points} results written...")

        h5file.flush()

    end_time = time.time()
    logger.info(f" Done in {end_time - start_time:.2f} seconds")
    logger.info(f" Saved to: {hdf5_filename}")

def compute_nfw_lens_test(w_values, y_values, rs_values, hdf5_filename, num_jobs=None):
    """
    Computes the NFW(w, y, rs) complex amplification factor over a 3D grid,
    and stores real and imaginary parts in an HDF5 file.

    Parameters:
    - w_values (np.ndarray): 1D array of dimensionless frequency
    - y_values (np.ndarray): 1D array of impact parameter
    - rs_values (np.ndarray): 1D array of scale radii
    - hdf5_filename (str): Path to output HDF5 file
    - num_jobs (int or None): Number of parallel jobs to use (default: all cores)
    """

    W_len, Y_len, R_len = len(w_values), len(y_values), len(rs_values)
    total = W_len * Y_len * R_len
    logger.info(f"Preparing {total} NFW evaluations ({W_len}×{Y_len}×{R_len})")
    logger.info(f"Saving to: {hdf5_filename}")

    # Start timer
    t0 = time.time()

    # Flatten all grid points as tasks
    tasks = [(i, j, k, w_values[i], y_values[j], rs_values[k])
             for i in range(W_len)
             for j in range(Y_len)
             for k in range(R_len)]

    def _compute_point(i, j, k, w, y, rs):
        val = NFW(w, y, rs)
        return (i, j, k, val.real, val.imag)

    # Run in parallel
    logger.info(f"Running in parallel with {num_jobs or os.cpu_count()} cores...")
    results = Parallel(n_jobs=num_jobs or -1, backend="loky", verbose=10)(
        delayed(_compute_point)(i, j, k, w, y, rs) for (i, j, k, w, y, rs) in tasks
    )

    # Allocate result array
    result_array = np.zeros((W_len, Y_len, R_len, 2), dtype=np.float64)
    for i, j, k, re, im in results:
        result_array[i, j, k, 0] = re
        result_array[i, j, k, 1] = im

    # Save everything to HDF5
    with h5py.File(hdf5_filename, "w") as f:
        f.create_dataset("w_values", data=w_values)
        f.create_dataset("y_values", data=y_values)
        f.create_dataset("rs_values", data=rs_values)
        f.create_dataset("results", data=result_array, chunks=True, compression="gzip")

    logger.info(f"Completed in {time.time() - t0:.2f} seconds.")
    logger.info("Interpolation grid saved and ready.")


def compute_nfw_lens_grid_mpi(w_values, y_values, rs_values,
                              hdf5_filename, t_start=None,
                              compression="gzip"):
    """
    NFW(w, y, rs) over a 3D grid, computed with MPI.

    Results are stored as (W_len, Y_len, R_len, 2), where the last
    axis is [real, imag].
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    W_len = len(w_values)
    Y_len = len(y_values)
    R_len = len(rs_values)
    total_points = W_len * Y_len * R_len

    if t_start is not None:
        max_startup = comm.allreduce(time.time() - t_start, op=MPI.MAX)
    else:
        max_startup = -1.0

    # ---- logging: rank 0 only ----
    logger = None
    if rank == 0:
        logger = logging.getLogger("compute_logger")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        try:
            config = configparser.ConfigParser()
            config.read("parameters.ini")
            output_folder = config["Output"]["outdir"]
        except Exception:
            output_folder = "."

        log_folder = os.path.join(output_folder, "logs")
        os.makedirs(log_folder, exist_ok=True)

        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                "%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(os.path.join(log_folder, "interpolation.log"))
        sh = logging.StreamHandler()
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)

        logger.info(f"Ranks: {size}")
        logger.info(f"Grid: {W_len} x {Y_len} x {R_len} = {total_points} points")
        logger.info(f"Tasks per rank: ~{total_points // size}")
        logger.info(f"mpmath precision: mp.dps = {mp.dps}")
        logger.info(f"Slowest-rank startup: {max_startup:.1f} s")

    t0 = time.time()

    # ---- mpmath inputs, rebuilt locally on every rank ----
    w_mp = [mpf(float(w)) for w in w_values]
    y_mp = [mpf(float(y)) for y in y_values]
    rs_mp = [mpf(float(r)) for r in rs_values]

    # ---- flatten the 3D index space, then stride ----
    tasks = [(i, j, k)
             for i in range(W_len)
             for j in range(Y_len)
             for k in range(R_len)]
    local_tasks = tasks[rank::size]

    local_results = []
    for i, j, k in local_tasks:
        amp = NFW(w_mp[i], y_mp[j], rs_mp[k])      # mpc
        local_results.append((i, j, k, float(amp.real), float(amp.imag)))

    t_compute = time.time() - t0

    # ---- balance diagnostics ----
    t_max = comm.allreduce(t_compute, op=MPI.MAX)
    t_min = comm.allreduce(t_compute, op=MPI.MIN)
    t_sum = comm.allreduce(t_compute, op=MPI.SUM)

    all_results = comm.gather(local_results, root=0)

    if rank == 0:
        logger.info(f"Compute  slowest rank: {t_max:.1f} s")
        logger.info(f"Compute  fastest rank: {t_min:.1f} s")
        logger.info(f"Compute  mean rank:    {t_sum/size:.1f} s")
        logger.info(f"Balance efficiency: {100*(t_sum/size)/t_max:.1f} %")

        flat = [item for sub in all_results for item in sub]
        if len(flat) != total_points:
            logger.warning(f"Expected {total_points} results, got {len(flat)}")

        # NaN-fill so a missing point is visibly wrong, not silently zero
        grid = np.full((W_len, Y_len, R_len, 2), np.nan, dtype="f8")
        for i, j, k, re_, im_ in flat:
            grid[i, j, k, 0] = re_
            grid[i, j, k, 1] = im_

        t_write = time.time()
        with h5py.File(hdf5_filename, "w") as h5:
             # ---- file-level metadata ----                                  
            try:                                                              
                gw_ver = version("gwlens")                                    
            except PackageNotFoundError:                                       
                gw_ver = "unknown"                                             
            h5.attrs["pipeline"] = "gwlens"                                    
            h5.attrs["gwlens_version"] = gw_ver                                
            h5.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()  
            h5.attrs["mpi_ranks"] = size                                      
            h5.attrs["mp_dps"] = mp.dps                                       

            for name, vals in (("w", w_values), ("y", y_values), ("rs", rs_values)):  
                h5.attrs[f"{name}_min"] = float(vals[0])                      
                h5.attrs[f"{name}_max"] = float(vals[-1])                     
                h5.attrs[f"{name}_num"] = len(vals)                           

                                    


            h5.create_dataset("w_values",  data=np.asarray(w_values,  dtype="f8"))
            h5.create_dataset("y_values",  data=np.asarray(y_values,  dtype="f8"))
            h5.create_dataset("rs_values", data=np.asarray(rs_values, dtype="f8"))
            dset = h5.create_dataset(
                "results",
                data=grid,
                chunks=(1, Y_len, R_len, 2),
                compression=compression,
            )
            dset.attrs["axes"] = "w, y, rs, [real|imag]"
            dset.attrs["mp_dps"] = mp.dps

        logger.info(f"HDF5 write: {time.time() - t_write:.1f} s")
        logger.info(f"Saved to {hdf5_filename}")
