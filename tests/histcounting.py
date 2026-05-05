import os, sys, time
import numpy as np
import pandas as pd
from scipy import signal
import scipy.interpolate as si
import matplotlib.pyplot as plt


from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

from histlens import TimeDelay, getMinMaxSaddle_flag, amplification_from_data, TFunc, dTFunc

def Images(beta_ra, beta_dec, zL, zS, lens_model_list, kwargs_lens_list, details=False):
    lens_model = LensModel(lens_model_list=lens_model_list)
    solver     = LensEquationSolver(lens_model)
    img_p      = solver.image_position_from_source(beta_ra, beta_dec, kwargs_lens_list)

    Roll   = len(img_p[0])
    img_p1 = np.roll(img_p[0], Roll, axis=0)
    img_p2 = np.roll(img_p[1], Roll, axis=0)

    mu   = lens_model.magnification(img_p1, img_p2, kwargs_lens_list)
    td   = TimeDelay(img_p1, img_p2, beta_ra, beta_dec, zL, zS, lens_model_list, kwargs_lens_list)
    n    = getMinMaxSaddle_flag(img_p1, img_p2, lens_model_list, kwargs_lens_list)
    xI12 = [img_p1, img_p2]

    if details:
        print('Number of Images:', len(n))
        print('Magnification:', mu)
        print('Positions (x1,x2):', img_p)
        print('Time Delay:', td)

    return len(n), xI12, mu, td, n


def SmoothingFunc(x, y, dx_interpolate, outlier_sigma=3, rolling_window=5):
    """Remove outliers via rolling median, then interpolate."""
    rolling_median = (pd.Series(y)
                      .rolling(rolling_window, center=True)
                      .median().bfill().ffill().to_numpy())
    rolling_sigma  = np.abs(y - rolling_median)
    sigma_median   = np.nanmedian(rolling_sigma)

    mask = rolling_sigma <= outlier_sigma * sigma_median
    x, y = x[mask], y[mask]

    rolling_median = (pd.Series(y)
                      .rolling(rolling_window, center=True)
                      .median().bfill().ffill().to_numpy())

    x_sampled = np.arange(x[0], x[-1], dx_interpolate)
    y_sampled = si.interp1d(x, rolling_median, kind='cubic')(x_sampled)
    return x_sampled, y_sampled

# Change 1 & 2: pure-numpy TreeClass; FtcFunc/FwcFunc take arrays not DataFrames

class TreeClass:
    """Adaptive tree. Accumulates good nodes as lists """

    def __init__(self, dt_require, tlims):
        self.dt_require = dt_require
        self.tlims      = tlims
        self._x1 = []
        self._x2 = []
        self._t  = []
        self._w  = []

    def get_good_arrays(self):
        """Return (x1, x2, t, weights) as numpy arrays."""
        return (np.asarray(self._x1), np.asarray(self._x2),
                np.asarray(self._t),  np.asarray(self._w))

    def SplitFunc(self, weights, x1_list, x2_list, T_list, dT_list):
        error_list = dT_list * (weights ** 0.5)
        flag_error = error_list < self.dt_require

        T_min_list = T_list - error_list
        T_max_list = T_list + error_list
        flag_tlim  = (T_min_list < self.tlims[1]) & (T_max_list > self.tlims[0])

        flag_bad  = flag_tlim & ~flag_error
        flag_good = flag_tlim &  flag_error

        self._x1.extend(x1_list[flag_good].tolist())
        self._x2.extend(x2_list[flag_good].tolist())
        self._t .extend(T_list [flag_good].tolist())
        self._w .extend((weights * np.ones(flag_good.sum())).tolist())

        self.bad_nodes = (x1_list[flag_bad], x2_list[flag_bad],
                          T_list [flag_bad], dT_list[flag_bad])


def FtcFunc(tI_arr, muI_arr, typeI_arr, t_list):
    """Singular F(t); takes plain arrays instead of a DataFrame."""
    tI_arr = tI_arr - tI_arr.min()
    Ftc    = np.zeros_like(t_list)
    for tI, muI, typ in zip(tI_arr, muI_arr, typeI_arr):
        if   typ == 'min':
            Ftc[t_list >= tI] += 2. * np.pi * muI**0.5
        elif typ == 'max':
            Ftc[t_list >= tI] -= 2. * np.pi * muI**0.5
        elif typ == 'saddle':
            
            Ftc += -2. * (-muI)**0.5 * np.log(np.abs(t_list - tI))
        else:
            raise ValueError('Unknown image type: {}'.format(typ))
    return Ftc


def FwcFunc(tI_arr, muI_arr, typeI_arr, w_list):
    """Singular F(w); takes plain arrays instead of a DataFrame."""
    tI_arr = tI_arr - tI_arr.min()
    Fwc    = np.zeros_like(w_list, dtype=complex)
    for tI, muI, typ in zip(tI_arr, muI_arr, typeI_arr):
        if   typ == 'min':
            Fwc += muI**0.5   * np.exp(1j * w_list * tI)
        elif typ == 'max':
            Fwc += muI**0.5   * np.exp(1j * w_list * tI - 1j * np.pi)
        elif typ == 'saddle':
            Fwc += (-muI)**0.5 * np.exp(1j * w_list * tI - 1j * np.pi * 0.5)
        else:
            raise ValueError('Unknown image type: {}'.format(typ))
    return Fwc




def FtHistFunc(beta_ra, beta_dec, zL, zS, lens_model_list, kwargs_lens_list,
               tlim_list=[0.5, 10, 100], dt_list=[5e-3, 5e-2, 5e-1]):
    """
    Calculate F(t) with adaptive histogram counting.
    Returns plain numpy arrays for tI, muI, typeI (no pandas).
    """
    nimages, xI12, muI, tI, typeI = Images(beta_ra, beta_dec, zL, zS,
                                            lens_model_list, kwargs_lens_list)
    print('Found {} images'.format(nimages))
    print('positions',     [(round(x,3), round(y,3)) for x,y in zip(xI12[0], xI12[1])])
    print('magnification', [round(float(m),3) for m in muI])
    print('time delay',    [round(float(t),3) for t in tI])

    xI1_arr   = np.asarray(xI12[0])
    xI2_arr   = np.asarray(xI12[1])
    muI_arr   = np.asarray(muI,   dtype=float)
    tI_arr    = np.asarray(tI,    dtype=float)
    typeI_arr = np.asarray(typeI)
    print("time delays:", tI_arr - tI_arr.min())   # shifted delays
    print("image types:", typeI_arr)
    print("magnifications:", muI_arr)

    tImin = tI_arr.min()
    tImax = tI_arr.max()
    tmax  = tImax + tlim_list[-1]

    dxI1 = float(xI1_arr.max() - xI1_arr.min()) or 1.0
    dxI2 = float(xI2_arr.max() - xI2_arr.min()) or 1.0
    boundI_x1 = [xI1_arr.min() - dxI1, xI1_arr.max() + dxI1]
    boundI_x2 = [xI2_arr.min() - dxI2, xI2_arr.max() + dxI2]

    # extend bounds until boundary T > tmax
    while True:
        N_b  = 500
        tmp  = np.linspace(boundI_x1[0], boundI_x1[1], N_b)
        x1_t = np.concatenate([tmp, np.full(N_b, boundI_x1[1]),
                                tmp, np.full(N_b, boundI_x1[0])])
        tmp  = np.linspace(boundI_x2[0], boundI_x2[1], N_b)
        x2_t = np.concatenate([np.full(N_b, boundI_x2[1]), tmp,
                                np.full(N_b, boundI_x2[0]), tmp])
        T_t = TFunc(x1_t, x2_t, beta_ra, beta_dec, zL, zS, lens_model_list, kwargs_lens_list)
        if T_t.min() > tmax:
            break
        boundI_x1 = [boundI_x1[0] - 0.5*dxI1, boundI_x1[1] + 0.5*dxI1]
        boundI_x2 = [boundI_x2[0] - 0.5*dxI2, boundI_x2[1] + 0.5*dxI2]

    # coarse initial grid — adaptive refinement handles precision
    N_x = 700
    x1_node = np.linspace(boundI_x1[0], boundI_x1[1], N_x)
    x2_node = np.linspace(boundI_x2[0], boundI_x2[1], N_x)
    dx1 = x1_node[1] - x1_node[0]
    dx2 = x2_node[1] - x2_node[0]
    x1_grid, x2_grid = np.meshgrid(x1_node, x2_node)
    x1_list = x1_grid.flatten()
    x2_list = x2_grid.flatten()

    T_list            = TFunc( x1_list, x2_list, beta_ra, beta_dec, zL, zS, lens_model_list, kwargs_lens_list)
    dtdx1, dtdx2      = dTFunc(x1_list, x2_list, beta_ra, beta_dec, zL, zS, lens_model_list, kwargs_lens_list)
    dT_list           = np.hypot(dtdx1, dtdx2)

    t_list_final, Ft_list_final = [], []

    for i_reg, tlim in enumerate(tlim_list):
        tmin_bin = 0 if i_reg == 0 else tmax_bin
        tmax_bin = tImax + tlim
        dt       = dt_list[i_reg]

        Tree = TreeClass(dt, [tmin_bin, tmax_bin])
        Tree.SplitFunc(dx1*dx2, x1_list, x2_list, T_list, dT_list)

        dx1_tmp, dx2_tmp = dx1, dx2
        MAX_ITER = 80

        for _ in range(MAX_ITER):
            if len(Tree.bad_nodes[0]) == 0:
                break
            N_bad  = len(Tree.bad_nodes[0])
            x1_bad = np.repeat(Tree.bad_nodes[0], 4)
            x2_bad = np.repeat(Tree.bad_nodes[1], 4)
            x1_sub = x1_bad + np.tile([-0.25*dx1_tmp,  0.25*dx1_tmp,
                                        -0.25*dx1_tmp,  0.25*dx1_tmp], N_bad)
            x2_sub = x2_bad + np.tile([ 0.25*dx2_tmp,  0.25*dx2_tmp,
                                        -0.25*dx2_tmp, -0.25*dx2_tmp], N_bad)

            T_sub        = TFunc( x1_sub, x2_sub, beta_ra, beta_dec, zL, zS, lens_model_list, kwargs_lens_list)
            d1_sub, d2_sub = dTFunc(x1_sub, x2_sub, beta_ra, beta_dec, zL, zS, lens_model_list, kwargs_lens_list)
            dT_sub       = np.hypot(d1_sub, d2_sub)
            dx1_tmp *= 0.5   # halve step: converges in ~10 iters instead of ~30
            dx2_tmp *= 0.5
            Tree.SplitFunc(dx1_tmp*dx2_tmp, x1_sub, x2_sub, T_sub, dT_sub)

        _, _, t_good, w_good = Tree.get_good_arrays()
        t_good = t_good - tImin

        N_bins            = min(max(1, int((t_good.max() - t_good.min()) / dt)), 500000)
        Ft_arr, bin_edges = np.histogram(t_good, N_bins, weights=w_good / dt)
        t_arr             = (bin_edges[1:] + bin_edges[:-1]) / 2.

        t_list_final .append(t_arr [5:-5])
        Ft_list_final.append(Ft_arr[5:-5])
    
    print(f"Region {i_reg}:bad nodes remaining = {len(Tree.bad_nodes[0])}")
    print(f"Region{i_reg}:good nodes={len(Tree._t)}")

    t_fin  = np.concatenate(t_list_final)
    Ft_fin = np.concatenate(Ft_list_final)
    Ftc    = FtcFunc(tI_arr, muI_arr, typeI_arr, t_fin)

    print("Ftc range: min={:.4g}  max={:.4g}".format(Ftc.min(), Ftc.max()))
    print("Ft_fin range: min={:.4g}  max={:.4g}".format(Ft_fin.min(), Ft_fin.max()))
    Ftd    = -Ft_fin + Ftc #changed signs here

        # In FtHistFunc, just after computing Ftc and Ftd
    idx1 = np.argmin(np.abs(t_fin - 0.289))   # near first saddle
    idx2 = np.argmin(np.abs(t_fin - 0.452))   # near second saddle

    print("Near first saddle (t=0.289):")
    print("  Ft_fin:", Ft_fin[idx1-2:idx1+3])
    print("  Ftc:   ", Ftc   [idx1-2:idx1+3])
    print("  Ftd:   ", Ftd   [idx1-2:idx1+3])

    print("Near second saddle (t=0.452):")
    print("  Ft_fin:", Ft_fin[idx2-2:idx2+3])
    print("  Ftc:   ", Ftc   [idx2-2:idx2+3])
    print("  Ftd:   ", Ftd   [idx2-2:idx2+3])

    return t_fin, Ftd, Ft_fin, tI_arr, muI_arr, typeI_arr


def HistMethod(beta_ra, beta_dec, zL, zS, lens_model_list, kwargs_lens_list,
               verbose=False, wlim=30., wmin=0.5,
               tlim_list=[0.005,0.5, 10, 100], dt_list=[5e-4,5e-3, 5e-2, 5e-1]):
    """
    Main function for the histogram counting method.

    Parameters
    ----------
    wlim : float  -- above this w, diffraction term zeroed (geo-optics dominates)
    wmin : float  -- below this w, FFT is too noisy; replaced by exact Fwc
    dt_list : list -- time-bin accuracy per region; coarser = faster (default 5e-3)
    """
    start = time.time()
    dt    = dt_list[0]

    t_ori, Ftd_ori,Ft_fin, tI_arr, muI_arr, typeI_arr = FtHistFunc(
        beta_ra, beta_dec, zL, zS, lens_model_list, kwargs_lens_list,
        tlim_list=tlim_list, dt_list=dt_list)
    mask = t_ori >= 0
    ##DEBUG
    #plt.plot(t_ori[mask], Ft_fin[mask], label='raw Ftd')
    #plt.show()
    t, Ftd = SmoothingFunc(t_ori, Ftd_ori, dt, outlier_sigma=50, rolling_window=5)
    

    t_low   = np.arange(-t[-1], t[0], dt)
    Ftc_low = 1. * FtcFunc(tI_arr, muI_arr, typeI_arr, t_low)#changed minus
    t       = np.concatenate([t_low[:-1], t])
    Ftd     = np.concatenate([Ftc_low[:-1], Ftd])
    ##DEBUG
    #plt.plot(t, Ftd, label='Ftd')
    #plt.plot(t_low, Ftc_low, label='Ftc_low')
    #plt.show()
    
    t_final, Ftd_final = SmoothingFunc(t, Ftd, dt, outlier_sigma=3, rolling_window=5)
    N_t = len(t_final)

    Fw   = np.fft.ihfft(Ftd_final) * N_t * dt
    freq = np.fft.rfftfreq(N_t, d=dt)
    w    = freq * 2. * np.pi

    t2, t1   = t_final[-1], t_final[0]
    Ft2, Ft1 = Ftd_final[-1], Ftd_final[0]
    Fw = (np.exp(1j*w*t2)*Ft2 - np.exp(1j*w*t1)*Ft1) / (2*np.pi) + Fw*w / (2j*np.pi)

    index  = signal.argrelextrema(np.abs(Fw), np.less_equal, order=3)[0]
    Fw, w  = Fw[index][1:], w[index][1:]

    Fw[w >= wlim] = 0
    Fw[w <= wmin] = 0
    Fwc  = FwcFunc(tI_arr, muI_arr, typeI_arr, w)
    Fw  -= Fwc#changed sign

    if verbose:
        print('finished in', round(time.time()-start, 2), 's')

    return w, Fw, Fwc, Ftd_final


#PML
#lens_model_list  = ['POINT_MASS']
#kwargs_lens_list = [{'theta_E': 1.0, 'center_x': 0., 'center_y': 0}]


#Bianry lens with two point masses
lens_model_list = ["POINT_MASS","POINT_MASS"]
kwargs_lens_list = [{'theta_E': 1.0, 'center_x': -0.5, 'center_y': 0}, {'theta_E': 1.0, 'center_x': 0.5, 'center_y': 0}]

#SIS
#lens_model_list = ["SIS"]
#kwargs_lens_list = [{'theta_E': 1.0, 'center_x': 0., 'center_y': 0}]

beta_ra  = 0.1
beta_dec = np.sqrt(3)/10
zL, zS   = 0.5, 2.0

#dt_list1 = [5e-4, 5e-3, 5e-2, 5e-1]
dt_list2 = [ 1e-4,1e-2,1e-1,1.0]

w, Fw, Fwc, Ftd_final = HistMethod(
    beta_ra, beta_dec, zL, zS, lens_model_list, kwargs_lens_list,
    verbose=True, wlim=30.0, wmin=0.01,
    tlim_list=[ 0.001,10, 100,1000], dt_list=dt_list2)


plt.figure(figsize=(9,5))
plt.plot(w, np.abs(Fw),  label='HistMethod (fast)')
plt.plot(w, np.abs(Fwc), label='Geometric optics Fwc', ls='--')
#plt.plot(w,np.abs(F),label = "GWLens")
plt.xscale('log')
plt.xlabel('w')
plt.ylabel('|F(w)|')
plt.xlim(w[0], 500)
plt.legend()
plt.title('Point Mass Lens')
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('histogram_counting.png', dpi=300)
plt.show()
