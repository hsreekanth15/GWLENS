import numpy as np
from scipy.special import jv
from scipy.integrate import quad
from utilities import ensure_1d_array
import warnings
from functools import lru_cache

warnings.simplefilter('ignore')

eps = 1.e-3  # Relative accuracy of F(w, y)

@lru_cache(maxsize=None)
def psi(x, rs):
    if x <= 1:
        return rs/2 * ((np.log(x/2))**2 - 4 * np.arctanh(np.sqrt((1 - x) / (1 + x)))**2)
    else:
        return rs/2 * ((np.log(x/2))**2 + 4 * np.arctan(np.sqrt((x - 1) / (1 + x)))**2)

def func(x, w, y, rs):
    return jv(0, w * y * np.sqrt(2 * x)) * np.exp(-1j * w * psi(np.sqrt(2 * x), rs))

def func2(x, w, y, rs):
    return func(x, w, y, rs) * np.exp(1j * w * x)

@lru_cache(maxsize=None)
def dfunc(x, w, y, rs):
    return (-w * y / np.sqrt(2 * x) * jv(1, w * y * np.sqrt(2 * x)) * np.exp(-1j * w * psi(np.sqrt(2 * x), rs)) -
            1j * w / (2 * x) * func(x, w, y, rs))

@lru_cache(maxsize=None)
def ddfunc(x, w, y, rs):
    return (w * y / (2 * x * np.sqrt(2 * x)) * (2 + 1j * w) * jv(1, w * y * np.sqrt(2 * x)) *
            np.exp(-1j * w * psi(np.sqrt(2 * x), rs)) - 1 / (2 * x) * (w * w * y * y - 1j * w / x) *
            func(x, w, y, rs) - 1j * w / (2 * x) * dfunc(x, w, y, rs))

def NFW(w, y, rs):
    a = 0.
    b = 1000. / w
    zzp = -1.
    while True:
        zz = quad(func2, a, b, args=(w, y, rs), limit=int(1e+7), epsrel=eps/3., complex_func=True)[0]
        zz += (-func(b, w, y, rs) / (1j * w) * np.exp(1j * w * b) -
               dfunc(b, w, y, rs) / (w * w) * np.exp(1j * w * b) +
               ddfunc(b, w, y, rs) / (1j * w * w * w) * np.exp(1j * w * b))
        if np.abs(zz / zzp - 1) < eps:
            break
        zzp = zz
        b = min(b * 10, 10**5 / w)  # Limit b to a maximum to avoid excessive ranges
    return -1j * w * np.exp(0.5 * 1j * w * y * y) * zz


''' 
def NFW(w,y,rs):
    Freal = []
    Fimag = []
    for i in range(len(w)):
        for j in range(len(y)):
            F = ampf(w[i],y[j],rs)
            Freal.append(F.real)
            Fimag.append(F.imag) 
            print(F)

    return Freal,Fimag'
    '''