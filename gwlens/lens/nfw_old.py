
import numpy as np
from scipy.special import jv
from scipy.integrate import quad
import warnings

#from functools import lru_cache

warnings.simplefilter('ignore')

eps = 1.e-4  # Relative accuracy of F(w, y)

def psi(x, rs):
    if x == 0:
        return 0
    elif x < 1:
        return rs/2 * ((np.log(x/2))**2 - np.arctanh(np.sqrt(1 - x**2))**2)
    else:
        return rs/2 * ((np.log(x/2))**2 +  np.arctan(np.sqrt(x**2 - 1))**2)

# def psi(x, rs):
#     if x == 0:
#         return 0
#     elif x<=1
#         return rs/2 * ((np.log(x/2))**2 - 4 * np.arctanh(np.sqrt((1 - x) / (1 + x)))**2)
#     else:
#         return rs/2 * ((np.log(x/2))**2 + 4 * np.arctan(np.sqrt((x - 1) / (1 + x)))**2)

def func(x, w, y, rs):
    return jv(0, w * y * np.sqrt(2 * x)) * np.exp(-1j * w * psi(np.sqrt(2 * x), rs))

def func2(x, w, y, rs):
    return func(x, w, y, rs) * np.exp(1j * w * x)


def dfunc(x, w, y, rs):
    return (-w * y / np.sqrt(2 * x) * jv(1, w * y * np.sqrt(2 * x)) * np.exp(-1j * w * psi(np.sqrt(2 * x), rs)) -
            1j * w / (2 * x) * func(x, w, y, rs))


def ddfunc(x, w, y, rs):
    return (w * y / (2 * x * np.sqrt(2 * x)) * (2 + 1j * w) * jv(1, w * y * np.sqrt(2 * x)) *
            np.exp(-1j * w * psi(np.sqrt(2 * x), rs)) - 1 / (2 * x) * (w * w * y * y - 1j * w / x) *
            func(x, w, y, rs) - 1j * w / (2 * x) * dfunc(x, w, y, rs))

def NFW(w, y, rs):
    a = 0.0001
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


# def NFW(w, y, rs):
#     a = 0.
#     b = 1000
#     zzp = -1.
#     while True:
#         zz = quad(func2, a, b, args=(w, y, rs), limit=int(1e+5), epsrel=eps/3., complex_func=True)[0]
#         zz += (-func(b, w, y, rs) / (1j * w) * np.exp(1j * w * b) -
#                dfunc(b, w, y, rs) / (w * w) * np.exp(1j * w * b) +
#                ddfunc(b, w, y, rs) / (1j * w * w * w) * np.exp(1j * w * b))
#         if np.abs(zz / zzp - 1) < eps:
#             break
#         zzp = zz
        
#     return -1j * w * np.exp(0.5 * 1j * w * y * y) * zz