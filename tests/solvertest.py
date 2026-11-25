# A simple quick test to see if everything works

from gwlens import solver1d,solver2d
ws = 0.1
y = 0.1

#SIS
sis = solver1d.sis()
F_sis = solver1d.amplification(ws,y,sis)
print("SIS-lens",abs(F_sis))

#NFW
nfw = solver1d.nfw()
F_nfw = solver1d.amplification(ws,y,nfw)
print("NFW-lens",abs(F_nfw))