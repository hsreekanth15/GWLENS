from general import amplification
from mpmath import log,mp
import numpy as np
import time
import matplotlib.pyplot as plt

mp.dps =200

def psi(x):
    return abs(x)


w_values = np.linspace(0.1,10,50)
y_values = np.linspace(0.1,1,1)
start =  time.time()

results = amplification(w_values,y_values,psi,num_processes=8)

end= time.time()
print(end-start)

F = results[:,0,0]+ 1j*results[:,0,1]

plt.plot(w_values,np.abs(F))
plt.xscale("log")
plt.show()