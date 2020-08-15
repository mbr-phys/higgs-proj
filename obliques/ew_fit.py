#!/bin/env python3

import flavio
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from stu import *
import multiprocessing as mp
from multiprocessing import Pool
import ctypes
from functools import partial

#def shared_zeros(n):
#    ''' create a 3D numpy array which can be then changed in different threads '''
#    shared_array_base = mp.Array(ctypes.c_double,n**3)
#    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
#    shared_array = shared_array.reshape(n,n,n)
#    return shared_array

Sce, Tce, Uce = 0.02, 0.06, 0.00
Supe, Sloe = Sce+0.07,Sce-0.07
Tupe, Tloe = Tce+0.06,Tce-0.06
Uupe, Uloe = Uce+0.09,Uce-0.09

steps = 200

par = flavio.default_parameters.get_central_all()
err = flavio.default_parameters.get_1d_errors_random()

mHp,mH0,mA0 = np.linspace(0.1,1000,steps),np.linspace(0.1,1000,steps),np.linspace(0.1,1000,steps)
#p,h,a = np.meshgrid(mHp,mH0,mA0)
#pha = np.array([p,h,a]).reshape(3,steps**3).T
h,a = np.meshgrid(mH0,mA0)
pha = np.array([h,a]).reshape(2,steps**2).T

pool = Pool()
args = [par,err, Sce, Supe, Sloe, Tce, Tupe, Tloe, Uce, Uupe, Uloe]#, 500]
fitting = partial(fit,args)
ms = np.array(pool.map(fitting,pha)).reshape((steps,steps))#,steps))
pool.close()
pool.join()

mini = np.min(ms)
mss = np.zeros((steps,steps))

for i in range(steps):
    for j in range(steps):
        mss[i,j] = chis(ms[i,j],mini,5.99)

#pm = ms[:,0]
#hm = ms[:,0]
#am = ms[:,1]
#
##df = pd.DataFrame({'mH+':pm,'mH0':hm,'mA0':am}).dropna()
#df = pd.DataFrame({'mH0':hm,'mA0':am}).dropna()
##x,y,z = [df[c] for c in df]
#x,y = [df[c] for c in df]

fig = plt.figure()
s = fig.add_subplot(1,1,1,xlabel=r'$m_{H^0}$',ylabel=r'$m_{A^0}$')
im = s.imshow(mss,extent=(mH0[0],mH0[-1],mA0[0],mA0[-1]),origin='lower',cmap='Blues')
plt.title(r'$m_{H^0}-m_{A^0}$ cross-section for $m_{H^+}=500\,$GeV')
plt.show()
#fig = px.scatter_3d(x=x,y=y,z=z,opacity=0.3)
#fig.show()

#fig = plt.figure()
#ax = fig.add_subplot(111,projection='3d')
#ax.plot_trisurf(x,y,z,color='darkorchid',linewidth=0.2,antialiased=True)
#ax.set_xlabel(r'$\log_{10}[m_{H^+}\,$(GeV)$]$')
#ax.set_ylabel(r'$\log_{10}[m_{H^0}\,$(GeV)$]$')
#ax.set_zlabel(r'$\log_{10}[m_{A^0}\,$(GeV)$]$')
##plt.savefig('ewp.png')
#plt.show()
