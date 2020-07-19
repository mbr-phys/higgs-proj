import os
import numpy as np
import flavio
import matplotlib.pyplot as plt
from extra import *

tanb = np.linspace(-1,2,300)
mH = np.linspace(1,3.5,250)
#tanbs = [np.linspace(-1,2,300),np.linspace(-1,1,200)]
#mHs = [np.linspace(1,3.5,250),np.linspace(2,3.5,150)]

par = flavio.default_parameters.get_central_all()
err = flavio.default_parameters.get_1d_errors_random()
#ckm_els = flavio.physics.ckm.get_ckm(par)
ckm_els, ckm_errs = ckm_err(par,err)
mus = ['m_u','m_c','m_c','m_u','m_u']
mds = ['m_b','m_d','m_s','m_s','m_d']
mms = ['m_B+','m_D+','m_Ds','m_K+','m_pi+']
lats = ['$m_{B^+}$','$m_{D^+}$','$m_{D_s}$','$m_{K^+}$',r'$m_{\pi^+}$']
pngs = ['mB','mD','mDs','mK','mpi']
heatmap = {}
errmap = {}

for m in range(len(mus)):
#    for x in range(2):
    heatmap_i = np.empty((mH.size,tanb.size))
    heatmap_e = np.empty((mH.size,tanb.size))
    for i in range(mH.size):
        for j in range(tanb.size):
            heatmap_i[i,j] = rh(par[mus[m]],par[mds[m]],par[mms[m]],10**tanb[j],10**mH[i])
            heatmap_e[i,j] =  errors1(par,err,mus,mds,mms,tanb,mH,i,j,m,heatmap_i[i,j])

#        fig_title = r"$\frac{|\tilde{V}_{ij}|}{|V_{ij}|}$ for " + lats[m]
#        fig_name = pngs[m] + "-" + "rH" + str(x) + ".png"
#        print("Range for "+fig_title+" is "+str(np.min(heatmap_i))+" to " +str(np.max(heatmap_i)))
#        if x == 0:
    heatmap[pngs[m]] = heatmap_i
    errmap[pngs[m]] = heatmap_e

#        fig = plt.figure()
#        s = fig.add_subplot(1,1,1,xlabel=r"$\log_{10}[\tan\beta]$",ylabel=r"$\log_{10}[m_{H^+}]$")
#        im = s.imshow(heatmap_i,extent=(tanb[0],tanb[-1],mH[0],mH[-1]),origin='lower')
#        fig.colorbar(im)
#        plt.title(fig_title)
#        plt.savefig(fig_name)
#        plt.show()

units = np.zeros((mH.size,tanb.size))
for i in range(mH.size):
    for j in range(tanb.size):
        r1, r2 = vp(ckm_els,heatmap,i,j)
        re1, re2 = errors2(ckm_els,ckm_errs,heatmap,errmap,i,j)
#        re1, re2 = 0,0
        if r1 < 1 and r2 < 1:
            units[i,j] = 1
        elif (r1 + re1) < 1 and r2 < 1:
            units[i,j] = 1
        elif (r1 - re1) < 1 and r2 < 1:
            units[i,j] = 1
        elif r1 < 1 and (r2 + re2) < 1:
            units[i,j] = 1
        elif r1 < 1 and (r2 - re2) < 1:
            units[i,j] = 1
        elif (r1 + re1) < 1 and (r2 + re2) < 1:
            units[i,j] = 1
        elif (r1 + re1) < 1 and (r2 - re2) < 1:
            units[i,j] = 1
        elif (r1 - re1) < 1 and (r2 + re2) < 1:
            units[i,j] = 1
        elif (r1 - re1) < 1 and (r2 - re2) < 1:
            units[i,j] = 1
#       else:
#           print("Row 1:",r1,"plus/minus",re1)
#           print("Row 2:",r2,"plus/minus",re2)

fig = plt.figure()
s = fig.add_subplot(1,1,1,xlabel=r"$\log_{10}[\tan\beta]$",ylabel=r"$\log_{10}[m_{H^+}]$")
im = s.imshow(units,extent=(tanb[0],tanb[-1],mH[0],mH[-1]),origin='lower',cmap='gray')
#fig.colorbar(im)
plt.title("Modification Regions Allowed By Unitarity using CKM first two rows")
plt.savefig("ckm_mod_2sig.png")
#plt.show()