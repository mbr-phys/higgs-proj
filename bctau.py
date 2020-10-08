#!/bin/env python3

import flavio
from flavio.statistics.likelihood import FastLikelihood
from flavio.physics.bdecays.formfactors import b_v, b_p
from flavio.classes import Parameter, AuxiliaryQuantity, Implementation
from flavio.statistics.functions import pvalue
from flavio.config import config
import multiprocessing as mp
from multiprocessing import Pool
import flavio.plots as fpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from functions import *

#### parameter setting

# should add these to flavio's yml doc at some point, but this will do for now
pars = flavio.default_parameters

vev = Parameter('vev')
vev.tex = r"$v$"
vev.description = "Vacuum Expectation Value of the SM Higgs"
pars.set_constraint('vev','246')

lam_QCD = Parameter('lam_QCD')
lam_QCD.tex = r"$\Lambda_{QCD}$"
lam_QCD.description = "QCD Lambda scale"
pars.set_constraint('lam_QCD','0.2275 + 0.01433 - 0.01372')

pars.set_constraint("tau_Bc",'774690000000 ± 13670999999')
#pars.set_constraint("f_Bc",'0.371 ± 0.017')

my_obs = ["BR(Bc->taunu)",]

#------------------------------
#   Leptonic and Semileptonic Tree Levels
#------------------------------

def leps(wcs):
    tanb, mH = wcs # state what the two parameters are going to be on the plot

    par = flavio.default_parameters.get_central_all()

    CSR_bc, CSL_bc = rh(par['m_c'],par['m_b'],par['m_tau'],10**tanb,10**mH)

    wc = flavio.WilsonCoefficients()
    wc.set_initial({ # tell flavio what WCs you're referring to with your variables
            'CSR_bctaunutau': CSR_bc, 'CSL_bctaunutau': CSL_bc,
        }, scale=4.2, eft='WET', basis='flavio')
    pred = flavio.np_prediction('BR(Bc->taunu)',wc_obj=wc)
    err = flavio.np_uncertainty('BR(Bc->taunu)',wc_obj=wc)
#    pred -= err
#    sm = flavio.sm_prediction('BR(Bc->taunu)')
    return pred*100,err*100

par = flavio.default_parameters.get_central_all()
err = flavio.default_parameters.get_1d_errors_random()
print("f_Bc:",par['f_Bc'],"+/-",err['f_Bc'])
print("tau_Bc:",par['tau_Bc']/1.519e12,"+/-",err['tau_Bc']/1.519e12)
print("SM prediction is:",flavio.sm_prediction('BR(Bc->taunu)')*100,"+/-",flavio.sm_uncertainty('BR(Bc->taunu)')*100)
one = leps((np.log10(2),np.log10(980)))
print("For tanb = 2 and mH+ = 980:",one[0],"+/-",one[1])
two = leps((np.log10(8),np.log10(1500)))
print("For tanb = 8 and mH+ = 1500:",two[0],"+/-",two[1])
three = leps((np.log10(2),np.log10(3200)))
print("For tanb = 2 and mH+ = 3200:",three[0],"+/-",three[1])
four = leps((np.log10(0.5),np.log10(3200)))
print("For tanb = 0.5 and mH+ = 3200:",four[0],"+/-",four[1])
quit()

steps = 60
tanb, mH = np.linspace(-1,1,steps),np.linspace(2.5,3.5,steps)
t,h = np.meshgrid(tanb,mH)
th = np.array([t,h]).reshape(2,steps**2).T

pool = Pool()
pred = np.array(pool.map(leps,th)).reshape((steps,steps))
pool.close()
pool.join()

#------------------------------
#   Plotting
#------------------------------

fig = plt.figure()
s = fig.add_subplot(1,1,1,xlabel=r'$\log_{10}[\tan\beta]$',ylabel=r'$\log_{10}[m_{H^+} (\text{GeV})]$')
im = s.imshow(pred,extent=(tanb[0],tanb[-1],mH[0],mH[-1]),origin='lower')
fig.colorbar(im)
plt.title(r'$\mathcal{B}r[B_c\to\tau\nu_\tau]\times10^{-2}$ Central Value')# \frac{\text{2HDM}}{\text{SM}}$')#, for $f_{B_c} = 434\pm15\,$MeV')
plt.savefig('bctaunu.png')
#plt.show()

#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.plot_surface(t,h,pred)
#ax.set_xlabel(r'$\log_{10}[\tan\beta]$')
#ax.set_ylabel(r'$\log_{10}[m_{H^+} (\text{GeV})]$')
#ax.set_zlabel(r'$\mathcal{B}r[B_c\to\tau\nu_\tau]$ 2HDM/SM')
#plt.show()

#def test(wcs):
#    t, m = wcs
#    tanb = 10**t
#    mH = 10**m
#
#    par = flavio.default_parameters.get_central_all()
#
#    rh = ((par['m_c']-par['m_b']*tanb**2)/(par['m_c']+par['m_b']))*(par['m_Bc']/mH)**2
#
#    return (1+rh)**2
