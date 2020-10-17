#!/bin/env python3

import flavio
from flavio.statistics.likelihood import FastLikelihood
from flavio.physics.bdecays.formfactors import b_v, b_p
from flavio.classes import Parameter, AuxiliaryQuantity, Implementation
from flavio.statistics.functions import pvalue
from flavio.config import config
import multiprocessing as mp
from multiprocessing import Pool
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import flavio.plots as fpl
import matplotlib.pyplot as plt
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

# a_mu SM from 2006.04822
pars.set_constraint('a_mu SM','116591810(43)e-11')

#### fitting stuff

flavio.measurements.read_file('world_avgs.yml') # read in the world averages we want to use

#Fmuon = FastLikelihood(name="muons",observables=['a_mu'],include_measurements=['Anomalous Magnetic Moments'])
#Fmuon.make_measurement(N=500,threads=4)

#------------------------------
#   Anomalous moments
#------------------------------

par = flavio.default_parameters.get_central_all()
#err = flavio.default_parameters.get_1d_errors_random()

sig = 2
exp = flavio.combine_measurements('a_mu',include_measurements=['Anomalous Magnetic Moments'])
expc = exp.central_value
expr =  sig*exp.error_right                                                                                  
expl = sig*exp.error_left

def muon(wcs):
    tanb,mH = wcs
    
#    mH0,mA0 = mH, mH
#    mH0,mA0 = np.log10(1500), mH
    mH0,mA0 = mH, np.log10(1500)

    csev = a_mu(par,'m_mu',10**tanb,10**mH0,10**mA0,10**mH)

    wc = flavio.WilsonCoefficients()
    wc.set_initial({'C7_mumu': csev},scale=1.0,eft='WET-3',basis='flavio')
    
    npp = flavio.np_prediction('a_mu',wc_obj=wc)
    npe = sig*flavio.np_uncertainty('a_mu',wc_obj=wc)

#    expp = ((expc+expr)+(expc-expl))/2
#    expe = (expc+expr)-expp  
#    sigs = np.sqrt(npe**2 + expe**2)
#    chisq = ((npp-expp)/sigs)**2 

    if ((expc >= npp) and (npp+npe >= expc-expl)) or ((expc < npp) and (npp-npe <= expc+expr)):
        return 1
    else:
        return 0


#    return npp
#    return chisq/-2
#    return Fmuon.log_likelihood(par,wc)

#------------------------------
#   Get Contour Data
#------------------------------

sigmas = (1,2)
#sigmas = (3,4)

#cmuon = fpl.likelihood_contour_data(muon,-3,3,-3,2, n_sigma=sigmas, threads=4, steps=100) 
#cmuon = fpl.likelihood_contour_data(muon,0,4,-1,4, n_sigma=sigmas, threads=4, steps=100) 

#------------------------------
#   Plotting
#------------------------------

#print(muon([1,3.5]))
#quit()

steps = 200
tanb, mH = np.linspace(-1,4,steps),np.linspace(0,4,steps)
t,h = np.meshgrid(tanb,mH)
th = np.array([t,h]).reshape(2,steps**2).T

pool = Pool()
pred = np.array(pool.map(muon,th)).reshape((steps,steps))
pool.close()
pool.join()

#plt.figure(figsize=(6,5))
#fpl.contour(**cmuon,col=6)
##plt.title(r'$m_{A^0}\sim m_{H^+}$ and $m_{H^0} = 1500\,$GeV',fontsize=18)
##plt.title(r'$m_{H^0}\sim m_{H^+}$ and $m_{A^0} = 1500\,$GeV',fontsize=18)
#plt.title(r'$m_{H^0},m_{A^0}\sim m_{H^+}$',fontsize=18)
##plt.axhline(y=np.log10(1220),color='black',linestyle='--') # Asim = 866, Hsim = 1220
##plt.axhline(y=np.log10(1660),color='black',linestyle='--') # Asim = 1660, Hsim = 1660
#plt.xlabel(r'$\log_{10}[\tan\beta]$')
#plt.ylabel(r'$\log_{10}[m_{H^+} (\text{GeV})]$')
#plt.savefig('amu2HDM.png')
#quit()

fig = plt.figure()
s = fig.add_subplot(1,1,1,xlabel=r'$\log_{10}[\tan\beta]$',ylabel=r'$\log_{10}[m_{H^+} (\text{GeV})]$')
im = s.imshow(pred,extent=(tanb[0],tanb[-1],mH[0],mH[-1]),origin='lower',cmap='gray',vmin=0,vmax=1)
fig.colorbar(im)
plt.title(r'$a_\mu$ 2HDM contribution')
#plt.savefig('amu2HDM.png')
plt.show()
quit()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(t,h,pred)
ax.set_xlabel(r'$\log_{10}[\tan\beta]$')
ax.set_ylabel(r'$\log_{10}[m_{H^+} (\text{GeV})]$')
plt.show()
