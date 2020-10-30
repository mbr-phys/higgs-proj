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

# Bag parameters update from 1909.11087
pars.set_constraint('bag_B0_1','0.835 +- 0.028')
pars.set_constraint('bag_B0_2','0.791 +- 0.034')
pars.set_constraint('bag_B0_3','0.775 +- 0.054')
pars.set_constraint('bag_B0_4','1.063 +- 0.041')
pars.set_constraint('bag_B0_5','0.994 +- 0.037')
pars.set_constraint('eta_tt_B0','0.537856') # Alex, private correspondence

pars.set_constraint('bag_Bs_1','0.849 +- 0.023')
pars.set_constraint('bag_Bs_2','0.835 +- 0.032')
pars.set_constraint('bag_Bs_3','0.854 +- 0.051')
pars.set_constraint('bag_Bs_4','1.031 +- 0.035')
pars.set_constraint('bag_Bs_5','0.959 +- 0.031')
pars.set_constraint('eta_tt_Bs','0.537856') # Alex, private correspondence

# a_mu SM from 2006.04822
pars.set_constraint('a_mu SM','116591810(43)e-11')

# Updating meson lifetimes
pars.set_constraint('tau_B0','2307767355946.246 +- 6077070061.741268')
pars.set_constraint('tau_Bs','2301690285884.505 +- 6077070061.741268')

#### fitting stuff

flavio.measurements.read_file('world_avgs.yml') # read in the world averages we want to use
par = flavio.default_parameters.get_central_all()

obs2 = ['BR(B+->pilnu)', 'BR(B0->pilnu)']

bpp = flavio.functions.get_dependent_parameters_sm(obs2[0])
bop = flavio.functions.get_dependent_parameters_sm(obs2[1])

for b in bpp:
    print(b+': '+str(par[b]))

print(' ')
for o in bop:
    print(o+': '+str(par[o]))

quit()

strings = 'Bptopilnu-chisq1'
#Fleps = FastLikelihood(name="trees",observables=[obs2[0]],include_measurements=['Tree Level Semileptonics']) 
#Fleps.make_measurement(N=500,threads=4)

#------------------------------
#   Leptonic and Semileptonic Tree Levels
#------------------------------

sig = 1
exp = flavio.combine_measurements(obs2[0],include_measurements=['Tree Level Semileptonics'])
expc = exp.central_value 
expr = sig*exp.error_right
expl = sig*exp.error_left
expp = ((expc+expr)+(expc-expl))/2
expe = (expc+expr)-expp

def leps(wcs):
    tanb, mH = wcs # state what the two parameters are going to be on the plot

    par = flavio.default_parameters.get_central_all()
    ckm_els = flavio.physics.ckm.get_ckm(par) # get out all the CKM elements

#    CSR_b_t, CSL_b_t = rh(par['m_u'],par['m_b'],par['m_tau'],10**tanb,10**mH)
    CSR_b_m, CSL_b_m = rh(par['m_u'],par['m_b'],par['m_mu'],10**tanb,10**mH)
    CSR_b_e, CSL_b_e = rh(par['m_u'],par['m_b'],par['m_e'],10**tanb,10**mH)

    wc = flavio.WilsonCoefficients()
    wc.set_initial({ # tell flavio what WCs you're referring to with your variables
#            'CSR_butaunutau': CSR_b_t, 'CSL_butaunutau': CSL_b_t,
            'CSR_bumunumu': CSR_b_m, 'CSL_bumunumu': CSL_b_m,
            'CSR_buenue': CSR_b_e, 'CSL_buenue': CSL_b_e, 
        }, scale=4.2, eft='WET', basis='flavio')

    npp = flavio.np_prediction(obs2[0],wc_obj=wc)
    npe = sig*flavio.np_uncertainty(obs2[0],wc_obj=wc)

    sigs = np.sqrt(npe**2 + expe**2)
    chisq = ((npp-expp)/sigs)**2

    return chisq/-2

#    if ((expc >= npp) and (npp+npe >= expc-expl)) or ((expc < npp) and (npp-npe <= expc+expr)): 
#        return 1 
#    else:
#        return 0

#    return Fleps.log_likelihood(par,wc)

#------------------------------
#   Get Contour Data
#------------------------------

sigmas = (1,2)
cleps = fpl.likelihood_contour_data(leps,-1,2,0.5,4, n_sigma=sigmas, threads=4, steps=100) 

#------------------------------
#   Plotting
#------------------------------

#steps = 100
#tanb, mH = np.linspace(-1,2,steps),np.linspace(0,4,steps)
#t,h = np.meshgrid(tanb,mH)                                                                      
#th = np.array([t,h]).reshape(2,steps**2).T
#
#pool = Pool()
#pred = np.array(pool.map(leps,th)).reshape((steps,steps))
#pool.close()
#pool.join()
#
#fig = plt.figure()
#s = fig.add_subplot(1,1,1,xlabel=r'$\log_{10}[\tan\beta]$',ylabel=r'$\log_{10}[m_{H^+} (\text{GeV})]$')
#im = s.imshow(pred,extent=(tanb[0],tanb[-1],mH[0],mH[-1]),origin='lower',cmap='gray',vmin=0,vmax=1)
#fig.colorbar(im)
#plt.title(strings+' scan')
#plt.savefig(strings+'.png')
##plt.show()
#quit()

plt.figure(figsize=(6,5))
fpl.contour(**cleps,col=2) 
plt.title(strings+' chisq basic fit at 1sig errors')
plt.xlabel(r'$\log_{10}[\tan\beta]$') 
plt.ylabel(r'$\log_{10}[m_{H^+} (\text{GeV})]$') 
plt.savefig(strings+'.png')
quit()

#plt.show()
# colours : 0 - blue, 1 - orange, 2 - green, 3 - pink, 4 - purple, 5 - brown, 6 - bright pink, 7 - grey, 8 - yellow, 9 - cyan
