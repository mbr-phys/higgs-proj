#!/bin/env python3

import flavio
from flavio.statistics.likelihood import FastLikelihood
from flavio.classes import Parameter
import flavio.plots as fpl
import matplotlib.pyplot as plt
import numpy as np
from functions import *

# should at these to flavio's yml doc at some point, but this will do for now
pars = flavio.default_parameters

vev = Parameter('vev')
vev.tex = r"$v$"
vev.description = "Vacuum Expectation Value of the SM Higgs Mechanism"
pars.set_constraint('vev','246')

lam_QCD = Parameter('lam_QCD')
vev.tex = r"$\Lambda_{QCD}$"
vev.description = "QCD Lambda scale"
pars.set_constraint('lam_QCD','0.2275 + 0.01433 - 0.01372')

flavio.measurements.read_file('world_avgs.yml') # read in the world averages we want to use

mH0 = 1500 # mass of heavy neutral Higgs, GeV

my_obs = [
    # what observables are we considering in the fit
    'BR(B+->taunu)',
    'BR(B+->munu)',
    'BR(D+->munu)',
    'BR(Ds->munu)',
    'BR(Ds->taunu)',
#    'BR(B->Xsgamma)',
#    'BR(B0->mumu)',
#    'BR(Bs->mumu)',
]

FL2 = FastLikelihood(name="likelihood test",observables=my_obs,
        include_measurements=[
                # choosing to include only the measurements from world_avgs
                'Tree Level Leptonics',
                'Radiative Decays',
                'FCNC Leptonic Decays',
                'B Mixing',
                'LFU D Ratios',
                'Higgs Signal Strengths'
                ])
FL2.make_measurement(N=500,threads=4)

def func(wcs):
    tanb, mH = wcs # state what the two parameters are going to be on the plot

    par = flavio.default_parameters.get_central_all()
    ckm_els = flavio.physics.ckm.get_ckm(par) # get out all the CKM elements
#    ckm_els = ckm_func(par) # my func instead of flavio's

    # we want functions to generate the NP contributions to WCs
    # i.e. convert our parameters into operator language 

    CSR_b, CSL_b = rh(par['m_u'],par['m_b'],10**tanb,10**mH)
    CSR_d, CSL_d = rh(par['m_c'],par['m_s'],10**tanb,10**mH)
    CSR_ds, CSL_ds = rh(par['m_c'],par['m_s'],10**tanb,10**mH)
#    C7_bs, C8_bs = bsgamma(par['m_W'],par['m_t'],par['lam_QCD'],10**tanb,10**mH) 
#    C10_s, C10p_s, CS_s, CSp_s = bmumu(par['m_t'],par['m_mu'],par['m_W'],par['m_b'],par['m_s'],par['m_c'],par['m_u'],par['s2w'],par['m_h'],par['vev'],ckm_els,mH0,par['lam_QCD'],10**tanb,10**mH)
#    C10_d, C10p_d, CS_d, CSp_d = bmumu(par['m_t'],par['m_mu'],par['m_W'],par['m_b'],par['m_d'],par['m_c'],par['m_u'],par['s2w'],par['m_h'],par['vev'],ckm_els,mH0,par['lam_QCD'],10**tanb,10**mH)

    wc = flavio.WilsonCoefficients()
    wc.set_initial({ # tell flavio what WCs you're referring to with your variables
                    'CSR_butaunutau': CSR_b, 'CSL_butaunutau': CSL_b, # B+->taunu
                    'CSR_bumunumu': CSR_b, 'CSL_bumunumu': CSL_b, # B+->munu
                    'CSR_dcmunumu': CSR_d, 'CSL_dcmunumu': CSL_d, # D+->munu
                    'CSR_scmunumu': CSR_ds, 'CSL_scmunumu': CSL_ds, # Ds->munu
                    'CSR_sctaunutau': CSR_ds, 'CSL_sctaunutau': CSL_ds, # Ds->taunu
#                    'C7_bs': C7_bs,'C8_bs': C8_bs, # B->Xsgamma
#                    'C10_bsmumu': C10_s,'C10p_bsmumu': C10p_s,'CS_bsmumu': CS_s,'CSp_bsmumu': CSp_s,'CP_bsmumu': CS_s,'CPp_bsmumu': CSp_s, # Bs->mumu
#                    'C10_bdmumu': C10_d,'C10p_bdmumu': C10p_d,'CS_bdmumu': CS_d,'CSp_bdmumu': CSp_d,'CP_bdmumu': CS_d,'CPp_bdmumu': CSp_d, # B0->mumu
                    },
                    scale=4.18, # mub I think, will almost always be the b-quark mass
                    eft='WET', basis='flavio')
    return FL2.log_likelihood(par,wc)

sigmas = (1,2)
cdat = fpl.likelihood_contour_data(func,-1,2,1,3.5,
                n_sigma=sigmas,
                threads=4, # multiprocessing stuff, essentially makes it run faster if your computer can handle doing it
                steps=40) # increments for going round contours, i.e. smoothing out the plot (this has more obvious effects when just plotting WCs)

minh,mint,maxt = mHmin(cdat)
print("Print outs are lists for values at", sigmas, "sigmas")
print("Minimum value of mH+ is:", minh)
print("Minimum value of tanb is:", mint)
print("Maximum value of tanb is:", maxt)

plt.figure(figsize=(8,6))
fpl.contour(**cdat, # data
        interpolation_factor=3) # smoothing out the contour in another way than steps
#plt.title(r'$B\to X_s\gamma$')
#plt.title(r'$B_q\to\mu\mu$')
plt.title(r'$M\to l\nu$')
#plt.title(r'Combined $B_q\to\mu\mu$ and $B\to X_s\gamma$')
plt.xlabel(r'$\tan\beta$') # log10
plt.ylabel(r'$m_{H^+}$') #log10
#plt.savefig('bsgamma_plot.png')
#plt.savefig('bmumu_plot.png')
#plt.savefig('comb_plot.png')
plt.savefig('mlnu_plot.png')
#plt.show()
