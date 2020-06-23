#!/bin/env python3

import flavio
from flavio.statistics.likelihood import FastLikelihood
import flavio.plots as fpl
import matplotlib.pyplot as plt
import numpy as np
from functions import *

flavio.measurements.read_file('world_avgs.yml') # read in the world averages we want to use

# i think both vev and lambda must be able to be pulled out of the flavio parameters somehow but I'm not sure how - either that or just add them as new parameters
vev = 246
lam_QCD, lam_err = [0.2275,[0.01433,-0.01372]] # lambda_QCD
mH0 = 1500 # mass of heavy neutral Higgs, GeV

my_obs = [
    # what observables are we considering in the fit
#    'BR(B->Xsgamma)',
    'BR(B0->mumu)',
    'BR(Bs->mumu)',
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

    # function to get out all the CKM element magnitudes, index output 0-8: increasing number goes first across the rows, then down the columns
    ckm_els = ckm_func(par['Vub'],par['Vcb'],par['Vus'],par['delta']) 

    # we want functions to generate the NP contributions to WCs

#    C7_bs, C8_bs = bsgamma(par['m_W'],par['m_t'],lam_QCD,10**mH,10**tanb) # convert parameters into operator language -
    C10_mu, C10p_mu, CS_mu, CSp_mu = bmumu(par['m_t'],ckm_els[8],ckm_els[7],par['m_mu'],par['m_W'],par['m_b'],par['m_s'],par['m_c'],par['m_u'],par['s2w'],par['m_h'],vev,par['Vus'],par['Vub'],ckm_els[4],par['Vcb'],mH0,lam_QCD,10**tanb,10**mH)
    C10_mu, C10p_mu, CS_mu, CSp_mu = bmumu(par['m_t'],ckm_els[8],ckm_els[6],par['m_mu'],par['m_W'],par['m_b'],par['m_d'],par['m_c'],par['m_u'],par['s2w'],par['m_h'],vev,ckm_els[0],par['Vub'],ckm_els[3],par['Vcb'],mH0,lam_QCD,10**tanb,10**mH)

    wc = flavio.WilsonCoefficients()
    wc.set_initial({ # tell flavio what WCs you're referring to with your variables
#                    'C7_bs': C7_bs,
#                    'C8_bs': C8_bs,
                    'C10_bsmumu': C10_mu,
                    'C10p_bsmumu': C10p_mu,
                    'CS_bsmumu': CS_mu,
                    'CSp_bsmumu': CSp_mu,
                    'CP_bsmumu': CS_mu,
                    'CPp_bsmumu': CSp_mu,
                    'C10_bdmumu': C10_mu,
                    'C10p_bdmumu': C10p_mu,
                    'CS_bdmumu': CS_mu,
                    'CSp_bdmumu': CSp_mu,
                    'CP_bdmumu': CS_mu,
                    'CPp_bdmumu': CSp_mu,
                    },
                    scale=4.18, # mub I think, will almost always be the b-quark mass
                    eft='WET', basis='flavio')
    return FL2.log_likelihood(par,wc)

cdat = fpl.likelihood_contour_data(func,-1,2,1,3.5,
                n_sigma=(1,2),
                threads=4, # multiprocessing stuff, essentially makes it run faster if your computer can handle doing it
                steps=40) # increments for going round contours, i.e. smoothing out the plot (this has more obvious effects when just plotting WCs)

minh,mint,maxt = mHmin(cdat)
print("Minimum value of mH+ is: ", minh)
print("Minimum value of tanb is: ", mint)
print("Maximum value of tanb is: ", maxt)

plt.figure(figsize=(8,6))
fpl.contour(**cdat, # data
        interpolation_factor=3) # smoothing out the contour in another way than steps
#plt.title(r'$B\to X_s\gamma$')
plt.title(r'$B_q\to\mu\mu$')
#plt.title(r'Combined $B_q\to\mu\mu$ and $B\to X_s\gamma$')
plt.xlabel(r'$\tan\beta$') # log10
plt.ylabel(r'$m_{H^+}$') #log10
#plt.savefig('bsgamma_plot.png')
plt.savefig('bmumu_plot.png')
#plt.savefig('comb_plot.png')
#plt.show()
