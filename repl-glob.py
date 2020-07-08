#!/bin/env python3

import flavio
from flavio.statistics.likelihood import FastLikelihood
from flavio.classes import Parameter
import flavio.plots as fpl
import matplotlib.pyplot as plt
import numpy as np
from functions import *

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

flavio.measurements.read_file('world_avgs.yml') # read in the world averages we want to use

mH0 = 1500 # mass of heavy neutral Higgs, GeV

my_obs = [
    # what observables are we considering in the fit
#    'BR(B+->taunu)',
#    'BR(B+->munu)',
#    'BR(D+->munu)',
#    'BR(Ds->munu)',
#    'BR(Ds->taunu)',
#    'BR(tau->Knu)',
#    'BR(K+->munu)',
#    'BR(tau->pinu)',
#    'Gamma(pi+->munu)',
#    'BR(B->Xsgamma)',
#    'BR(Bs->mumu)',
#    'BR(B0->mumu)',
#    'Rtaul(B->Dlnu)',
#    'Rtaul(B->D*lnu)',
    'DeltaM_d',
    'DeltaM_s',
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

#V_vals = [0.5,0.75,0.9,1.0,1.1,1.25,1.5]
#
#for Vcb in V_vals:
def func(wcs):
    tanb, mH = wcs # state what the two parameters are going to be on the plot

    par = flavio.default_parameters.get_central_all()
    ckm_els = flavio.physics.ckm.get_ckm(par) # get out all the CKM elements

    # we want functions to generate the NP contributions to WCs
    # i.e. convert our parameters into operator language

#    CSR_b, CSL_b, CVL_b = rh(par['m_u'],par['m_b'],10**tanb,10**mH,Vub)
#    CSR_d, CSL_d, CVL_d = rh(par['m_c'],par['m_s'],10**tanb,10**mH,Vcd)
#    CSR_ds, CSL_ds, CVL_ds = rh(par['m_c'],par['m_s'],10**tanb,10**mH,Vcs)
#    CSR_k, CSL_k, CVL_k = rh(par['m_u'],par['m_s'],10**tanb,10**mH,Vus)
#    CSR_p, CSL_p, CVL_p = rh(par['m_u'],par['m_d'],10**tanb,10**mH,Vud)
#    C7_bs, C8_bs = bsgamma(par,10**tanb,10**mH)
#    C10_s, C10p_s, CS_s, CSp_s = bmumu(par,['m_s',1],ckm_els,mH0,10**tanb,10**mH)
#    C10_d, C10p_d, CS_d, CSp_d = bmumu(par,['m_d',0],ckm_els,mH0,10**tanb,10**mH)
#    CSL_tr, CSR_tr = rat_d(par,'m_tau',10**tanb,10**mH,Vcb)
#    CSL_mr, CSR_mr = rat_d(par,'m_mu',10**tanb,10**mH,Vcb)
#    CSL_er, CSR_er = rat_d(par,'m_e',10**tanb,10**mH,Vcb)
    CVLL_bs, CVRR_bs, CSLL_bs, CSRR_bs, CSLR_bs, CVLR_bs = mixing(par,ckm_els,['m_s',1,'m_d'],10**tanb,10**mH)
    CVLL_bd, CVRR_bd, CSLL_bd, CSRR_bd, CSLR_bd, CVLR_bd = mixing(par,ckm_els,['m_d',0,'m_s'],10**tanb,10**mH)
#    CVLL_bs = mixing2(par,abs(ckm_els[2,1]*np.conj(ckm_els[2,2]))**2,10**tanb,10**mH)
#    CVLL_bd = mixing2(par,abs(ckm_els[2,0]*np.conj(ckm_els[2,2]))**2,10**tanb,10**mH)

    wc = flavio.WilsonCoefficients()
    wc.set_initial({ # tell flavio what WCs you're referring to with your variables
#            'CSR_butaunutau': CSR_b, 'CSL_butaunutau': CSL_b, 'CVL_butaunutau': CVL_b, # B+->taunu
#            'CSR_bumunumu': CSR_b, 'CSL_bumunumu': CSL_b, 'CVL_bumunumu': CVL_b, # B+->munu
#            'CSR_dcmunumu': CSR_d, 'CSL_dcmunumu': CSL_d, 'CVL_dcmunumu': CVL_d, # D+->munu
#            'CSR_scmunumu': CSR_ds, 'CSL_scmunumu': CSL_ds, 'CVL_scmunumu': CVL_ds, # Ds->munu
#            'CSR_sctaunutau': CSR_ds, 'CSL_sctaunutau': CSL_ds, 'CVL_sctaunutau': CVL_ds,# Ds->taunu
#            'CSR_sumunumu': CSR_k, 'CSL_sumunumu': CSL_k, 'CVL_sumunumu': CVL_k,  # K+->munu
#            'CSR_sutaunutau': CSR_k, 'CSL_sutaunutau': CSL_k, 'CVL_sutaunutau': CVL_k, # tau->Knu
#            'CSR_dumunumu': CSR_p, 'CSL_dumunumu': CSL_p, 'CVL_dumunumu': CVL_p, # pi+->munu
#            'CSR_dutaunutau': CSR_p, 'CSL_dutaunutau': CSL_p, 'CVL_dutaunutau': CVL_p, # tau->pinu
#            'C7_bs': C7_bs,'C8_bs': C8_bs, # B->Xsgamma
#            'C10_bsmumu': C10_s,'C10p_bsmumu': C10p_s,'CS_bsmumu': CS_s,'CSp_bsmumu': CSp_s,'CP_bsmumu': CS_s,'CPp_bsmumu': CSp_s, # Bs->mumu
#            'C10_bdmumu': C10_d,'C10p_bdmumu': C10p_d,'CS_bdmumu': CS_d,'CSp_bdmumu': CSp_d,'CP_bdmumu': CS_d,'CPp_bdmumu': CSp_d, # B0->mumu
#            'CSR_bctaunutau': CSR_tr, 'CSL_bctaunutau': CSL_tr,'CSR_bcmunumu': CSR_mr, 'CSL_bcmunumu': CSL_mr,'CSR_bcenue': CSR_er, 'CSL_bcenue': CSL_er, # R(D) and R(D*)
            'CVLL_bsbs': CVLL_bs,'CVRR_bsbs': CVRR_bs,'CSLL_bsbs': CSLL_bs,'CSRR_bsbs': CSRR_bs,'CSLR_bsbs': CSLR_bs,'CVLR_bsbs': CVLR_bs, # DeltaM_s
            'CVLL_bdbd': CVLL_bd,'CVRR_bdbd': CVRR_bd,'CSLL_bdbd': CSLL_bd,'CSRR_bdbd': CSRR_bd,'CSLR_bdbd': CSLR_bd,'CVLR_bdbd': CVLR_bd, # DeltaM_d
        },
        scale=4.2, # mub I think, will almost always be the b-quark mass
        eft='WET', basis='flavio')
    return FL2.log_likelihood(par,wc)

sigmas = (1,2)
cdat = fpl.likelihood_contour_data(func,-1,2,1,3.5,
                n_sigma=sigmas,
                threads=4, # multiprocessing stuff, essentially makes it run faster if your computer can handle doing it
                steps=40) # increments for going round contours, i.e. smoothing out the plot (this has more obvious effects when just plotting WCs)

#bf,minh,mint,maxt = mHmin(cdat)
#print("Best fit value is found for (tanb,mH) =", bf)
#print("Print outs are lists for values at", sigmas, "sigmas")
#print("Minimum value of mH+ is:", minh)
#print("Minimum value of tanb is:", mint)
#print("Maximum value of tanb is:", maxt)

#fig_title = r"$\tau\to K\nu$ \& $K\to\mu\nu$ from " + str(Vus) + r"$\times$Vus"
#fig_title = r"$B\to l\nu$ from " + str(Vub) + r"$\times$Vub"
#fig_title = r"$R(D)$ \& $R(D^*)$ from " + str(Vcb) + r"$\times$Vcb"
#fig_name = "rds" + str(Vcb) + ".png"

plt.figure(figsize=(6,5))
fpl.contour(**cdat, # data
        interpolation_factor=3) # smoothing out the contour in another way than steps
#plt.title(fig_title)
plt.xlabel(r'$\log_{10}(\tan\beta)$') # log10
plt.ylabel(r'$\log_{10}(m_{H^+})$') #log10
#plt.savefig(fig_name)
plt.show()
