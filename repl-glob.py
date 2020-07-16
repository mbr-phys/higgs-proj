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

#flavio.config['renormalization scale']['bxgamma'] = 1.8045
flavio.config['renormalization scale']['bxgamma'] = 1.74

mH0 = 1500 # mass of heavy neutral Higgs, GeV

my_obs = [
    'BR(B+->taunu)', 'BR(B+->munu)', 'BR(D+->munu)', 'BR(Ds->munu)', 'BR(Ds->taunu)', 'BR(tau->Knu)', 'BR(K+->munu)', 'BR(tau->pinu)', 'Gamma(pi+->munu)', # [:9]
    'BR(B->Xsgamma)', # [9]
    'BR(Bs->mumu)','BR(B0->mumu)', # [10:12]
    'DeltaM_d','DeltaM_s', # [12:14]
    'Rtaul(B->Dlnu)', 'Rtaul(B->D*lnu)', # [14:16]
    'BR(B0->Dlnu)', 'BR(B+->Dlnu)', 'BR(B0->D*lnu)', 'BR(B+->D*lnu)', 'BR(B+->pilnu)', 'BR(B0->pilnu)', 'BR(B+->rholnu)', 'BR(B0->rholnu)', 'BR(D+->Kenu)', 'BR(D+->Kmunu)', 'BR(D+->pienu)', 'BR(D+->pimunu)', 'BR(D0->Kenu)', 'BR(D0->Kmunu)', 'BR(D0->pienu)', 'BR(D0->pimunu)', 'BR(K+->pienu)', 'BR(K+->pimunu)', 'BR(KL->pienu)', 'BR(KL->pimunu)', 'BR(KS->pienu)', 'BR(KS->pimunu)', # [16:]
]

sigmas = (1,2)

Fleps = FastLikelihood(name="trees",observables=my_obs[:9]+my_obs[14:],include_measurements=['Tree Level Leptonics','LFU D Ratios','Tree Level Semileptonics']) 
Fleps.make_measurement(N=500,threads=4)

def leps(wcs):
    tanb, mH = wcs # state what the two parameters are going to be on the plot

    par = flavio.default_parameters.get_central_all()
    ckm_els = flavio.physics.ckm.get_ckm(par) # get out all the CKM elements

    CSR_b, CSL_b = rh(par['m_u'],par['m_b'],10**tanb,10**mH)
    CSR_d, CSL_d = rh(par['m_c'],par['m_s'],10**tanb,10**mH)
    CSR_ds, CSL_ds = rh(par['m_c'],par['m_s'],10**tanb,10**mH)
    CSR_k, CSL_k = rh(par['m_u'],par['m_s'],10**tanb,10**mH)
    CSR_p, CSL_p = rh(par['m_u'],par['m_d'],10**tanb,10**mH)
    CSL_bc, CSR_bc = rh(par['m_c'],par['m_b'],10**tanb,10**mH)

    wc = flavio.WilsonCoefficients()
    wc.set_initial({ # tell flavio what WCs you're referring to with your variables
            'CSR_bctaunutau': CSR_bc, 'CSL_bctaunutau': CSL_bc,
            'CSR_bcmunumu': CSR_bc, 'CSL_bcmunumu': CSL_bc,
            'CSR_bcenue': CSR_bc, 'CSL_bcenue': CSL_bc, 
            'CSR_butaunutau': CSR_b, 'CSL_butaunutau': CSL_b,
            'CSR_bumunumu': CSR_b, 'CSL_bumunumu': CSL_b,
            'CSR_buenue': CSR_b, 'CSL_buenue': CSL_b, 
            'CSR_dcmunumu': CSR_d, 'CSL_dcmunumu': CSL_d,
            'CSR_dcenue': CSR_d, 'CSL_dcenue': CSL_d, 
            'CSR_sctaunutau': CSR_ds, 'CSL_sctaunutau': CSL_ds,
            'CSR_scmunumu': CSR_ds, 'CSL_scmunumu': CSL_ds,
            'CSR_scenue': CSR_ds, 'CSL_scenue': CSL_ds, 
            'CSR_sutaunutau': CSR_k, 'CSL_sutaunutau': CSL_k, 
            'CSR_sumunumu': CSR_k, 'CSL_sumunumu': CSL_k, 
            'CSR_suenue': CSR_k, 'CSL_suenue': CSL_k, 
            'CSR_dutaunutau': CSR_p, 'CSL_dutaunutau': CSL_p, 
            'CSR_dumunumu': CSR_p, 'CSL_dumunumu': CSL_p, 
        }, scale=4.2, eft='WET', basis='flavio')
    return Fleps.log_likelihood(par,wc)

cleps = fpl.likelihood_contour_data(leps,-1,2,1,3.5, n_sigma=sigmas, threads=4, steps=40) 

Fmix = FastLikelihood(name="mix",observables=my_obs[12:14],include_measurements=['B Mixing',]) 
Fmix.make_measurement(N=500,threads=4)

def mix(wcs):
    tanb, mH = wcs # state what the two parameters are going to be on the plot

    par = flavio.default_parameters.get_central_all()
    ckm_els = flavio.physics.ckm.get_ckm(par) # get out all the CKM elements

    CVLL_bs, CVRR_bs, CSLL_bs, CSRR_bs, CSLR_bs, CVLR_bs = mixing(par,ckm_els,['m_s',1,'m_d'],10**tanb,10**mH)
    CVLL_bd, CVRR_bd, CSLL_bd, CSRR_bd, CSLR_bd, CVLR_bd = mixing(par,ckm_els,['m_d',0,'m_s'],10**tanb,10**mH)
#    CVLL_bs = mixing2(par,abs(ckm_els[2,1]*np.conj(ckm_els[2,2]))**2,10**tanb,10**mH)
#    CVLL_bd = mixing2(par,abs(ckm_els[2,0]*np.conj(ckm_els[2,2]))**2,10**tanb,10**mH)

    wc = flavio.WilsonCoefficients()
    wc.set_initial({ # tell flavio what WCs you're referring to with your variables
            'CVLL_bsbs': CVLL_bs,'CVRR_bsbs': CVRR_bs,'CSLL_bsbs': CSLL_bs,'CSRR_bsbs': CSRR_bs,'CSLR_bsbs': CSLR_bs,'CVLR_bsbs': CVLR_bs, # DeltaM_s
            'CVLL_bdbd': CVLL_bd,'CVRR_bdbd': CVRR_bd,'CSLL_bdbd': CSLL_bd,'CSRR_bdbd': CSRR_bd,'CSLR_bdbd': CSLR_bd,'CVLR_bdbd': CVLR_bd, # DeltaM_d
        }, scale=4.2, eft='WET', basis='flavio')
    return Fmix.log_likelihood(par,wc)

cmix = fpl.likelihood_contour_data(mix,-1,2,1,3.5, n_sigma=sigmas, threads=4, steps=40) 

Frad = FastLikelihood(name="rad",observables=['BR(B->Xsgamma)'],include_measurements=['Radiative Decays']) 
Frad.make_measurement(N=500,threads=4)

def rad(wcs):
    tanb, mH = wcs # state what the two parameters are going to be on the plot

    par = flavio.default_parameters.get_central_all()
    ckm_els = flavio.physics.ckm.get_ckm(par) # get out all the CKM elements

    C7_bs, C8_bs = bsgamma(par,10**tanb,10**mH)

    wc = flavio.WilsonCoefficients()
    wc.set_initial({ # tell flavio what WCs you're referring to with your variables
            'C7_bs': C7_bs,'C8_bs': C8_bs, # B->Xsgamma
        }, scale=4.2, eft='WET', basis='flavio')
    return Frad.log_likelihood(par,wc)

crad = fpl.likelihood_contour_data(rad,-1,2,1,3.5, n_sigma=sigmas, threads=4, steps=40) 

Fmu = FastLikelihood(name="mu",observables=my_obs[10:12],include_measurements=['FCNC Leptonic Decays',]) # choosing to include only the measurements from world_avgs
Fmu.make_measurement(N=500,threads=4)

def mu(wcs):
    tanb, mH = wcs # state what the two parameters are going to be on the plot

    par = flavio.default_parameters.get_central_all()
    ckm_els = flavio.physics.ckm.get_ckm(par) # get out all the CKM elements

    C10_s, C10p_s, CS_s, CSp_s = bmumu(par,['m_s',1],ckm_els,mH0,10**tanb,10**mH)
    C10_d, C10p_d, CS_d, CSp_d = bmumu(par,['m_d',0],ckm_els,mH0,10**tanb,10**mH)

    wc = flavio.WilsonCoefficients()
    wc.set_initial({ # tell flavio what WCs you're referring to with your variables
            'C10_bsmumu': C10_s,'C10p_bsmumu': C10p_s,'CS_bsmumu': CS_s,'CSp_bsmumu': CSp_s,'CP_bsmumu': CS_s,'CPp_bsmumu': CSp_s, # Bs->mumu
            'C10_bdmumu': C10_d,'C10p_bdmumu': C10p_d,'CS_bdmumu': CS_d,'CSp_bdmumu': CSp_d,'CP_bdmumu': CS_d,'CPp_bdmumu': CSp_d, # B0->mumu
        }, scale=4.2, eft='WET', basis='flavio')
    return Fmu.log_likelihood(par,wc)

cmu = fpl.likelihood_contour_data(mu,-1,2,1,3.5, n_sigma=sigmas, threads=4, steps=40) 

FL2 = FastLikelihood(name="likelihood test",observables=my_obs,include_measurements=['Tree Level Leptonics','Radiative Decays','FCNC Leptonic Decays','B Mixing','LFU D Ratios','Tree Level Semileptonics'])
FL2.make_measurement(N=500,threads=4)

def func(wcs):
    tanb, mH = wcs # state what the two parameters are going to be on the plot

    par = flavio.default_parameters.get_central_all()
    ckm_els = flavio.physics.ckm.get_ckm(par) # get out all the CKM elements

    CSR_b, CSL_b = rh(par['m_u'],par['m_b'],10**tanb,10**mH)
    CSR_d, CSL_d = rh(par['m_c'],par['m_s'],10**tanb,10**mH)
    CSR_ds, CSL_ds = rh(par['m_c'],par['m_s'],10**tanb,10**mH)
    CSR_k, CSL_k = rh(par['m_u'],par['m_s'],10**tanb,10**mH)
    CSR_p, CSL_p = rh(par['m_u'],par['m_d'],10**tanb,10**mH)
    CSL_bc, CSR_bc = rh(par['m_c'],par['m_b'],10**tanb,10**mH)
    C7_bs, C8_bs = bsgamma(par,10**tanb,10**mH)
    C10_s, C10p_s, CS_s, CSp_s = bmumu(par,['m_s',1],ckm_els,mH0,10**tanb,10**mH)
    C10_d, C10p_d, CS_d, CSp_d = bmumu(par,['m_d',0],ckm_els,mH0,10**tanb,10**mH)
#    CVLL_bs, CVRR_bs, CSLL_bs, CSRR_bs, CSLR_bs, CVLR_bs = mixing(par,ckm_els,['m_s',1,'m_d'],10**tanb,10**mH)
#    CVLL_bd, CVRR_bd, CSLL_bd, CSRR_bd, CSLR_bd, CVLR_bd = mixing(par,ckm_els,['m_d',0,'m_s'],10**tanb,10**mH)
    CVLL_bs = mixing2(par,abs(ckm_els[2,1]*np.conj(ckm_els[2,2]))**2,10**tanb,10**mH)
    CVLL_bd = mixing2(par,abs(ckm_els[2,0]*np.conj(ckm_els[2,2]))**2,10**tanb,10**mH)

    wc = flavio.WilsonCoefficients()
    wc.set_initial({ # tell flavio what WCs you're referring to with your variables
            'CSR_bctaunutau': CSR_bc, 'CSL_bctaunutau': CSL_bc,
            'CSR_bcmunumu': CSR_bc, 'CSL_bcmunumu': CSL_bc,
            'CSR_bcenue': CSR_bc, 'CSL_bcenue': CSL_bc, 
            'CSR_butaunutau': CSR_b, 'CSL_butaunutau': CSL_b,
            'CSR_bumunumu': CSR_b, 'CSL_bumunumu': CSL_b,
            'CSR_buenue': CSR_b, 'CSL_buenue': CSL_b, 
            'CSR_dcmunumu': CSR_d, 'CSL_dcmunumu': CSL_d,
            'CSR_dcenue': CSR_d, 'CSL_dcenue': CSL_d, 
            'CSR_sctaunutau': CSR_ds, 'CSL_sctaunutau': CSL_ds,
            'CSR_scmunumu': CSR_ds, 'CSL_scmunumu': CSL_ds,
            'CSR_scenue': CSR_ds, 'CSL_scenue': CSL_ds, 
            'CSR_sutaunutau': CSR_k, 'CSL_sutaunutau': CSL_k, 
            'CSR_sumunumu': CSR_k, 'CSL_sumunumu': CSL_k, 
            'CSR_suenue': CSR_k, 'CSL_suenue': CSL_k, 
            'CSR_dutaunutau': CSR_p, 'CSL_dutaunutau': CSL_p, 
            'CSR_dumunumu': CSR_p, 'CSL_dumunumu': CSL_p, 
            'C7_bs': C7_bs,'C8_bs': C8_bs, # B->Xsgamma
            'C10_bsmumu': C10_s,'C10p_bsmumu': C10p_s,'CS_bsmumu': CS_s,'CSp_bsmumu': CSp_s,'CP_bsmumu': CS_s,'CPp_bsmumu': CSp_s, # Bs->mumu
            'C10_bdmumu': C10_d,'C10p_bdmumu': C10p_d,'CS_bdmumu': CS_d,'CSp_bdmumu': CSp_d,'CP_bdmumu': CS_d,'CPp_bdmumu': CSp_d, # B0->mumu
            'CVLL_bsbs': CVLL_bs,#'CVRR_bsbs': CVRR_bs,'CSLL_bsbs': CSLL_bs,'CSRR_bsbs': CSRR_bs,'CSLR_bsbs': CSLR_bs,'CVLR_bsbs': CVLR_bs, # DeltaM_s
            'CVLL_bdbd': CVLL_bd,#'CVRR_bdbd': CVRR_bd,'CSLL_bdbd': CSLL_bd,'CSRR_bdbd': CSRR_bd,'CSLR_bdbd': CSLR_bd,'CVLR_bdbd': CVLR_bd, # DeltaM_d
        }, scale=4.2, eft='WET', basis='flavio')
    return FL2.log_likelihood(par,wc)

cdat = fpl.likelihood_contour_data(func,-1,2,1,3.5, n_sigma=sigmas, threads=4, steps=40) 

#------------------------------
#   Print Out Values
#------------------------------

bf,minh,mint,maxt = mHmin(cdat)
print("Best fit value is found for (tanb,mH) =", bf)
print("Print outs are lists for values at", sigmas, "sigmas")
print("Minimum value of mH+ is:", minh)
print("Minimum value of tanb is:", mint)
print("Maximum value of tanb is:", maxt)

#------------------------------
#   Plotting
#------------------------------

plt.figure(figsize=(6,5))
fpl.contour(**cleps,interpolation_factor=3,col=2) 
plt.title('Tree Level Leptonic and Semileptonics and Hadronic Tau Decays')
plt.xlabel(r'$\log_{10}(\tan\beta)$') # log10
plt.ylabel(r'$\log_{10}(m_{H^+})$') #log10
plt.savefig('qqlnu_plot.png')

plt.figure(figsize=(6,5))
fpl.contour(**cmix,interpolation_factor=3,col=0) 
#plt.title(r'$\Delta M_{d,s}$ from Approximated Expressions')
plt.title(r'$\Delta M_{d,s}$ from Crivellin 1903.10440')
plt.xlabel(r'$\log_{10}(\tan\beta)$') # log10
plt.ylabel(r'$\log_{10}(m_{H^+})$') #log10
#plt.savefig('bmix_plot.png')
plt.savefig('bcriv_plot.png')

plt.figure(figsize=(6,5))
fpl.contour(**crad,interpolation_factor=3,col=3) 
plt.title(r'$\bar{B}\to X_s\gamma$ Radiative Decay')
plt.xlabel(r'$\log_{10}(\tan\beta)$') # log10
plt.ylabel(r'$\log_{10}(m_{H^+})$') #log10
plt.savefig('bsgamma_plot.png')

plt.figure(figsize=(6,5))
fpl.contour(**cmu,interpolation_factor=3,col=9) 
plt.title(r'FCNC Leptonic B Decays ($B_{s,d}\to\mu^+\mu^-$)')
plt.xlabel(r'$\log_{10}(\tan\beta)$') # log10
plt.ylabel(r'$\log_{10}(m_{H^+})$') #log10
plt.savefig('bmumu_plot.png')

plt.figure(figsize=(6,5))
fpl.contour(**cdat,interpolation_factor=3,col=4) 
plt.title('All Observables')
plt.xlabel(r'$\log_{10}(\tan\beta)$') # log10
plt.ylabel(r'$\log_{10}(m_{H^+})$') #log10
plt.savefig('comb_plot.png')

#plt.show()
# colours: 0 - blue, 1 - orange, 2 - green, 3 - pink, 4 - purple, 5 - brown, 6 - bright pink, 7 - grey, 8 - yellow, 9 - cyan
