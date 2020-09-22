#!/bin/env python3

import flavio
from flavio.statistics.likelihood import FastLikelihood
from flavio.physics.bdecays.formfactors import b_v, b_p
from flavio.classes import Parameter, AuxiliaryQuantity, Implementation
from flavio.config import config
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

#### defining Bs->Ds(*)lnu

def ff_function(function, process, **kwargs):
    return lambda wc_obj, par_dict, q2: function(process, q2, par_dict, **kwargs)

config['implementation']['Bs->Ds* form factor'] = 'Bs->Ds* CLN' 
bsds = AuxiliaryQuantity('Bs->Ds* form factor') 
bsdsi = Implementation(name='Bs->Ds* CLN',quantity='Bs->Ds* form factor',function=ff_function(b_v.cln.ff,'Bs->Ds*',scale=config['renormalization scale']['bvll']))
bsdsi.set_description("CLN parameterization")
#print(flavio.sm_prediction('BR(Bs->Ds*munu)')*1e2)
#print(flavio.sm_uncertainty('BR(Bs->Ds*munu)')*1e2)

config['implementation']['Bs->Ds form factor'] = 'Bs->Ds CLN'
bsd = AuxiliaryQuantity('Bs->Ds form factor')
bsdi = Implementation(name='Bs->Ds CLN',quantity='Bs->Ds form factor',function=ff_function(b_p.cln.ff,'Bs->Ds',scale=config['renormalization scale']['bpll']))
bsdsi.set_description("CLN parameterization")
#print(flavio.sm_prediction('BR(Bs->Dsmunu)')*1e2)
#print(flavio.sm_uncertainty('BR(Bs->Dsmunu)')*1e2)

#print(flavio.functions.get_dependent_parameters_sm('BR(Bs->Dsmunu)'))   

#### fitting stuff

flavio.measurements.read_file('world_avgs.yml') # read in the world averages we want to use

config['renormalization scale']['bxgamma'] = 1.74

sigmas = (1,2)
#sigmas = (3,4)

my_obs = [
    'BR(B+->taunu)', 'BR(B+->munu)', 'BR(D+->munu)', 'BR(Ds->munu)', 'BR(Ds->taunu)', 'BR(tau->Knu)', 'BR(K+->munu)', 'BR(tau->pinu)', 'Gamma(pi+->munu)', # [:9]
    'BR(B->Xsgamma)', # [9]
    'DeltaM_d','DeltaM_s', # [10:12]
    'BR(Bs->mumu)','BR(B0->mumu)', # [12:14]
    'Rtaul(B->Dlnu)', 'Rtaul(B->D*lnu)', # [14:16]
    'BR(B0->Dlnu)', 'BR(B+->Dlnu)', 'BR(B0->D*lnu)', 'BR(B+->D*lnu)', 'BR(B+->pilnu)', 'BR(B0->pilnu)', 'BR(B+->rholnu)', 'BR(B0->rholnu)', 'BR(D+->Kenu)', 'BR(D+->Kmunu)', 'BR(D+->pienu)', 'BR(D+->pimunu)', 'BR(D0->Kenu)', 'BR(D0->Kmunu)', 'BR(D0->pienu)', 'BR(D0->pimunu)', 'BR(K+->pienu)', 'BR(K+->pimunu)', 'BR(KL->pienu)', 'BR(KL->pimunu)', 'BR(KS->pienu)', 'BR(KS->pimunu)', # [16:38]
    ("<Rmue>(B+->Kll)", 1.0, 6.0), # [38]
    ("<Rmue>(B0->K*ll)", 0.045, 1.1), # [39]
    ("<Rmue>(B0->K*ll)", 1.1, 6.0), # [40]
]

obs2 = [
    'BR(Bs->Dsmunu)','BR(Bs->Ds*munu)',
]

#------------------------------
#   Leptonic and Semileptonic Tree Levels
#------------------------------

#Fleps = FastLikelihood(name="trees",observables=my_obs[:9]+my_obs[14:38],include_measurements=['Tree Level Leptonics','LFU D Ratios','Tree Level Semileptonics']) 
Fleps = FastLikelihood(name="trees",observables=obs2,include_measurements=['Tree Level Semileptonics']) 
Fleps.make_measurement(N=500,threads=4)

def leps(wcs):
    tanb, mH = wcs # state what the two parameters are going to be on the plot

    par = flavio.default_parameters.get_central_all()
    ckm_els = flavio.physics.ckm.get_ckm(par) # get out all the CKM elements

#    CSR_b, CSL_b = rh(par['m_u'],par['m_b'],10**tanb,10**mH)
#    CSR_d, CSL_d = rh(par['m_c'],par['m_d'],10**tanb,10**mH)
#    CSR_ds, CSL_ds = rh(par['m_c'],par['m_s'],10**tanb,10**mH)
#    CSR_k, CSL_k = rh(par['m_u'],par['m_s'],10**tanb,10**mH)
#    CSR_p, CSL_p = rh(par['m_u'],par['m_d'],10**tanb,10**mH)
    CSL_bc, CSR_bc = rh(par['m_c'],par['m_b'],10**tanb,10**mH)

    wc = flavio.WilsonCoefficients()
    wc.set_initial({ # tell flavio what WCs you're referring to with your variables
#            'CSR_bctaunutau': CSR_bc, 'CSL_bctaunutau': CSL_bc,
            'CSR_bcmunumu': CSR_bc, 'CSL_bcmunumu': CSL_bc,
#            'CSR_bcenue': CSR_bc, 'CSL_bcenue': CSL_bc, 
#            'CSR_butaunutau': CSR_b, 'CSL_butaunutau': CSL_b,
#            'CSR_bumunumu': CSR_b, 'CSL_bumunumu': CSL_b,
#            'CSR_buenue': CSR_b, 'CSL_buenue': CSL_b, 
#            'CSR_dcmunumu': CSR_d, 'CSL_dcmunumu': CSL_d,
#            'CSR_dcenue': CSR_d, 'CSL_dcenue': CSL_d, 
#            'CSR_sctaunutau': CSR_ds, 'CSL_sctaunutau': CSL_ds,
#            'CSR_scmunumu': CSR_ds, 'CSL_scmunumu': CSL_ds,
#            'CSR_scenue': CSR_ds, 'CSL_scenue': CSL_ds, 
#            'CSR_sutaunutau': CSR_k, 'CSL_sutaunutau': CSL_k, 
#            'CSR_sumunumu': CSR_k, 'CSL_sumunumu': CSL_k, 
#            'CSR_suenue': CSR_k, 'CSL_suenue': CSL_k, 
#            'CSR_dutaunutau': CSR_p, 'CSL_dutaunutau': CSL_p, 
#            'CSR_dumunumu': CSR_p, 'CSL_dumunumu': CSL_p, 
        }, scale=4.2, eft='WET', basis='flavio')
    return Fleps.log_likelihood(par,wc)

#------------------------------
#   B Mixing
#------------------------------

Fmix = FastLikelihood(name="mix",observables=my_obs[10:12],include_measurements=['B Mixing',]) 
Fmix.make_measurement(N=500,threads=4)

def mix(wcs):
    tanb, mH = wcs # state what the two parameters are going to be on the plot

    par = flavio.default_parameters.get_central_all()
    ckm_els = flavio.physics.ckm.get_ckm(par) # get out all the CKM elements

    CVLL_bs, CVRR_bs, CSLL_bs, CSRR_bs, CSLR_bs, CVLR_bs = mixing(par,ckm_els,['m_s',1,'m_d'],10**tanb,10**mH)
    CVLL_bd, CVRR_bd, CSLL_bd, CSRR_bd, CSLR_bd, CVLR_bd = mixing(par,ckm_els,['m_d',0,'m_s'],10**tanb,10**mH)

    wc = flavio.WilsonCoefficients()
    wc.set_initial({ # tell flavio what WCs you're referring to with your variables
            'CVLL_bsbs': CVLL_bs,'CVRR_bsbs': CVRR_bs,'CSLL_bsbs': CSLL_bs,'CSRR_bsbs': CSRR_bs,'CSLR_bsbs': CSLR_bs,'CVLR_bsbs': CVLR_bs, # DeltaM_s
            'CVLL_bdbd': CVLL_bd,'CVRR_bdbd': CVRR_bd,'CSLL_bdbd': CSLL_bd,'CSRR_bdbd': CSRR_bd,'CSLR_bdbd': CSLR_bd,'CVLR_bdbd': CVLR_bd, # DeltaM_d
        }, scale=4.2, eft='WET', basis='flavio')
    return Fmix.log_likelihood(par,wc)

#------------------------------
#   B->Xsgamma Radiative Decay
#------------------------------

Frad = FastLikelihood(name="rad",observables=[my_obs[9]],include_measurements=['Radiative Decays'])
Frad.make_measurement(N=500,threads=4)

def rad(wcs):
    tanb, mH = wcs # state what the two parameters are going to be on the plot

    par = flavio.default_parameters.get_central_all()
    ckm_els = flavio.physics.ckm.get_ckm(par) # get out all the CKM elements

    C7, C7p, C8, C8p = bsgamma2(par,ckm_els,flavio.config['renormalization scale']['bxgamma'],10**tanb,10**mH)

    wc = flavio.WilsonCoefficients()
    wc.set_initial({ # tell flavio what WCs you're referring to with your variables
        'C7_bs': C7, 'C8_bs': C8,
        'C7p_bs': C7p, 'C8p_bs': C8p,
        }, scale=4.2, eft='WET', basis='flavio')
    return Frad.log_likelihood(par,wc)

#------------------------------
#   B(s/d) -> mumu + R(K) & R(K*)
#------------------------------

Fmu = FastLikelihood(name="mu",observables=my_obs[-2:],include_measurements=['LFU K Ratios']) 
#Fmu = FastLikelihood(name="mu",observables=my_obs[12:14],include_measurements=['FCNC Leptonic Decays',]) 
#Fmu = FastLikelihood(name="mu",observables=my_obs[12:14]+my_obs[-3:],include_measurements=['FCNC Leptonic Decays','LFU K Ratios']) 
Fmu.make_measurement(N=500,threads=4)

def mu(wcs):
    tanb, mH = wcs # state what the two parameters are going to be on the plot
    mH0 = mH
#    mH0 = np.log10(1500)

    par = flavio.default_parameters.get_central_all()
    ckm_els = flavio.physics.ckm.get_ckm(par) # get out all the CKM elements

    C9_se, C9p_se, C10_se, C10p_se, CS_se, CSp_se, CP_se, CPp_se = bsll(par,ckm_els,['m_s','m_d',1],['m_e','m_mu',1],10**mH0,10**tanb,10**mH)
    C9_s, C9p_s, C10_s, C10p_s, CS_s, CSp_s, CP_s, CPp_s = bsll(par,ckm_els,['m_s','m_d',1],['m_mu','m_e',1],10**mH0,10**tanb,10**mH)
#    C9_d, C9p_d, C10_d, C10p_d, CS_d, CSp_d, CP_d, CPp_d = bsll(par,ckm_els,['m_d','m_s',0],['m_mu','m_e',1],10**mH0,10**tanb,10**mH)
    C7, C7p, C8, C8p = bsgamma2(par,ckm_els,flavio.config['renormalization scale']['bxgamma'],10**tanb,10**mH)

    wc = flavio.WilsonCoefficients()
    wc.set_initial({ # tell flavio what WCs you're referring to with your variables
           'C7_bs': C7,'C7p_bs': C7p, 
           'C8_bs': C8,'C8p_bs': C8p, 
           'C9_bsee': C9_se,'C9p_bsee': C9p_se,
           'C9_bsmumu': C9_s,'C9p_bsmumu': C9p_s,
           'C10_bsee': C10_se,'C10p_bsee': C10p_se,
           'C10_bsmumu': C10_s,'C10p_bsmumu': C10p_s,#'CS_bsmumu': CS_s,'CSp_bsmumu': CSp_s,'CP_bsmumu': CP_s,'CPp_bsmumu': CPp_s, # Bs->mumu
#            'C10_bdmumu': C10_d,'C10p_bdmumu': C10p_d,'CS_bdmumu': CS_d,'CSp_bdmumu': CSp_d,'CP_bdmumu': CP_d,'CPp_bdmumu': CPp_d, # B0->mumu
        }, scale=4.2, eft='WET', basis='flavio')
    return Fmu.log_likelihood(par,wc)

#------------------------------
#   All Observables
#------------------------------

#FL2 = FastLikelihood(name="likelihood test",observables=my_obs,include_measurements=['Tree Level Leptonics','Radiative Decays','FCNC Leptonic Decays','B Mixing','LFU D Ratios','Tree Level Semileptonics','LFU K Ratios',])
FL2 = FastLikelihood(name="likelihood test",observables=my_obs[:16],include_measurements=['Tree Level Leptonics','Radiative Decays','FCNC Leptonic Decays','B Mixing','LFU D Ratios',])
#FL2 = FastLikelihood(name="likelihood test",observables=my_obs[:12],include_measurements=['Tree Level Leptonics','Radiative Decays','B Mixing',])
FL2.make_measurement(N=500,threads=4)

def func(wcs):
    tanb, mH = wcs # state what the two parameters are going to be on the plot
    mH0 = mH
#    mH0 = np.log10(1500)

    par = flavio.default_parameters.get_central_all()
    ckm_els = flavio.physics.ckm.get_ckm(par) # get out all the CKM elements

    CSR_b, CSL_b = rh(par['m_u'],par['m_b'],10**tanb,10**mH)
    CSR_d, CSL_d = rh(par['m_c'],par['m_d'],10**tanb,10**mH)
    CSR_ds, CSL_ds = rh(par['m_c'],par['m_s'],10**tanb,10**mH)
    CSR_k, CSL_k = rh(par['m_u'],par['m_s'],10**tanb,10**mH)
    CSR_p, CSL_p = rh(par['m_u'],par['m_d'],10**tanb,10**mH)
    CSL_bc, CSR_bc = rh(par['m_c'],par['m_b'],10**tanb,10**mH)
    C7, C7p, C8, C8p = bsgamma2(par,ckm_els,flavio.config['renormalization scale']['bxgamma'],10**tanb,10**mH)
    C9_s, C9p_s, C10_s, C10p_s, CS_s, CSp_s, CP_s, CPp_s = bsll(par,ckm_els,['m_s','m_d',1],['m_mu','m_e',1],10**mH0,10**tanb,10**mH)
    C9_d, C9p_d, C10_d, C10p_d, CS_d, CSp_d, CP_d, CPp_d = bsll(par,ckm_els,['m_d','m_s',0],['m_mu','m_e',1],10**mH0,10**tanb,10**mH)
    CVLL_bs, CVRR_bs, CSLL_bs, CSRR_bs, CSLR_bs, CVLR_bs = mixing(par,ckm_els,['m_s',1,'m_d'],10**tanb,10**mH)
    CVLL_bd, CVRR_bd, CSLL_bd, CSRR_bd, CSLR_bd, CVLR_bd = mixing(par,ckm_els,['m_d',0,'m_s'],10**tanb,10**mH)

    wc = flavio.WilsonCoefficients()
    wc.set_initial({ # tell flavio what WCs you're referring to with your variables
            'CSR_bctaunutau': CSR_bc, 'CSL_bctaunutau': CSL_bc,
            'CSR_bcmunumu': CSR_bc, 'CSL_bcmunumu': CSL_bc,
            'CSR_bcenue': CSR_bc, 'CSL_bcenue': CSL_bc, 
            'CSR_butaunutau': CSR_b, 'CSL_butaunutau': CSL_b,
            'CSR_bumunumu': CSR_b, 'CSL_bumunumu': CSL_b,
#            'CSR_buenue': CSR_b, 'CSL_buenue': CSL_b, 
            'CSR_dcmunumu': CSR_d, 'CSL_dcmunumu': CSL_d,
#            'CSR_dcenue': CSR_d, 'CSL_dcenue': CSL_d, 
            'CSR_sctaunutau': CSR_ds, 'CSL_sctaunutau': CSL_ds,
            'CSR_scmunumu': CSR_ds, 'CSL_scmunumu': CSL_ds,
#            'CSR_scenue': CSR_ds, 'CSL_scenue': CSL_ds, 
            'CSR_sutaunutau': CSR_k, 'CSL_sutaunutau': CSL_k, 
            'CSR_sumunumu': CSR_k, 'CSL_sumunumu': CSL_k, 
#            'CSR_suenue': CSR_k, 'CSL_suenue': CSL_k, 
            'CSR_dutaunutau': CSR_p, 'CSL_dutaunutau': CSL_p, 
            'CSR_dumunumu': CSR_p, 'CSL_dumunumu': CSL_p, 
            'C7_bs': C7,'C8_bs': C8,'C7p_bs': C7p, 'C8p_bs': C8p, # B -> Xsgamma
            'C10_bsmumu': C10_s,'C10p_bsmumu': C10p_s,'CS_bsmumu': CS_s,'CSp_bsmumu': CSp_s,'CP_bsmumu': CP_s,'CPp_bsmumu': CPp_s, # Bs->mumu
            'C10_bdmumu': C10_d,'C10p_bdmumu': C10p_d,'CS_bdmumu': CS_d,'CSp_bdmumu': CSp_d,'CP_bdmumu': CP_d,'CPp_bdmumu': CPp_d, # B0->mumu
            'CVLL_bsbs': CVLL_bs,'CVRR_bsbs': CVRR_bs,'CSLL_bsbs': CSLL_bs,'CSRR_bsbs': CSRR_bs,'CSLR_bsbs': CSLR_bs,'CVLR_bsbs': CVLR_bs, # DeltaM_s
            'CVLL_bdbd': CVLL_bd,'CVRR_bdbd': CVRR_bd,'CSLL_bdbd': CSLL_bd,'CSRR_bdbd': CSRR_bd,'CSLR_bdbd': CSLR_bd,'CVLR_bdbd': CVLR_bd, # DeltaM_d
        }, scale=4.2, eft='WET', basis='flavio')
    return FL2.log_likelihood(par,wc)

#------------------------------
#   Get Contour Data
#------------------------------

cleps = fpl.likelihood_contour_data(leps,-1,2,1,3.5, n_sigma=sigmas, threads=4, steps=60) 
#cmix = fpl.likelihood_contour_data(mix,-1,2,1,3.5, n_sigma=sigmas, threads=4, steps=60) 
#crad = fpl.likelihood_contour_data(rad,-1,2,1,3.5, n_sigma=sigmas, threads=4, steps=60) 
#cmu = fpl.likelihood_contour_data(mu,-1,2,1,3.5, n_sigma=sigmas, threads=4, steps=60) 
#cdat = fpl.likelihood_contour_data(func,-1,2,1,3.5, n_sigma=sigmas, threads=4, steps=60) 

#------------------------------
#   Print Out Values
#------------------------------

#bf,minh,mint,maxt = mHmin(cdat)
#print("Best fit value is found for (tanb,mH) =", bf)
#print("Print outs are lists for values at", sigmas, "sigmas")
#print("Minimum value of mH+ is:", minh)
#print("Minimum value of tanb is:", mint)
#print("Maximum value of tanb is:", maxt)

#------------------------------
#   Plotting
#------------------------------

plt.figure(figsize=(6,5))
fpl.contour(**cleps,col=2) 
#plt.title('Tree Level Leptonic and Semileptonics and Hadronic Tau Decays')
plt.title(r'$B_s\to D_s^{(*)}\mu\nu_\mu$ Fit')
plt.xlabel(r'$\log_{10}[\tan\beta]$') 
plt.ylabel(r'$\log_{10}[m_{H^+} (\text{GeV})]$') 
plt.savefig('leps_plot.png')
quit()

#plt.figure(figsize=(6,5))
#fpl.contour(**cmix,col=0) 
#plt.title(r'$\Delta M_{d,s}$')
#plt.xlabel(r'$\log_{10}[\tan\beta]$') 
#plt.ylabel(r'$\log_{10}[m_{H^+} (\text{GeV})]$') 
#plt.savefig('bmix_plot.png')

#plt.figure(figsize=(6,5))
#fpl.contour(**crad,col=3) 
#plt.title(r'$\bar{B}\to X_s\gamma$ Radiative Decay')
#plt.xlabel(r'$\log_{10}[\tan\beta]$') 
#plt.ylabel(r'$\log_{10}[m_{H^+} (\text{GeV})]$') 
#plt.savefig('bsgamma_plot.png')

# (2.4,4.2,-1,1)
z_min1 = -5.182818890948422
# (1,3,0,3.5)
z_min2 = -2.1479938777001504
# (-1,2.5,0,3.5)
z_min3 = 1.6111964187252177
# (-1,2,0,3.5)
z_min4 = 3.5229317450326256

plt.figure(figsize=(6,5))
fpl.contour(**cmu,col=9)#,z_min=z_min1) 
#plt.title(r'FCNC Leptonic B Decays ($B_{s,d}\to\mu^+\mu^-$), $m_{H^0}\sim m_{H^+}$')
#plt.title(r'FCNC Leptonic B Decays ($B_{s,d}\to\mu^+\mu^-$), $m_{H^0}=1500\,$GeV')
plt.title(r'$R_K$ for $q^2\in[1,6]$ \& $R_{K^{*0}}$ for $q^2\in[0.045,6]$')
#plt.title(r'$R_K$ for $q^2\in[1,6]$')
#plt.title(r'$R_{K^{*0}}$ for $q^2\in[0.045,6]$')
#plt.title(r'$b\to sl^+l^-$ transitions ($B_{s,d}\to\mu^+\mu^-$ \& $R_K(q^2\in[1,6]),R_{K^{*0}}(q^2\in[0.045,6])$), $m_{H^0}\sim m_{H^+}$')
#plt.title(r'$b\to sl^+l^-$ transitions ($B_{s,d}\to\mu^+\mu^-$ \& $R_K(q^2\in[1,6]),R_{K^{*0}}(q^2\in[0.045,6])$), $m_{H^0}=1500\,$GeV')
plt.xlabel(r'$\log_{10}[\tan\beta]$') 
plt.ylabel(r'$\log_{10}[m_{H^+} (\text{GeV})]$') 
#plt.savefig('bsll_plot.png')
#plt.savefig('bmumu_fix.png')
plt.savefig('rks_plot.png')

#plt.figure(figsize=(6,5))
#fpl.contour(**cdat,col=4) 
#plt.title(r'Combined Tree-Level Leptonics, $\Delta M_{d,s}$, $\bar{B}\to X_s\gamma$')
#plt.title(r'Combined Tree-Level Leptonics, $\Delta M_{d,s}$, $\bar{B}\to X_s\gamma$,' '\n' r'$\mathcal{R}(D^{(*)})$ and $B_{s,d}\to\mu^+\mu^-$ for $m_{H^0}\sim m_{H^+}$')
#plt.title(r'Combined Tree-Level Leptonics, $\Delta M_{d,s}$, $\bar{B}\to X_s\gamma$,' '\n' r'$\mathcal{R}(D^{(*)})$ and $B_{s,d}\to\mu^+\mu^-$ for $m_{H^0}=1500\,$GeV')
#plt.xlabel(r'$\log_{10}[\tan\beta]$') 
#plt.ylabel(r'$\log_{10}[m_{H^+} (\text{GeV})]$')
#plt.savefig('comb1_plot.png')
#plt.savefig('comb2_fix.png')
#plt.savefig('comb2_apx.png')

#plt.show()
# colours : 0 - blue, 1 - orange, 2 - green, 3 - pink, 4 - purple, 5 - brown, 6 - bright pink, 7 - grey, 8 - yellow, 9 - cyan
