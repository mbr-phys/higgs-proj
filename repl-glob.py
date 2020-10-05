#!/bin/env python3

import flavio
from flavio.statistics.likelihood import FastLikelihood
from flavio.physics.bdecays.formfactors import b_v, b_p
from flavio.classes import Parameter, AuxiliaryQuantity, Implementation
from flavio.statistics.functions import pvalue
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

obs2 = ['BR(Bs->Dsmunu)','BR(Bs->Ds*munu)',]

#Fleps = FastLikelihood(name="trees",observables=my_obs[:9]+my_obs[14:38]+obs2,include_measurements=['Tree Level Leptonics','LFU D Ratios','Tree Level Semileptonics']) 
#Fleps = FastLikelihood(name="trees",observables=my_obs[14:38]+obs2,include_measurements=['LFU D Ratios','Tree Level Semileptonics']) 
Fleps = FastLikelihood(name="trees",observables=my_obs[14:16],include_measurements=['LFU D Ratios',]) 
#Fleps = FastLikelihood(name="trees",observables=[my_obs[14]],include_measurements=['LFU D Ratios',]) 
#Fleps = FastLikelihood(name="trees",observables=[my_obs[15]],include_measurements=['LFU D Ratios',]) 
#Fmix = FastLikelihood(name="mix",observables=my_obs[10:12],include_measurements=['B Mixing',]) 
#Frad = FastLikelihood(name="rad",observables=[my_obs[9]],include_measurements=['Radiative Decays'])
#------------------------------
#Fmu = FastLikelihood(name="mu",observables=my_obs[-3:],include_measurements=['LFU K Ratios']) 
#Fmu = FastLikelihood(name="mu",observables=my_obs[12:14],include_measurements=['FCNC Leptonic Decays',]) 
#------------------------------
#obs_list = my_obs+obs2
#FL2 = FastLikelihood(name="glob",observables=obs_list,include_measurements=['Tree Level Leptonics','Radiative Decays','FCNC Leptonic Decays','B Mixing','LFU D Ratios','Tree Level Semileptonics','LFU K Ratios',])
#FL2 = FastLikelihood(name="glob",observables=my_obs[:-3]+obs2,include_measurements=['Tree Level Leptonics','Radiative Decays','FCNC Leptonic Decays','B Mixing','LFU D Ratios','Tree Level Semileptonics',])
#------------------------------
#Fmuon = FastLikelihood(name="muons",observables=['a_mu'],include_measurements=['Anomalous Magnetic Moments'])
#Fmuon.make_measurement(N=500,threads=4)

Fleps.make_measurement(N=500,threads=4)
#Fmix.make_measurement(N=500,threads=4)
#Frad.make_measurement(N=500,threads=4)
#Fmu.make_measurement(N=500,threads=4)
#FL2.make_measurement(N=500,threads=4)

#------------------------------
#   Leptonic and Semileptonic Tree Levels
#------------------------------

def leps(wcs):
    tanb, mH = wcs # state what the two parameters are going to be on the plot

    par = flavio.default_parameters.get_central_all()
    ckm_els = flavio.physics.ckm.get_ckm(par) # get out all the CKM elements

#    CSR_b_t, CSL_b_t = rh(par['m_u'],par['m_b'],par['m_tau'],10**tanb,10**mH)
#    CSR_b_m, CSL_b_m = rh(par['m_u'],par['m_b'],par['m_mu'],10**tanb,10**mH)
#    CSR_b_e, CSL_b_e = rh(par['m_u'],par['m_b'],par['m_e'],10**tanb,10**mH)
#    CSR_d_m, CSL_d_m = rh(par['m_c'],par['m_d'],par['m_mu'],10**tanb,10**mH)
#    CSR_d_e, CSL_d_e = rh(par['m_c'],par['m_d'],par['m_e'],10**tanb,10**mH)
#    CSR_ds_t, CSL_ds_t = rh(par['m_c'],par['m_s'],par['m_tau'],10**tanb,10**mH)
#    CSR_ds_m, CSL_ds_m = rh(par['m_c'],par['m_s'],par['m_mu'],10**tanb,10**mH)
#    CSR_ds_e, CSL_ds_e = rh(par['m_c'],par['m_s'],par['m_e'],10**tanb,10**mH)
#    CSR_k_t, CSL_k_t = rh(par['m_u'],par['m_s'],par['m_tau'],10**tanb,10**mH)
#    CSR_k_m, CSL_k_m = rh(par['m_u'],par['m_s'],par['m_mu'],10**tanb,10**mH)
#    CSR_k_e, CSL_k_e = rh(par['m_u'],par['m_s'],par['m_e'],10**tanb,10**mH)
#    CSR_p_t, CSL_p_t = rh(par['m_u'],par['m_d'],par['m_tau'],10**tanb,10**mH)
#    CSR_p_m, CSL_p_m = rh(par['m_u'],par['m_d'],par['m_mu'],10**tanb,10**mH)
    CSR_bc_t, CSL_bc_t = rh(par['m_c'],par['m_b'],par['m_tau'],10**tanb,10**mH)
    CSR_bc_m, CSL_bc_m = rh(par['m_c'],par['m_b'],par['m_mu'],10**tanb,10**mH)
    CSR_bc_e, CSL_bc_e = rh(par['m_c'],par['m_b'],par['m_e'],10**tanb,10**mH)

    wc = flavio.WilsonCoefficients()
    wc.set_initial({ # tell flavio what WCs you're referring to with your variables
            'CSR_bctaunutau': CSR_bc_t, 'CSL_bctaunutau': CSL_bc_t,
            'CSR_bcmunumu': CSR_bc_m, 'CSL_bcmunumu': CSL_bc_m,
            'CSR_bcenue': CSR_bc_e, 'CSL_bcenue': CSL_bc_e, 
#            'CSR_butaunutau': CSR_b_t, 'CSL_butaunutau': CSL_b_t,
#            'CSR_bumunumu': CSR_b_m, 'CSL_bumunumu': CSL_b_m,
#            'CSR_buenue': CSR_b_e, 'CSL_buenue': CSL_b_e, 
#            'CSR_dcmunumu': CSR_d_m, 'CSL_dcmunumu': CSL_d_m,
#            'CSR_dcenue': CSR_d_e, 'CSL_dcenue': CSL_d_e, 
#            'CSR_sctaunutau': CSR_ds_t, 'CSL_sctaunutau': CSL_ds_t,
#            'CSR_scmunumu': CSR_ds_m, 'CSL_scmunumu': CSL_ds_m,
#            'CSR_scenue': CSR_ds_e, 'CSL_scenue': CSL_ds_e, 
#            'CSR_sutaunutau': CSR_k_t, 'CSL_sutaunutau': CSL_k_t, 
#            'CSR_sumunumu': CSR_k_m, 'CSL_sumunumu': CSL_k_m, 
#            'CSR_suenue': CSR_k_e, 'CSL_suenue': CSL_k_e, 
#            'CSR_dutaunutau': CSR_p_t, 'CSL_dutaunutau': CSL_p_t, 
#            'CSR_dumunumu': CSR_p_m, 'CSL_dumunumu': CSL_p_m, 
        }, scale=4.2, eft='WET', basis='flavio')
    return Fleps.log_likelihood(par,wc)

#------------------------------
#   B Mixing
#------------------------------

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

def mu(wcs):
    tanb, mH = wcs # state what the two parameters are going to be on the plot
#    mH0 = mH
    mH0 = np.log10(1500)

    par = flavio.default_parameters.get_central_all()
    ckm_els = flavio.physics.ckm.get_ckm(par) # get out all the CKM elements

#    C9_se, C9p_se, C10_se, C10p_se, CS_se, CSp_se, CP_se, CPp_se = bsll(par,ckm_els,['m_s','m_d',1],['m_e','m_mu',1],10**mH0,10**tanb,10**mH)
    C9_s, C9p_s, C10_s, C10p_s, CS_s, CSp_s, CP_s, CPp_s = bsll(par,ckm_els,['m_s','m_d',1],['m_mu','m_e',1],10**mH0,10**tanb,10**mH)
    C9_d, C9p_d, C10_d, C10p_d, CS_d, CSp_d, CP_d, CPp_d = bsll(par,ckm_els,['m_d','m_s',0],['m_mu','m_e',1],10**mH0,10**tanb,10**mH)
#    C7, C7p, C8, C8p = bsgamma2(par,ckm_els,flavio.config['renormalization scale']['bxgamma'],10**tanb,10**mH)

    wc = flavio.WilsonCoefficients()
    wc.set_initial({ # tell flavio what WCs you're referring to with your variables
#           'C7_bs': C7,'C7p_bs': C7p, 
#           'C8_bs': C8,'C8p_bs': C8p, 
#           'C9_bsee': C9_se,'C9p_bsee': C9p_se,
#           'C9_bsmumu': C9_s,'C9p_bsmumu': C9p_s,
#           'C10_bsee': C10_se,'C10p_bsee': C10p_se,
           'C10_bsmumu': C10_s,'C10p_bsmumu': C10p_s,'CS_bsmumu': CS_s,'CSp_bsmumu': CSp_s,'CP_bsmumu': CP_s,'CPp_bsmumu': CPp_s, # Bs->mumu
            'C10_bdmumu': C10_d,'C10p_bdmumu': C10p_d,'CS_bdmumu': CS_d,'CSp_bdmumu': CSp_d,'CP_bdmumu': CP_d,'CPp_bdmumu': CPp_d, # B0->mumu
        }, scale=4.2, eft='WET', basis='flavio')
    return Fmu.log_likelihood(par,wc)

#------------------------------
#   Anomalous moments
#------------------------------

def muon(wcs):
    tanb,mH = wcs
    
#    mH0,mA0 = mH, mH
#    mH0,mA0 = np.log10(1500), mH
    mH0,mA0 = mH, np.log10(1500)

    par = flavio.default_parameters.get_central_all()

    csev = a_mu(par,'m_mu',10**tanb,10**mH0,10**mA0,10**mH)

    wc = flavio.WilsonCoefficients()
    wc.set_initial({'C7_mumu': csev},scale=1.0,eft='WET-3',basis='flavio')
    return Fmuon.log_likelihood(par,wc)

#------------------------------
#   All Observables
#------------------------------

def func(wcs):
    tanb, mH = wcs # state what the two parameters are going to be on the plot
#    mH0 = mH
    mH0 = np.log10(1500)

    par = flavio.default_parameters.get_central_all()
    ckm_els = flavio.physics.ckm.get_ckm(par) # get out all the CKM elements

    CSR_b_t, CSL_b_t = rh(par['m_u'],par['m_b'],par['m_tau'],10**tanb,10**mH)
    CSR_b_m, CSL_b_m = rh(par['m_u'],par['m_b'],par['m_mu'],10**tanb,10**mH)
    CSR_b_e, CSL_b_e = rh(par['m_u'],par['m_b'],par['m_e'],10**tanb,10**mH)
    CSR_d_m, CSL_d_m = rh(par['m_c'],par['m_d'],par['m_mu'],10**tanb,10**mH)
    CSR_d_e, CSL_d_e = rh(par['m_c'],par['m_d'],par['m_e'],10**tanb,10**mH)
    CSR_ds_t, CSL_ds_t = rh(par['m_c'],par['m_s'],par['m_tau'],10**tanb,10**mH)
    CSR_ds_m, CSL_ds_m = rh(par['m_c'],par['m_s'],par['m_mu'],10**tanb,10**mH)
    CSR_ds_e, CSL_ds_e = rh(par['m_c'],par['m_s'],par['m_e'],10**tanb,10**mH)
    CSR_k_t, CSL_k_t = rh(par['m_u'],par['m_s'],par['m_tau'],10**tanb,10**mH)
    CSR_k_m, CSL_k_m = rh(par['m_u'],par['m_s'],par['m_mu'],10**tanb,10**mH)
    CSR_k_e, CSL_k_e = rh(par['m_u'],par['m_s'],par['m_e'],10**tanb,10**mH)
    CSR_p_t, CSL_p_t = rh(par['m_u'],par['m_d'],par['m_tau'],10**tanb,10**mH)
    CSR_p_m, CSL_p_m = rh(par['m_u'],par['m_d'],par['m_mu'],10**tanb,10**mH)
    CSR_bc_t, CSL_bc_t = rh(par['m_c'],par['m_b'],par['m_tau'],10**tanb,10**mH)
    CSR_bc_m, CSL_bc_m = rh(par['m_c'],par['m_b'],par['m_mu'],10**tanb,10**mH)
    CSR_bc_e, CSL_bc_e = rh(par['m_c'],par['m_b'],par['m_e'],10**tanb,10**mH)
    C7, C7p, C8, C8p = bsgamma2(par,ckm_els,flavio.config['renormalization scale']['bxgamma'],10**tanb,10**mH)
    C9_se, C9p_se, C10_se, C10p_se, CS_se, CSp_se, CP_se, CPp_se = bsll(par,ckm_els,['m_s','m_d',1],['m_e','m_mu',1],10**mH0,10**tanb,10**mH)
    C9_s, C9p_s, C10_s, C10p_s, CS_s, CSp_s, CP_s, CPp_s = bsll(par,ckm_els,['m_s','m_d',1],['m_mu','m_e',1],10**mH0,10**tanb,10**mH)
    C9_d, C9p_d, C10_d, C10p_d, CS_d, CSp_d, CP_d, CPp_d = bsll(par,ckm_els,['m_d','m_s',0],['m_mu','m_e',1],10**mH0,10**tanb,10**mH)
    CVLL_bs, CVRR_bs, CSLL_bs, CSRR_bs, CSLR_bs, CVLR_bs = mixing(par,ckm_els,['m_s',1,'m_d'],10**tanb,10**mH)
    CVLL_bd, CVRR_bd, CSLL_bd, CSRR_bd, CSLR_bd, CVLR_bd = mixing(par,ckm_els,['m_d',0,'m_s'],10**tanb,10**mH)

    wc = flavio.WilsonCoefficients()
    wc.set_initial({ # tell flavio what WCs you're referring to with your variables
            'CSR_bctaunutau': CSR_bc_t, 'CSL_bctaunutau': CSL_bc_t,
            'CSR_bcmunumu': CSR_bc_m, 'CSL_bcmunumu': CSL_bc_m,
            'CSR_bcenue': CSR_bc_e, 'CSL_bcenue': CSL_bc_e, 
            'CSR_butaunutau': CSR_b_t, 'CSL_butaunutau': CSL_b_t,
            'CSR_bumunumu': CSR_b_m, 'CSL_bumunumu': CSL_b_m,
            'CSR_buenue': CSR_b_e, 'CSL_buenue': CSL_b_e, 
            'CSR_dcmunumu': CSR_d_m, 'CSL_dcmunumu': CSL_d_m,
            'CSR_dcenue': CSR_d_e, 'CSL_dcenue': CSL_d_e, 
            'CSR_sctaunutau': CSR_ds_t, 'CSL_sctaunutau': CSL_ds_t,
            'CSR_scmunumu': CSR_ds_m, 'CSL_scmunumu': CSL_ds_m,
            'CSR_scenue': CSR_ds_e, 'CSL_scenue': CSL_ds_e, 
            'CSR_sutaunutau': CSR_k_t, 'CSL_sutaunutau': CSL_k_t, 
            'CSR_sumunumu': CSR_k_m, 'CSL_sumunumu': CSL_k_m, 
            'CSR_suenue': CSR_k_e, 'CSL_suenue': CSL_k_e, 
            'CSR_dutaunutau': CSR_p_t, 'CSL_dutaunutau': CSL_p_t, 
            'CSR_dumunumu': CSR_p_m, 'CSL_dumunumu': CSL_p_m, 
            'C7_bs': C7,'C7p_bs': C7p, 
            'C8_bs': C8,'C8p_bs': C8p, 
            'C9_bsee': C9_se,'C9p_bsee': C9p_se,
            'C9_bsmumu': C9_s,'C9p_bsmumu': C9p_s,
            'C10_bsee': C10_se,'C10p_bsee': C10p_se,
            'C10_bsmumu': C10_s,'C10p_bsmumu': C10p_s,'CS_bsmumu': CS_s,'CSp_bsmumu': CSp_s,'CP_bsmumu': CP_s,'CPp_bsmumu': CPp_s, # Bs->mumu
            'C10_bdmumu': C10_d,'C10p_bdmumu': C10p_d,'CS_bdmumu': CS_d,'CSp_bdmumu': CSp_d,'CP_bdmumu': CP_d,'CPp_bdmumu': CPp_d, # B0->mumu
            'CVLL_bsbs': CVLL_bs,'CVRR_bsbs': CVRR_bs,'CSLL_bsbs': CSLL_bs,'CSRR_bsbs': CSRR_bs,'CSLR_bsbs': CSLR_bs,'CVLR_bsbs': CVLR_bs, # DeltaM_s
            'CVLL_bdbd': CVLL_bd,'CVRR_bdbd': CVRR_bd,'CSLL_bdbd': CSLL_bd,'CSRR_bdbd': CSRR_bd,'CSLR_bdbd': CSLR_bd,'CVLR_bdbd': CVLR_bd, # DeltaM_d
        }, scale=4.2, eft='WET', basis='flavio')
    return FL2.log_likelihood(par,wc)

#------------------------------
#   Get Contour Data
#------------------------------

sigmas = (1,2)
#sigmas = (3,4)

#cmuon = fpl.likelihood_contour_data(muon,0,4,-1,4, n_sigma=sigmas, threads=4, steps=100) 
#cmuon = fpl.likelihood_contour_data(muon,-1,2,1,3.5, n_sigma=sigmas, threads=4, steps=100) 
cleps = fpl.likelihood_contour_data(leps,-1,2,1,3.5, n_sigma=sigmas, threads=4, steps=60) 
#cmix = fpl.likelihood_contour_data(mix,-1,2,1,3.5, n_sigma=sigmas, threads=4, steps=60) 
#crad = fpl.likelihood_contour_data(rad,-1,2,2.5,3.5, n_sigma=sigmas, threads=4, steps=60) 
#cmu = fpl.likelihood_contour_data(mu,-1,2,1.5,4, n_sigma=sigmas, threads=4, steps=60) 
#cdat = fpl.likelihood_contour_data(func,-1,2,1,3.5, n_sigma=sigmas, threads=4, steps=60) 

#------------------------------
#   Print Out Values
#------------------------------

#bf,minh,mint,maxt,minz = mHmin(cmu)
#print("Best fit value is found for (tanb,mH) =", bf)
#print("Print outs are lists for values at", sigmas, "sigmas")
#print("Minimum value of mH+ is:", minh)
#print("Minimum value of tanb is:", mint)
#print("Maximum value of tanb is:", maxt)

#------------------------------
#   p-values
#------------------------------

#chi2 = chi2_func(bf[0],bf[1],bf[1],obs_list) # mH0 = mH+
#chi2 = chi2_func(bf[0],bf[1],1500,obs_list) # mH0 = 1500 GeV
#degs = len(obs_list)-2
#pval = pvalue(chi2,degs)
#print("chi2tilde_min is:",minz)
#print("chi2_min is:",chi2)
#print("chi2_nu is:",chi2/degs)
#print("2log(L(theta)) = ",chi2-minz)
#print("p-value at chi2_min point with dof =",degs," is",pval*100,"%")

#------------------------------
#   Plotting
#------------------------------

#plt.figure(figsize=(6,5))
#fpl.contour(**cmuon,col=0) 
##plt.title(r'Anomalous magnetic moment of the muon, $a_\mu$'+'\n'+'for $m_{A^0}\sim m_{H^+}$ and $m_{H^0} = 1500\,$GeV')
#plt.title(r'Anomalous magnetic moment of the muon, $a_\mu$'+'\n'+'for $m_{H^0}\sim m_{H^+}$ and $m_{A^0} = 1500\,$GeV')
##plt.title(r'Anomalous magnetic moment of the muon, $a_\mu$'+'\n'+'for $m_{H^0},m_{A^0}\sim m_{H^+}$')
#plt.axhline(y=np.log10(1220),color='black',linestyle='--')
#plt.axhline(y=np.log10(1660),color='black',linestyle='--')
#plt.xlabel(r'$\log_{10}[\tan\beta]$') 
#plt.ylabel(r'$\log_{10}[m_{H^+} (\text{GeV})]$') 
#plt.savefig('muon_plot.png')

plt.figure(figsize=(6,5))
fpl.contour(**cleps,col=2) 
#plt.title('Tree-Level Leptonic and Semileptonic Meson Decays and Hadronic Tau Decays')
#plt.title('Tree-Level Semileptonic Meson Decays')
plt.title(r'$\mathcal{R}(D)$ and $\mathcal{R}(D^*)$')
plt.xlabel(r'$\log_{10}[\tan\beta]$') 
plt.ylabel(r'$\log_{10}[m_{H^+} (\text{GeV})]$') 
plt.savefig('rd_both.png')
quit()

#plt.figure(figsize=(6,5))
#fpl.contour(**cmix,col=0,interpolation_factor=1.1,interpolation_order=1) 
#plt.title(r'$\Delta M_{d,s}$')
#plt.xlabel(r'$\log_{10}[\tan\beta]$') 
#plt.ylabel(r'$\log_{10}[m_{H^+} (\text{GeV})]$') 
#plt.savefig('bmix_plot2.png')

#plt.figure(figsize=(12,4))
#fpl.contour(**crad,col=3) 
#plt.title(r'$\bar{B}\to X_s\gamma$ Radiative Decay')
#plt.xlabel(r'$\log_{10}[\tan\beta]$') 
#plt.ylabel(r'$\log_{10}[m_{H^+} (\text{GeV})]$') 
#plt.yticks(np.arange(2.5,3.6,0.25))
#plt.savefig('bsgamma_plot2.png')

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
#plt.title(r'FCNC Leptonic B Decays ($B_{s,d}\to\mu^+\mu^-$), $m_{H^0}\sim m_{H^+}$'+'\n'+r'$\cos(\beta-\alpha)=0.05$')
plt.title(r'FCNC Leptonic B Decays ($B_{s,d}\to\mu^+\mu^-$), $m_{H^0}=1500\,$GeV'+'\n'+r'$\cos(\beta-\alpha)=0$')
#plt.title(r'$R_K$ for $q^2\in[1,6]$ \& $R_{K^{*0}}$ for $q^2\in[0.045,6]$')
#plt.title(r'$R_K$ for $q^2\in[1,6]$')
#plt.title(r'$R_{K^{*0}}$ for $q^2\in[0.045,6]$')
#plt.title(r'$b\to sl^+l^-$ transitions ($B_{s,d}\to\mu^+\mu^-$ \& $R_K(q^2\in[1,6]),R_{K^{*0}}(q^2\in[0.045,6])$), $m_{H^0}\sim m_{H^+}$')
#plt.title(r'$b\to sl^+l^-$ transitions ($B_{s,d}\to\mu^+\mu^-$ \& $R_K(q^2\in[1,6]),R_{K^{*0}}(q^2\in[0.045,6])$), $m_{H^0}=1500\,$GeV')
plt.axhline(y=np.log10(866),color='black',linestyle='--')
plt.axhline(y=np.log10(1658),color='black',linestyle='--')
plt.xlabel(r'$\log_{10}[\tan\beta]$') 
plt.ylabel(r'$\log_{10}[m_{H^+} (\text{GeV})]$') 
#plt.savefig('bsll_plot.png')
plt.savefig('bmumu_fix_cba02.png')
#plt.savefig('rks_plot.png')
quit()

plt.figure(figsize=(6,5))
fpl.contour(**cdat,col=4) 
#plt.title(r'Combined Tree-Level Leptonics, $\Delta M_{d,s}$, $\bar{B}\to X_s\gamma$')
#plt.title(r'Combined Fit of All Observables for $m_{H^0}\sim m_{H^+}$')
#plt.title(r'Combined Fit of All Observables for $m_{H^0}=1500\,$GeV')
#plt.title(r'Combined Fit of All Observables excl. $R_{K^{(*0)}}$ for $m_{H^0}\sim m_{H^+}$')
plt.title(r'Combined Fit of All Observables excl. $R_{K^{(*0)}}$ for $m_{H^0}=1500\,$GeV')
plt.xlabel(r'$\log_{10}[\tan\beta]$') 
plt.ylabel(r'$\log_{10}[m_{H^+} (\text{GeV})]$')
#plt.savefig('comb1_plot.png')
#plt.savefig('comb2_apx_rk.png')
#plt.savefig('comb2_fix_rk.png')
#plt.savefig('comb2_apx.png')
plt.savefig('comb2_fix.png')

#plt.show()
# colours : 0 - blue, 1 - orange, 2 - green, 3 - pink, 4 - purple, 5 - brown, 6 - bright pink, 7 - grey, 8 - yellow, 9 - cyan
