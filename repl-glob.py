#!/bin/env python3
import time
start_time = time.time()
import datetime
print(datetime.datetime.now())

import flavio
from flavio.statistics.likelihood import FastLikelihood
from flavio.physics.bdecays.formfactors import b_v, b_p
from flavio.classes import Parameter, AuxiliaryQuantity, Implementation
from flavio.statistics.functions import pvalue
from flavio.config import config
import flavio.plots as fpl
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
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

# Updating Vub, Vus and Vcb from PDG 2020 Averages 
pars.set_constraint('Vus','0.2245 +- 0.0008') 
pars.set_constraint('Vcb','0.041 +- 0.0014') 
pars.set_constraint('Vub','0.00382 +- 0.00024') 

# Picking Vub to use if not average
#pars.set_constraint('Vub','0.0037 +- 0.0001 +- 0.00012') # Exclusive B->pilnu PDG
#pars.set_constraint('Vub','0.00425 +- 0.00012 + 0.00015 - 0.00014 +- 0.00023') # Inclusive B->Xulnu PDG

#### defining Bs->Ds(*)lnu

def ff_function(function, process, **kwargs):
    return lambda wc_obj, par_dict, q2: function(process, q2, par_dict, **kwargs)

config['implementation']['Bs->Ds* form factor'] = 'Bs->Ds* CLN' 
bsds = AuxiliaryQuantity('Bs->Ds* form factor') 
bsdsi = Implementation(name='Bs->Ds* CLN',quantity='Bs->Ds* form factor',function=ff_function(b_v.cln.ff,'Bs->Ds*',scale=config['renormalization scale']['bvll']))
bsdsi.set_description("CLN parameterization")

config['implementation']['Bs->Ds form factor'] = 'Bs->Ds CLN'
bsd = AuxiliaryQuantity('Bs->Ds form factor')
bsdi = Implementation(name='Bs->Ds CLN',quantity='Bs->Ds form factor',function=ff_function(b_p.cln.ff,'Bs->Ds',scale=config['renormalization scale']['bpll']))
bsdsi.set_description("CLN parameterization")

#### fitting stuff

flavio.measurements.read_file('world_avgs.yml') # read in the world averages we want to use
config['renormalization scale']['bxgamma'] = 1.74

my_obs = [
    'BR(B+->taunu)', 'BR(B+->munu)', 'BR(D+->munu)', 'BR(Ds->munu)', 'BR(Ds->taunu)', 'BR(tau->Knu)', 'BR(K+->munu)', 'BR(tau->pinu)', 'Gamma(pi+->munu)', # [:9]
    'BR(B->Xsgamma)', # [9]
    'DeltaM_d','DeltaM_s', # [10:12]
    'BR(Bs->mumu)','BR(B0->mumu)', # [12:14]
    'Rtaul(B->Dlnu)', 'Rtaul(B->D*lnu)', # [14:16]
    'BR(B0->Dlnu)', 'BR(B+->Dlnu)', 'BR(B0->D*lnu)', 'BR(B+->D*lnu)', 'BR(Bs->Dsmunu)','BR(Bs->Ds*munu)','BR(B+->rholnu)', 'BR(B0->rholnu)', 'BR(D+->Kenu)', 'BR(D+->Kmunu)', 'BR(D+->pienu)', 'BR(D+->pimunu)', 'BR(D0->Kenu)', 'BR(D0->Kmunu)', 'BR(D0->pienu)', 'BR(D0->pimunu)', 'BR(K+->pienu)', 'BR(K+->pimunu)', 'BR(KL->pienu)', 'BR(KL->pimunu)', 'BR(KS->pienu)', 'BR(KS->pimunu)', # [16:38]
    'BR(D+->taunu)', # [38]
]

obs2 = ['BR(B+->pilnu)', 'BR(B0->pilnu)',
        ("<Rmue>(B+->Kll)", 1.0, 6.0),("<Rmue>(B0->K*ll)", 0.045, 1.1),("<Rmue>(B0->K*ll)", 1.1, 6.0),'a_mu',]

obs3 = [('<dBR/dq2>(B+->Kmumu)',1.0,6.0),('<dBR/dq2>(B0->K*mumu)',1.0,6.0),]
#        ('<P5p>(B0->K*mumu)',0.1,0.98),('<P5p>(B0->K*mumu)',1.1,6.0), ('<P5p>(B0->K*mumu)',15.0,19.0),]

obs5 = ['<FL>(B0->K*mumu)','<S3>(B0->K*mumu)','<S4>(B0->K*mumu)','<S5>(B0->K*mumu)','<AFB>(B0->K*mumu)','<S7>(B0->K*mumu)','<S8>(B0->K*mumu)','<S9>(B0->K*mumu)','<P1>(B0->K*mumu)','<P2>(B0->K*mumu)','<P3>(B0->K*mumu)','<P4p>(B0->K*mumu)','<P5p>(B0->K*mumu)','<P6p>(B0->K*mumu)','<P8p>(B0->K*mumu)',]
bins1 = [0.1,1.1,2.5,4.0,1.1,6.0,11.0,15.0,17.0,15.0]
bins2 = [0.98,2.5,4.0,6.0,6.0,8.0,12.5,17.0,19.0,19.0]

ims1 = ['BKll Observables',]
ims = ['BKll Observables','LHCb B+->Kmumu BR 2014','CDF B+>Kmumu 2012','CMS B->K*mumu 2015 4.3-6 BR','CMS B->K*mumu 2013 combined with 2015 BR','LHCb B0->K*mumu BR 2016','CDF B0->K*mumu BR 2012']
ims2 = [['LHCb B+->Kmumu BR 2014','CDF B+>Kmumu 2012',],
        ['CMS B->K*mumu 2015 4.3-6 BR','CMS B->K*mumu 2013 combined with 2015 BR','LHCb B0->K*mumu BR 2016','CDF B0->K*mumu BR 2012',],['BKll Observables',],['BKll Observables',],['BKll Observables',],]

obs3_png = ['dBRdq2(B+->Kmumu)','dBRdq2(B0->K(st)mumu)',
            'P5p(B0->K(st)mumu)-1','P5p(B0->K(st)mumu)-2','P5p(B0->K(st)mumu)-3',]

obs4 = [r'$\langle\frac{dBR}{dq^2}\rangle(B^+\to K^+\mu^+\mu^-),\,q^2\in[1,6]$',
        r"$\langle\frac{dBR}{dq^2}\rangle(B_0\to K_0^*\mu^+\mu^-),\,q^2\in[1,6]$",
        r"$\langle P5'\rangle(B_0\to K_0^*\mu^+\mu^-),\,q^2\in[0.1,0.98]$",
        r"$\langle P5'\rangle(B_0\to K_0^*\mu^+\mu^-),\,q^2\in[1.1,6]$",
        r"$\langle P5'\rangle(B_0\to K_0^*\mu^+\mu^-),\,q^2\in[15,19]$",]

strings = 'B0topilnu-ExclusiveVub'
#Fleps = FastLikelihood(name="trees",observables=[my_obs[38]],include_measurements=['Tree Level Leptonics'])
#Fleps = FastLikelihood(name="trees",observables=[obs2[1]],include_measurements=['Tree Level Semileptonics',]) 
#Fleps = FastLikelihood(name="trees",observables=my_obs[:9]+my_obs[14:39]+obs2[:2],include_measurements=['Tree Level Leptonics','LFU D Ratios','Tree Level Semileptonics']) 
#Fleps = FastLikelihood(name="trees",observables=my_obs[14:16],include_measurements=['LFU D Ratios',]) 
#Fleps = FastLikelihood(name="trees",observables=[my_obs[14]],include_measurements=['LFU D Ratios',]) 
#Fleps = FastLikelihood(name="trees",observables=[my_obs[15]],include_measurements=['LFU D Ratios',]) 
#------------------------------
#Fmix = FastLikelihood(name="mix",observables=[my_obs[10]],include_measurements=['B Mixing',]) 
#Fmix = FastLikelihood(name="mix",observables=[my_obs[11]],include_measurements=['B Mixing',]) 
Fmix = FastLikelihood(name="mix",observables=my_obs[10:12],include_measurements=['B Mixing',]) 
Frad = FastLikelihood(name="rad",observables=[my_obs[9]],include_measurements=['Radiative Decays'])
#------------------------------
#Fmu = FastLikelihood(name="mu",observables=obs2[2:5],include_measurements=['LFU K Ratios']) 
#Fmu = FastLikelihood(name="mu",observables=my_obs[12:14],include_measurements=['FCNC Leptonic Decays',]) 
#Fmu = FastLikelihood(name="mu",observables=obs3)
#------------------------------
#obs_list = my_obs+obs2
#FL2 = FastLikelihood(name="glob",observables=obs_list,include_measurements=['Tree Level Leptonics','Radiative Decays','FCNC Leptonic Decays','B Mixing','LFU D Ratios','Tree Level Semileptonics','LFU K Ratios','Anomalous Magnetic Moments',])
#FL2 = FastLikelihood(name="glob",observables=obs_list,include_measurements=['Tree Level Leptonics','Radiative Decays','FCNC Leptonic Decays','B Mixing','LFU D Ratios','Tree Level Semileptonics',])
#------------------------------
#Fmuon = FastLikelihood(name="muons",observables=['a_mu'],include_measurements=['Anomalous Magnetic Moments'])
#Fmuon.make_measurement(N=500,threads=4)

#Fleps.make_measurement(N=500,threads=4)
Fmix.make_measurement(N=500,threads=4)
Frad.make_measurement(N=500,threads=4)
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
    CSR_b_m, CSL_b_m = rh(par['m_u'],par['m_b'],par['m_mu'],10**tanb,10**mH)
    CSR_b_e, CSL_b_e = rh(par['m_u'],par['m_b'],par['m_e'],10**tanb,10**mH)
#    CSR_d_t, CSL_d_t = rh(par['m_c'],par['m_d'],par['m_tau'],10**tanb,10**mH)
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
#    CSR_bc_t, CSL_bc_t = rh(par['m_c'],par['m_b'],par['m_tau'],10**tanb,10**mH)
#    CSR_bc_m, CSL_bc_m = rh(par['m_c'],par['m_b'],par['m_mu'],10**tanb,10**mH)
#    CSR_bc_e, CSL_bc_e = rh(par['m_c'],par['m_b'],par['m_e'],10**tanb,10**mH)

    wc = flavio.WilsonCoefficients()
    wc.set_initial({ # tell flavio what WCs you're referring to with your variables
#            'CSR_bctaunutau': CSR_bc_t, 'CSL_bctaunutau': CSL_bc_t,
#            'CSR_bcmunumu': CSR_bc_m, 'CSL_bcmunumu': CSL_bc_m,
#            'CSR_bcenue': CSR_bc_e, 'CSL_bcenue': CSL_bc_e, 
#            'CSR_butaunutau': CSR_b_t, 'CSL_butaunutau': CSL_b_t,
            'CSR_bumunumu': CSR_b_m, 'CSL_bumunumu': CSL_b_m,
            'CSR_buenue': CSR_b_e, 'CSL_buenue': CSL_b_e, 
#            'CSR_dctaunutau': CSR_d_t, 'CSL_dctaunutau': CSL_d_t,
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

def mu(app,wcs):
    tanb, mH = wcs # state what the two parameters are going to be on the plot
    mass, align = app
    if mass == 0:
        mH0 = mH
    elif mass == 1:
        mH0 = np.log10(1500)

    par = flavio.default_parameters.get_central_all()
    ckm_els = flavio.physics.ckm.get_ckm(par) # get out all the CKM elements

#    C9_se, C9p_se, C10_se, C10p_se, CS_se, CSp_se, CP_se, CPp_se = bsll(par,ckm_els,['m_s','m_d',1],['m_e','m_mu',1],10**mH0,10**tanb,10**mH)
    C9_s, C9p_s, C10_s, C10p_s, CS_s, CSp_s, CP_s, CPp_s = bsll(par,ckm_els,['m_s','m_d',1],['m_mu','m_e',1],10**mH0,10**tanb,10**mH,align)
    C9_d, C9p_d, C10_d, C10p_d, CS_d, CSp_d, CP_d, CPp_d = bsll(par,ckm_els,['m_d','m_s',0],['m_mu','m_e',1],10**mH0,10**tanb,10**mH,align)
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

#mu_apx_ali = partial(mu,(0,0))
#mu_fix_ali = partial(mu,(1,0))
#mu_apx_dev = partial(mu,(0,1))
#mu_fix_dev = partial(mu,(1,1))

#------------------------------
#   Anomalous moments
#------------------------------

def muon(wcs):
    tanb,mH = wcs
    
    mH0,mA0 = mH, mH
#    mH0,mA0 = np.log10(1500), mH
#    mH0,mA0 = mH, np.log10(1500)

    par = flavio.default_parameters.get_central_all()

    csev = a_mu2(par,'m_mu',10**tanb,10**mH0,10**mA0,10**mH)

    wc = flavio.WilsonCoefficients()
    wc.set_initial({'C7_mumu': csev},scale=1.0,eft='WET-3',basis='flavio')
    return Fmuon.log_likelihood(par,wc)

#------------------------------
#   All Observables
#------------------------------

def func(wcs):
    tanb, mH = wcs # state what the two parameters are going to be on the plot

    mH0,mA0 = mH, mH
#    mH0,mA0 = mH, np.log10(1500)
#    mH0,mA0 = np.log10(1500), mH

    par = flavio.default_parameters.get_central_all()
    ckm_els = flavio.physics.ckm.get_ckm(par) # get out all the CKM elements

    csev = a_mu2(par,'m_mu',10**tanb,10**mH0,10**mA0,10**mH)
    CSR_b_t, CSL_b_t = rh(par['m_u'],par['m_b'],par['m_tau'],10**tanb,10**mH)
    CSR_b_m, CSL_b_m = rh(par['m_u'],par['m_b'],par['m_mu'],10**tanb,10**mH)
    CSR_b_e, CSL_b_e = rh(par['m_u'],par['m_b'],par['m_e'],10**tanb,10**mH)
    CSR_d_t, CSL_d_t = rh(par['m_c'],par['m_d'],par['m_tau'],10**tanb,10**mH)
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
            'C7_mumu': csev,
            'CSR_bctaunutau': CSR_bc_t, 'CSL_bctaunutau': CSL_bc_t,
            'CSR_bcmunumu': CSR_bc_m, 'CSL_bcmunumu': CSL_bc_m,
            'CSR_bcenue': CSR_bc_e, 'CSL_bcenue': CSL_bc_e, 
            'CSR_butaunutau': CSR_b_t, 'CSL_butaunutau': CSL_b_t,
            'CSR_bumunumu': CSR_b_m, 'CSL_bumunumu': CSL_b_m,
            'CSR_buenue': CSR_b_e, 'CSL_buenue': CSL_b_e, 
            'CSR_dctaunutau': CSR_d_t, 'CSL_dctaunutau': CSL_d_t,
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
#sigmas = (1,2,3,4,5,6)

#cmuon = fpl.likelihood_contour_data(muon,-1,2.5,-2,4, n_sigma=sigmas, threads=4, steps=60) 
#cmuon = fpl.likelihood_contour_data(muon,0,4,-1,4, n_sigma=sigmas, threads=4, steps=60) 
#cleps = fpl.likelihood_contour_data(leps,-1,2,1.5,4, n_sigma=sigmas, threads=4, steps=100) 
cmix = fpl.likelihood_contour_data(mix,-1,2,1.5,4, n_sigma=sigmas, threads=4, steps=60) 
crad = fpl.likelihood_contour_data(rad,-1,2,1.5,4, n_sigma=sigmas, threads=4, steps=800) 
#cmu = fpl.likelihood_contour_data(mu,2.5,4,-2,1, n_sigma=sigmas, threads=4, steps=60) 
#cmu = fpl.likelihood_contour_data(mu,-1,2,0,4, n_sigma=sigmas, threads=4, steps=60) 
#cmu = fpl.likelihood_contour_data(mu,-1,2,1.5,4, n_sigma=sigmas, threads=4, steps=60) 
#cdat = fpl.likelihood_contour_data(func,-1,2,1.5,4, n_sigma=sigmas, threads=4, steps=60) 
#cdat = fpl.likelihood_contour_data(func,-1,2,2.9,3.2, n_sigma=sigmas, threads=4, steps=400) 

#------------------------------
#   Print Out Values
#------------------------------

minz_bsgam = -19.066889111832868
minz_allsim = -759.1565659885084
minz_Hsim = -757.8282739220956
minz_Asim = -756.8886543429611

bf,minh,mint,maxt,minz = mHmin(crad)#,minz_allsim)
print("Best fit value is found for (tanb,mH) =", bf)
print("Print outs are lists for values at", sigmas, "sigmas")
print("Minimum value of mH+ is:", minh)
print("Minimum value of tanb is:", mint)
print("Maximum value of tanb is:", maxt)
print(minz)
#print("--- %s seconds ---" % (time.time() - start_time))
#print(datetime.datetime.now())
#quit()

#------------------------------
#   p-values
#------------------------------

#chi2 = chi2_func(bf[0],bf[1],bf[1],bf[1],obs_list) # mH0 = mH+ = mA0
#chi2 = chi2_func(bf[0],bf[1],bf[1],1500,obs_list) # mA0 = 1500 GeV, mH0 = mH+
#chi2 = chi2_func(bf[0],bf[1],1500,bf[1],obs_list) # mH0 = 1500 GeV, mA0 = mH+
#degs = len(obs_list)-2
#pval = pvalue(chi2,degs)
#print("chi2tilde_min is:",minz)
#print("chi2_min is:",chi2)
#print("chi2_nu is:",chi2/degs)
#print("2log(L(theta)) = ",chi2-minz)
#print("p-value at chi2_min point with dof =",degs," is",pval*100,"%")
#quit()

#------------------------------
#   Plotting
#------------------------------

#plt.figure(figsize=(6,5))
#fpl.contour(**cmuon,col=6) 
##plt.title(r'$m_{A^0}\sim m_{H^+}$ and $m_{H^0} = 1500\,$GeV',fontsize=18)
##plt.title(r'$m_{H^0}\sim m_{H^+}$ and $m_{A^0} = 1500\,$GeV',fontsize=18)
#plt.title(r'$m_{H^0},m_{A^0}\sim m_{H^+}$',fontsize=18)
##plt.axhline(y=np.log10(1220),color='black',linestyle='--') # Asim = 866, Hsim = 1220
##plt.axhline(y=np.log10(1660),color='black',linestyle='--') # Asim = 1660, Hsim = 1660
#plt.xlabel(r'$\log_{10}[\tan\beta]$') 
#plt.ylabel(r'$\log_{10}[m_{H^+}/\text{GeV}]$') 
##plt.savefig('muon_Asim.png') # 1,2 sig
##plt.savefig('muon_Hsim.png') # 3.3 sig
#plt.savefig('muon_allsim.png') # 3.3 sig
##plt.show()
#quit()

#plt.figure(figsize=(6,5))
#fpl.contour(**cleps,col=2)#,interpolation_factor=1.09,interpolation_order=1) 
##plt.title('Tree-Level Leptonic and Semileptonic Meson Decays and Hadronic Tau Decays')
##plt.title('Tree-Level Semileptonic Meson Decays')
##plt.title(r'$\mathcal{R}(D)$ and $\mathcal{R}(D^*)$')
#plt.title(strings)
#plt.xlabel(r'$\log_{10}[\tan\beta]$') 
#plt.ylabel(r'$\log_{10}[m_{H^+}/\text{GeV}]$') 
#plt.savefig(strings+'.png')
##plt.savefig('qqlnu_plot.png')
#quit()

plt.figure(figsize=(6,5))
fpl.contour(**cmix,col=0,interpolation_factor=1.015,interpolation_order=1) 
##plt.title(r'$\Delta M_{d,s}$')
plt.xlabel(r'$\log_{10}[\tan\beta]$') 
plt.ylabel(r'$\log_{10}[m_{H^+}/\text{GeV}]$') 
plt.savefig('bmix_plot.png')
#plt.show()

##plt.figure(figsize=(12,4))
plt.figure(figsize=(6,5))
fpl.contour(**crad,col=3)#,z_min=minz_bsgam)
##plt.title(r'$\bar{B}\to X_s\gamma$ Radiative Decay')
plt.xlabel(r'$\log_{10}[\tan\beta]$') 
plt.ylabel(r'$\log_{10}[m_{H^+}/\text{GeV}]$') 
###plt.yticks(np.arange(2.5,3.6,0.25))
plt.savefig('bsgamma_plot2.png')

# (2.4,4.2,-2,1)
z_min1 = -5.182818890948422

mt = [r'$m_{H^0}\sim m_{H^+},\,$',r'$m_{H^0}=1500\,$GeV,$\,$']
at = [r'$\cos(\beta-\alpha)=0$',r'$\cos(\beta-\alpha)=0.05$',r'$\cos(\beta-\alpha)=-0.05$']
bmu_mass = ['apx_','fix_']
bmu_ali = ['cba02','cbap2','cbam2']
for i in range(2):
    for j in range(3):
        Fmu = FastLikelihood(name="mu",observables=my_obs[12:14],include_measurements=['FCNC Leptonic Decays',]) 
        Fmu.make_measurement(N=500,threads=4)

        mu0 = partial(mu,(i,j))
        cmu = fpl.likelihood_contour_data(mu0,-1,2,1.5,4, n_sigma=sigmas, threads=4, steps=60) 
        plt.figure(figsize=(6,5))
        fpl.contour(**cmu,col=9)
        plt.title(mt[i]+at[j],fontsize=18)
        if i == 1:
            plt.axhline(y=np.log10(866),color='black',linestyle='--')
            plt.axhline(y=np.log10(1658),color='black',linestyle='--')
        plt.xlabel(r'$\log_{10}[\tan\beta]$') 
        plt.ylabel(r'$\log_{10}[m_{H^+}/\text{GeV}]$') 
        plt.savefig('bmumu_'+bmu_mass[i]+bmu_ali[j]+'.png')

quit()

#for i in range(len(obs3)):
#    Fmu = FastLikelihood(name="mu",observables=[obs3[i]],include_measurements=ims)
#    Fmu.make_measurement(N=500,threads=4)
#
#    cmu = fpl.likelihood_contour_data(mu0,-1,2,1.5,4, n_sigma=sigmas, threads=4, steps=60) 
#    plt.figure(figsize=(6,5))
#    fpl.contour(**cmu,col=9)
#    plt.title(r'$m_{H^0}\sim m_{H^+},$'+'\n'+obs4[i],fontsize=18)
#    plt.xlabel(r'$\log_{10}[\tan\beta]$') 
#    plt.ylabel(r'$\log_{10}[m_{H^+}/\text{GeV}]$') 
#    plt.savefig(obs3_png[i]+'_apx.png')
#
#    Fmu = FastLikelihood(name="mu",observables=[obs3[i]],include_measurements=ims)
#    Fmu.make_measurement(N=500,threads=4)
#
#    cmu = fpl.likelihood_contour_data(mu1,-1,2,1.5,4, n_sigma=sigmas, threads=4, steps=60) 
#    plt.figure(figsize=(6,5))
#    fpl.contour(**cmu,col=9)
#    plt.axhline(y=np.log10(866),color='black',linestyle='--')
#    plt.axhline(y=np.log10(1658),color='black',linestyle='--')
#    plt.title(r'$m_{H^0}=1500\,$GeV,'+'\n'+obs4[i],fontsize=18)
#    plt.xlabel(r'$\log_{10}[\tan\beta]$') 
#    plt.ylabel(r'$\log_{10}[m_{H^+}/\text{GeV}]$') 
#    plt.savefig(obs3_png[i]+'_fix.png')

angle_list = []
for i in obs5:
    for j in range(len(bins1)):
        angle_list.append((i,bins1[j],bins2[j]))

print(angle_list)
print(len(angle_list))
Fmu = FastLikelihood(name="mu",observables=angle_list,include_measurements=ims1)
Fmu.make_measurement(N=500,threads=4)

cmu = fpl.likelihood_contour_data(mu0,-1,2,1.5,4, n_sigma=sigmas, threads=4, steps=60) 
plt.figure(figsize=(6,5))
fpl.contour(**cmu,col=9)#,z_min=z_min1) 
plt.title(r'$m_{H^0}\sim m_{H^+}$',fontsize=18)
plt.xlabel(r'$\log_{10}[\tan\beta]$') 
plt.ylabel(r'$\log_{10}[m_{H^+}/\text{GeV}]$') 
plt.savefig('bkell_apx2.png')

Fmu = FastLikelihood(name="mu",observables=angle_list,include_measurements=ims1)
Fmu.make_measurement(N=500,threads=4)

cmu = fpl.likelihood_contour_data(mu1,-1,2,1.5,4, n_sigma=sigmas, threads=4, steps=60) 
plt.figure(figsize=(6,5))
fpl.contour(**cmu,col=9)#,z_min=z_min1) 
plt.axhline(y=np.log10(866),color='black',linestyle='--')
plt.axhline(y=np.log10(1658),color='black',linestyle='--')
plt.title(r'$m_{H^0}=1500\,$GeV',fontsize=18)
plt.xlabel(r'$\log_{10}[\tan\beta]$') 
plt.ylabel(r'$\log_{10}[m_{H^+}/\text{GeV}]$') 
plt.savefig('bkell_fix2.png')

    #plt.title(r'$m_{H^0}\sim m_{H^+},\,\cos(\beta-\alpha)=0$',fontsize=18)
    #plt.title(r'$m_{H^0}=1500\,$GeV,$\,\cos(\beta-\alpha)=0$',fontsize=18)
    #plt.title(r'$m_{H^0}\sim m_{H^+},\,\cos(\beta-\alpha)=-0.05$',fontsize=18)
    #plt.title(r'$m_{H^0}=1500\,$GeV,$\,\cos(\beta-\alpha)=-0.05$',fontsize=18)
    #plt.title(r'$1,2\sigma$ Contours',fontsize=18)
    #plt.title(r'$3,4\sigma$ Contours',fontsize=18)
    #plt.savefig('bmumu_wsl2.png')
    #plt.savefig('bmumu_apx_cba02.png')
    #plt.savefig('bmumu_fix_cba02.png')
    #plt.savefig('bmumu_apx_cbam2.png')
    #plt.savefig('bmumu_fix_cbam2.png')
    #plt.savefig('rks_34sig.png')
    #plt.title(r'$m_{H^0}\sim m_{H^+}$, $\cos(\beta-\alpha)=\sin(2\beta)$',fontsize=18)
    #plt.title(r'$m_{H^0}=1500\,$GeV, $\cos(\beta-\alpha)=\sin(2\beta)$',fontsize=18)

quit()

plt.figure(figsize=(6,5))
fpl.contour(**cdat,col=4) 
#plt.title(r'Combined Tree-Level Leptonics, $\Delta M_{d,s}$, $\bar{B}\to X_s\gamma$')
#plt.title(r'$m_{H^0},m_{A^0}\sim m_{H^+}$',fontsize=18)
plt.title(r'$m_{H^0}\sim m_{H^+},\; m_{A^0}=1500\,$GeV',fontsize=18)
#plt.title(r'$m_{A^0}\sim m_{H^+},\; m_{H^0}=1500\,$GeV',fontsize=18)
plt.xlabel(r'$\log_{10}[\tan\beta]$') 
plt.ylabel(r'$\log_{10}[m_{H^+}/\text{GeV}]$')
plt.axhline(y=np.log10(1220),color='black',linestyle='--') # Asim = 866, Hsim = 1220
plt.axhline(y=np.log10(1660),color='black',linestyle='--') # Asim = 1660, Hsim = 1660
#plt.savefig('comb1_plot.png')
#plt.savefig('comb2_allsim.png')
plt.savefig('comb2_Hsim.png')
#plt.savefig('comb2_Asim.png')

#plt.show()
# colours : 0 - blue, 1 - orange, 2 - green, 3 - pink, 4 - purple, 5 - brown, 6 - bright pink, 7 - grey, 8 - yellow, 9 - cyan
