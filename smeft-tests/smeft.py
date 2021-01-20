#!/bin/env python3
import time
start_time = time.time()
import datetime
print(datetime.datetime.now())

import flavio
from flavio.statistics.likelihood import FastLikelihood, Likelihood
from flavio.physics.bdecays.formfactors import b_v, b_p
from flavio.classes import Parameter, AuxiliaryQuantity, Implementation, Observable, Measurement
from flavio.statistics.functions import pvalue, delta_chi2, pull
from flavio.config import config
import multiprocessing as mp
from multiprocessing import Pool
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
flavio.measurements.read_file('bkll_avgs.yml') 
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
        ("<Rmue>(B+->Kll)",1.0,6.0),("<Rmue>(B+->Kll)",1.1,6.0),
        ("<Rmue>(B0->K*ll)",0.045,1.1),("<Rmue>(B0->K*ll)",1.1,6.0),("<Rmue>(B0->K*ll)",15.0,19.0),
        ("<Rmue>(B+->K*ll)",0.045,1.1),("<Rmue>(B+->K*ll)",1.1,6.0),("<Rmue>(B+->K*ll)",15.0,19.0),
        ("<Rmue>(B->Kll)",0.1,8.12),("<Rmue>(B->K*ll)",0.1,8.12),
        'a_mu']

ims = [
       'LHCb-2012.13241 P 0.1-0.98','LHCb-2012.13241 P 1.1-2.5','LHCb-2012.13241 P 2.5-4.0',
       'LHCb-2012.13241 P 4.0-6.0','LHCb-2012.13241 P 15.0-17.0','LHCb-2012.13241 P 17.0-19.0',
       'LHCb-2003.04831 P 0.1-0.98','LHCb-2003.04831 P 1.1-2.5','LHCb-2003.04831 P 2.5-4.0',
       'LHCb-2003.04831 P 4.0-6.0','LHCb-2003.04831 P 11.0-12.5','LHCb-2003.04831 P 15.0-17.0',
       'LHCb-2003.04831 P 17.0-19.0','LHCb-2003.04831 P 1.1-6.0','LHCb-2003.04831 P 15.0-19.0',
       'Belle-1612.05014 P45',
       'LHCb-1606.04731','LHCb-1403.80441','LHCb-1506.08777 BRs',
       'LHCb-1506.08777 S 0.1-2.0','LHCb-1506.08777 S 2.0-5.0',
       'LHCb-1506.08777 S 15.0-17.0','LHCb-1506.08777 S 17.0-19.0',
       'LHCb-1503.07138','LHCb-1808.00264','LHCb-1501.03038','LHCb-1304.3035','LHCb-1406.6482',
       'ATLAS-1805.04000 S 0.04-2.0','ATLAS-1805.04000 S 2.0-4.0','ATLAS-1805.04000 S 4.0-6.0',
       'ATLAS-1805.04000 P 0.04-2.0','ATLAS-1805.04000 P 2.0-4.0','ATLAS-1805.04000 P 4.0-6.0',
       'CMS-1507.08126 1.0-2.0','CMS-1507.08126 2.0-4.3','CMS-1507.08126 4.3-6.0',
       'CMS-1507.08126 14.18-16.0','CMS-1507.08126 16.0-19.0',
       'CMS-1710.02846 P 1.0-2.0','CMS-1710.02846 P 2.0-4.3','CMS-1710.02846 P 4.3-6.0',
       'CMS-1710.02846 P 14.18-16.0','CMS-1710.02846 P 16.0-19.0',
       'CMS-1806.00636 1.0-6.0','CMS-1806.00636 16.0-18.0','CMS-1806.00636 18.0-22.0',
       'CDF 0.0-2.0','CDF 2.0-4.3','BaBar-1312.5364 Xs','BaBar-1204.3933 RKs',
       'Belle-1908.01848','Belle-1908.01848 RKs','Belle-1904.02440','Belle-1904.02440 RKs 1','Belle-1904.02440 RKs 2']

obs6 = [('<P4p>(B->K*mumu)',1.0,6.0),('<P4p>(B->K*mumu)',14.18,19.0),
        ('<P5p>(B->K*mumu)',1.0,6.0),('<P5p>(B->K*mumu)',14.18,19.0),
        ('<P4p>(B->K*ee)',1.0,6.0),('<P4p>(B->K*ee)',14.18,19.0),
        ('<P5p>(B->K*ee)',1.0,6.0),('<P5p>(B->K*ee)',14.18,19.0),
        ('<FL>(B0->K*ee)',0.002,1.12),#('<dBR/dq2>(B0->K*ee)',0.003,1.0),
        ('<BR>(B->Xsmumu)',14.2,25.0),('<BR>(B->Xsmumu)',1.0,6.0),
        ('<BR>(B->Xsee)',14.2,25.0),('<BR>(B->Xsee)',0.1,2.0),('<BR>(B->Xsee)',2.0,4.3),('<BR>(B->Xsee)',4.3,6.8)]

obs7 = [
        ('<dBR/dq2>(B0->K*mumu)',1.0,2.0),('<dBR/dq2>(B0->K*mumu)',2.0,4.3),('<dBR/dq2>(B0->K*mumu)',4.3,6.0),
        ('<dBR/dq2>(B0->K*mumu)',14.18,16.0),('<dBR/dq2>(B0->K*mumu)',16.0,19.0),
        ('<AFB>(B0->K*mumu)',1.0,2.0),('<AFB>(B0->K*mumu)',2.0,4.3),('<AFB>(B0->K*mumu)',4.3,6.0),
        ('<AFB>(B0->K*mumu)',14.18,16.0),('<AFB>(B0->K*mumu)',16.0,19.0),
        ('<FL>(B0->K*mumu)',1.0,2.0),('<FL>(B0->K*mumu)',14.18,16.0),('<FL>(B0->K*mumu)',0.04,2.0),
        ('<FL>(B0->K*mumu)',0.1,0.98),('<FL>(B0->K*mumu)',1.1,2.5),('<FL>(B0->K*mumu)',2.5,4.0),
        ('<FL>(B0->K*mumu)',4.0,6.0),('<FL>(B0->K*mumu)',15.0,17.0),('<FL>(B0->K*mumu)',17.0,19.0),
        ('<P1>(B0->K*mumu)',1.0,2.0),('<P1>(B0->K*mumu)',14.18,16.0),('<P1>(B0->K*mumu)',0.04,2.0),
        ('<P1>(B0->K*mumu)',0.1,0.98),('<P1>(B0->K*mumu)',1.1,2.5),('<P1>(B0->K*mumu)',2.5,4.0),
        ('<P1>(B0->K*mumu)',4.0,6.0),('<P1>(B0->K*mumu)',15.0,17.0),('<P1>(B0->K*mumu)',17.0,19.0),
        ('<P2>(B0->K*mumu)',0.1,0.98),('<P2>(B0->K*mumu)',1.1,2.5),('<P2>(B0->K*mumu)',2.5,4.0),
        ('<P2>(B0->K*mumu)',4.0,6.0),('<P2>(B0->K*mumu)',15.0,17.0),('<P2>(B0->K*mumu)',17.0,19.0),
        ('<P3>(B0->K*mumu)',0.1,0.98),('<P3>(B0->K*mumu)',1.1,2.5),('<P3>(B0->K*mumu)',2.5,4.0),
        ('<P3>(B0->K*mumu)',4.0,6.0),('<P3>(B0->K*mumu)',15.0,17.0),('<P3>(B0->K*mumu)',17.0,19.0),
        ('<P4p>(B0->K*mumu)',0.04,2.0),('<P4p>(B0->K*mumu)',2.0,4.0),
        ('<P4p>(B0->K*mumu)',0.1,0.98),('<P4p>(B0->K*mumu)',1.1,2.5),#('<P4p>(B0->K*mumu)',2.5,4.0),
        ('<P4p>(B0->K*mumu)',4.0,6.0),('<P4p>(B0->K*mumu)',15.0,17.0),('<P4p>(B0->K*mumu)',17.0,19.0),
        ('<P5p>(B0->K*mumu)',1.0,2.0),#('<P5p>(B0->K*mumu)',2.0,4.3),('<P5p>(B0->K*mumu)',4.3,6.0),
        ('<P5p>(B0->K*mumu)',14.18,16.0),('<P5p>(B0->K*mumu)',0.04,2.0),('<P5p>(B0->K*mumu)',2.0,4.0),
        ('<P5p>(B0->K*mumu)',0.1,0.98),('<P5p>(B0->K*mumu)',1.1,2.5),#('<P5p>(B0->K*mumu)',2.5,4.0),
        ('<P5p>(B0->K*mumu)',4.0,6.0),('<P5p>(B0->K*mumu)',15.0,17.0),('<P5p>(B0->K*mumu)',17.0,19.0),
        ('<P6p>(B0->K*mumu)',0.04,2.0),('<P6p>(B0->K*mumu)',2.0,4.0),
        ('<P6p>(B0->K*mumu)',0.1,0.98),('<P6p>(B0->K*mumu)',1.1,2.5),#('<P6p>(B0->K*mumu)',2.5,4.0),
        ('<P6p>(B0->K*mumu)',4.0,6.0),('<P6p>(B0->K*mumu)',15.0,17.0),('<P6p>(B0->K*mumu)',17.0,19.0),
        ('<P8p>(B0->K*mumu)',0.04,2.0),('<P8p>(B0->K*mumu)',2.0,4.0),
        ('<P8p>(B0->K*mumu)',0.1,0.98),('<P8p>(B0->K*mumu)',1.1,2.5),#('<P8p>(B0->K*mumu)',2.5,4.0),
        ('<P8p>(B0->K*mumu)',4.0,6.0),('<P8p>(B0->K*mumu)',15.0,17.0),('<P8p>(B0->K*mumu)',17.0,19.0),
        ('<dBR/dq2>(B0->Kmumu)',0.1,2.0),('<dBR/dq2>(B0->Kmumu)',2.0,4.0),('<dBR/dq2>(B0->Kmumu)',4.0,6.0),
        ('<dBR/dq2>(B0->Kmumu)',15.0,17.0),('<dBR/dq2>(B0->Kmumu)',17.0,19.0),
        ('<dBR/dq2>(B+->Kmumu)',21.0,22.0),('<dBR/dq2>(B+->Kmumu)',4.0,8.12),
        ('<dBR/dq2>(B+->Kmumu)',0.1,0.98),('<dBR/dq2>(B+->Kmumu)',1.1,2.0),('<dBR/dq2>(B+->Kmumu)',2.0,3.0),
        ('<dBR/dq2>(B+->Kmumu)',3.0,4.0),('<dBR/dq2>(B+->Kmumu)',4.0,5.0),('<dBR/dq2>(B+->Kmumu)',5.0,6.0),
        ('<dBR/dq2>(B+->Kmumu)',15.0,16.0),('<dBR/dq2>(B+->Kmumu)',16.0,17.0),('<dBR/dq2>(B+->Kmumu)',17.0,18.0),
        ('<dBR/dq2>(B+->Kmumu)',18.0,19.0),('<dBR/dq2>(B+->Kmumu)',19.0,20.0),('<dBR/dq2>(B+->Kmumu)',20.0,21.0),
        ('<AFB>(B+->Kmumu)',1.0,6.0),('<AFB>(B+->Kmumu)',16.0,18.0),('<AFB>(B+->Kmumu)',18.0,22.0),
        ('<FH>(B+->Kmumu)',1.0,6.0),('<FH>(B+->Kmumu)',16.0,18.0),('<FH>(B+->Kmumu)',18.0,22.0),
        ('<dBR/dq2>(B+->K*mumu)',0.1,2.0),('<dBR/dq2>(B+->K*mumu)',2.0,4.0),('<dBR/dq2>(B+->K*mumu)',4.0,6.0),
        ('<dBR/dq2>(B+->K*mumu)',15.0,17.0),('<dBR/dq2>(B+->K*mumu)',17.0,19.0),
        ('<FL>(B+->K*mumu)',0.1,0.98),('<FL>(B+->K*mumu)',1.1,2.5),('<FL>(B+->K*mumu)',2.5,4.0),
        ('<FL>(B+->K*mumu)',4.0,6.0),('<FL>(B+->K*mumu)',15.0,17.0),('<FL>(B+->K*mumu)',17.0,19.0),
        ('<P1>(B+->K*mumu)',0.1,0.98),('<P1>(B+->K*mumu)',1.1,2.5),('<P1>(B+->K*mumu)',2.5,4.0),
        ('<P1>(B+->K*mumu)',4.0,6.0),('<P1>(B+->K*mumu)',15.0,17.0),('<P1>(B+->K*mumu)',17.0,19.0),
        ('<P2>(B+->K*mumu)',0.1,0.98),('<P2>(B+->K*mumu)',1.1,2.5),('<P2>(B+->K*mumu)',2.5,4.0),
        ('<P2>(B+->K*mumu)',4.0,6.0),('<P2>(B+->K*mumu)',15.0,17.0),('<P2>(B+->K*mumu)',17.0,19.0),
        ('<P3>(B+->K*mumu)',0.1,0.98),('<P3>(B+->K*mumu)',1.1,2.5),('<P3>(B+->K*mumu)',2.5,4.0),
        ('<P3>(B+->K*mumu)',4.0,6.0),('<P3>(B+->K*mumu)',15.0,17.0),('<P3>(B+->K*mumu)',17.0,19.0),
        ('<P4p>(B+->K*mumu)',0.1,0.98),('<P4p>(B+->K*mumu)',1.1,2.5),('<P4p>(B+->K*mumu)',2.5,4.0),
        ('<P4p>(B+->K*mumu)',4.0,6.0),('<P4p>(B+->K*mumu)',15.0,17.0),('<P4p>(B+->K*mumu)',17.0,19.0),
        ('<P5p>(B+->K*mumu)',0.1,0.98),('<P5p>(B+->K*mumu)',1.1,2.5),('<P5p>(B+->K*mumu)',2.5,4.0),
        ('<P5p>(B+->K*mumu)',4.0,6.0),('<P5p>(B+->K*mumu)',15.0,17.0),('<P5p>(B+->K*mumu)',17.0,19.0),
        ('<P6p>(B+->K*mumu)',0.1,0.98),('<P6p>(B+->K*mumu)',1.1,2.5),('<P6p>(B+->K*mumu)',2.5,4.0),
        ('<P6p>(B+->K*mumu)',4.0,6.0),('<P6p>(B+->K*mumu)',15.0,17.0),('<P6p>(B+->K*mumu)',17.0,19.0),
        ('<P8p>(B+->K*mumu)',0.1,0.98),('<P8p>(B+->K*mumu)',1.1,2.5),('<P8p>(B+->K*mumu)',2.5,4.0),
        ('<P8p>(B+->K*mumu)',4.0,6.0),('<P8p>(B+->K*mumu)',15.0,17.0),('<P8p>(B+->K*mumu)',17.0,19.0)]

obs8 = [('<dBR/dq2>(Bs->phimumu)',0.1,2.0),('<dBR/dq2>(Bs->phimumu)',2.0,5.0),
        ('<dBR/dq2>(Bs->phimumu)',15.0,17.0),('<dBR/dq2>(Bs->phimumu)',17.0,19.0),
        ('<FL>(Bs->phimumu)',0.1,2.0),('<FL>(Bs->phimumu)',2.0,5.0),('<FL>(Bs->phimumu)',15.0,17.0),('<FL>(Bs->phimumu)',17.0,19.0),
        ('<S3>(Bs->phimumu)',0.1,2.0),('<S3>(Bs->phimumu)',2.0,5.0),('<S3>(Bs->phimumu)',15.0,17.0),('<S3>(Bs->phimumu)',17.0,19.0),
        ('<S4>(Bs->phimumu)',0.1,2.0),('<S4>(Bs->phimumu)',2.0,5.0),('<S4>(Bs->phimumu)',15.0,17.0),('<S4>(Bs->phimumu)',17.0,19.0),
        ('<S7>(Bs->phimumu)',0.1,2.0),('<S7>(Bs->phimumu)',2.0,5.0),('<S7>(Bs->phimumu)',15.0,17.0),('<S7>(Bs->phimumu)',17.0,19.0),
        ('<dBR/dq2>(Lambdab->Lambdamumu)',1.1,6.0),('<dBR/dq2>(Lambdab->Lambdamumu)',15.0,16.0),
        ('<dBR/dq2>(Lambdab->Lambdamumu)',16.0,18.0),('<dBR/dq2>(Lambdab->Lambdamumu)',18.0,20.0),
        ('<AFBh>(Lambdab->Lambdamumu)',15.0,20.0),('<AFBl>(Lambdab->Lambdamumu)',15.0,20.0),
        ('<AFBlh>(Lambdab->Lambdamumu)',15.0,20.0)]


angle_list = obs7 + obs8 + obs6 #+ obs2[2:-1]

obs_list = my_obs+obs2[:2]#+angle_list
FL2 = FastLikelihood(name="glob",observables=obs_list,include_measurements=['Tree Level Leptonics','Radiative Decays','FCNC Leptonic Decays','B Mixing','LFU D Ratios','Tree Level Semileptonics',])
FL2.make_measurement(N=500,threads=4)

def func(app,wcs):
    tanb, mH = wcs # state what the two parameters are going to be on the plot

    if app == 0:
        mH0,mA0 = mH, mH
    elif app == 2:
        mH0,mA0 = mH, np.log10(1500)
    elif app == 1:
        mH0,mA0 = np.log10(1500), mH

    par = flavio.default_parameters.get_central_all()
    ckm_els = flavio.physics.ckm.get_ckm(par) # get out all the CKM elements

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

    C9_sde, C9p_sde, C10_sde, C10p_sde, CS_sde, CSp_sde, CP_sde, CPp_sde = bsll(par,ckm_els,1,0,0,10**mH0,10**tanb,10**mH,0)
    C9_sd, C9p_sd, C10_sd, C10p_sd, CS_sd, CSp_sd, CP_sd, CPp_sd = bsll(par,ckm_els,1,0,1,10**mH0,10**tanb,10**mH,0)
    C9_sdt, C9p_sdt, C10_sdt, C10p_sdt, CS_sdt, CSp_sdt, CP_sdt, CPp_sdt = bsll(par,ckm_els,1,0,2,10**mH0,10**tanb,10**mH,0)

    C9_bde, C9p_bde, C10_bde, C10p_bde, CS_bde, CSp_bde, CP_bde, CPp_bde = bsll(par,ckm_els,2,0,0,10**mH0,10**tanb,10**mH,0)
    C9_bd, C9p_bd, C10_bd, C10p_bd, CS_bd, CSp_bd, CP_bd, CPp_bd = bsll(par,ckm_els,2,0,1,10**mH0,10**tanb,10**mH,0)
    C9_bdt, C9p_bdt, C10_bdt, C10p_bdt, CS_bdt, CSp_bdt, CP_bdt, CPp_bdt = bsll(par,ckm_els,2,0,2,10**mH0,10**tanb,10**mH,0)

    C9_bse, C9p_bse, C10_bse, C10p_bse, CS_bse, CSp_bse, CP_bse, CPp_bse = bsll(par,ckm_els,2,1,0,10**mH0,10**tanb,10**mH,0)
    C9_bs, C9p_bs, C10_bs, C10p_bs, CS_bs, CSp_bs, CP_bs, CPp_bs = bsll(par,ckm_els,2,1,1,10**mH0,10**tanb,10**mH,0)
    C9_bst, C9p_bst, C10_bst, C10p_bst, CS_bst, CSp_bst, CP_bst, CPp_bst = bsll(par,ckm_els,2,1,2,10**mH0,10**tanb,10**mH,0)

    CVLL_bs, CVRR_bs, CSLL_bs, CSRR_bs, CSLR_bs, CVLR_bs = mixing(par,ckm_els,['m_s',1,'m_d'],10**tanb,10**mH)
    CVLL_bd, CVRR_bd, CSLL_bd, CSRR_bd, CSLR_bd, CVLR_bd = mixing(par,ckm_els,['m_d',0,'m_s'],10**tanb,10**mH)

    wc1 = flavio.WilsonCoefficients()
    wc1.set_initial({ # tell flavio what WCs you're referring to with your variables
            'CSR_bctaunutau': CSR_bc_t, 'CSL_bctaunutau': CSL_bc_t,'CSR_bcmunumu': CSR_bc_m, 'CSL_bcmunumu': CSL_bc_m,
            'CSR_bcenue': CSR_bc_e, 'CSL_bcenue': CSL_bc_e,'CSR_butaunutau': CSR_b_t, 'CSL_butaunutau': CSL_b_t,
            'CSR_bumunumu': CSR_b_m, 'CSL_bumunumu': CSL_b_m,'CSR_buenue': CSR_b_e, 'CSL_buenue': CSL_b_e, 
            'CSR_dctaunutau': CSR_d_t, 'CSL_dctaunutau': CSL_d_t,'CSR_dcmunumu': CSR_d_m, 'CSL_dcmunumu': CSL_d_m,
            'CSR_dcenue': CSR_d_e, 'CSL_dcenue': CSL_d_e,'CSR_sctaunutau': CSR_ds_t, 'CSL_sctaunutau': CSL_ds_t,
            'CSR_scmunumu': CSR_ds_m, 'CSL_scmunumu': CSL_ds_m,'CSR_scenue': CSR_ds_e, 'CSL_scenue': CSL_ds_e, 
            'CSR_sutaunutau': CSR_k_t, 'CSL_sutaunutau': CSL_k_t,'CSR_sumunumu': CSR_k_m, 'CSL_sumunumu': CSL_k_m, 
            'CSR_suenue': CSR_k_e, 'CSL_suenue': CSL_k_e,'CSR_dutaunutau': CSR_p_t, 'CSL_dutaunutau': CSL_p_t, 
            'CSR_dumunumu': CSR_p_m, 'CSL_dumunumu': CSL_p_m,'C7_bs': C7,'C7p_bs': C7p,'C8_bs': C8,'C8p_bs': C8p, 
            'C9_sdee': C9_sde,'C9p_sdee': C9p_sde, #sdll
            'C10_sdee': C10_sde,'C10p_sdee': C10p_sde,'CS_sdee': CS_sde,'CSp_sdee': CSp_sde,'CP_sdee': CP_sde,'CPp_sdee': CPp_sde, 
            'C9_sdmumu': C9_sd,'C9p_sdmumu': C9p_sd,
            'C10_sdmumu': C10_sd,'C10p_sdmumu': C10p_sd,'CS_sdmumu': CS_sd,'CSp_sdmumu': CSp_sd,'CP_sdmumu': CP_sd,'CPp_sdmumu': CPp_sd,
            'C9_sdtautau': C9_sdt,'C9p_sdtautau': C9p_sdt,'C10_sdtautau': C10_sdt,
            'C10p_sdtautau': C10p_sdt,'CS_sdtautau': CS_sdt,'CSp_sdtautau': CSp_sdt,'CP_sdtautau': CP_sdt,'CPp_sdtautau': CPp_sdt,
            'C9_bsee': C9_bse,'C9p_bsee': C9p_bse, #bsll
            'C10_bsee': C10_bse,'C10p_bsee': C10p_bse,'CS_bsee': CS_bse,'CSp_bsee': CSp_bse,'CP_bsee': CP_bse,'CPp_bsee': CPp_bse, 
            'C9_bsmumu': C9_bs,'C9p_bsmumu': C9p_bs,
            'C10_bsmumu': C10_bs,'C10p_bsmumu': C10p_bs,'CS_bsmumu': CS_bs,'CSp_bsmumu': CSp_bs,'CP_bsmumu': CP_bs,'CPp_bsmumu': CPp_bs,
            'C9_bstautau': C9_bst,'C9p_bstautau': C9p_bst,'C10_bstautau': C10_bst,
            'C10p_bstautau': C10p_bst,'CS_bstautau': CS_bst,'CSp_bstautau': CSp_bst,'CP_bstautau': CP_bst,'CPp_bstautau': CPp_bst,
            'C9_bdee': C9_bde,'C9p_bdee': C9p_bde, #bdll
            'C10_bdee': C10_bde,'C10p_bdee': C10p_bde,'CS_bdee': CS_bde,'CSp_bdee': CSp_bde,'CP_bdee': CP_bde,'CPp_bdee': CPp_bde, 
            'C9_bdmumu': C9_bd,'C9p_bdmumu': C9p_bd,
            'C10_bdmumu': C10_bd,'C10p_bdmumu': C10p_bd,'CS_bdmumu': CS_bd,'CSp_bdmumu': CSp_bd,'CP_bdmumu': CP_bd,'CPp_bdmumu': CPp_bd,
            'C9_bdtautau': C9_bdt,'C9p_bdtautau': C9p_bdt,'C10_bdtautau': C10_bdt,
            'C10p_bdtautau': C10p_bdt,'CS_bdtautau': CS_bdt,'CSp_bdtautau': CSp_bdt,'CP_bdtautau': CP_bdt,'CPp_bdtautau': CPp_bdt,
            'CVLL_bsbs': CVLL_bs,'CVRR_bsbs': CVRR_bs,'CSLL_bsbs': CSLL_bs,'CSRR_bsbs': CSRR_bs,'CSLR_bsbs': CSLR_bs,'CVLR_bsbs': CVLR_bs,
            'CVLL_bdbd': CVLL_bd,'CVRR_bdbd': CVRR_bd,'CSLL_bdbd': CSLL_bd,'CSRR_bdbd': CSRR_bd,'CSLR_bdbd': CSLR_bd,'CVLR_bdbd': CVLR_bd,
        }, scale=4.2, eft='WET', basis='flavio')
    wc2 = wc1.get_wc(sector='all',scale=125,par=par,eft='WET',basis='flavio')

    wc3 = flavio.WilsonCoefficients()
    wc3.set_initial({
        'ledq_2231': wc2['CSR_bumunumu']+wc2['CSR_bumunumu']+wc2['CS_bdmumu']+wc2['CSp_bdmumu']-wc2['CP_bdmumu']-wc2['CPp_bdmumu'], 
        'ledq_2232': wc2['CSR_bcmunumu']+wc2['CSR_bcmunumu']+wc2['CS_bsmumu']+wc2['CSp_bsmumu']-wc2['CP_bsmumu']-wc2['CPp_bsmumu'], 
        'ledq_3331': wc2['CSR_butaunutau']+wc2['CSR_butaunutau']+wc2['CS_bdtautau']+wc2['CSp_bdtautau']-wc2['CP_bdtautau']-wc2['CPp_bdtautau'], 
        'ledq_1132': wc2['CSR_bcenue']+wc2['CSR_bcenue']+wc2['CS_bsee']+wc2['CSp_bsee']-wc2['CP_bsee']-wc2['CPp_bsee'], 
        'ledq_3332': wc2['CSR_bctaunutau']+wc2['CSR_bctaunutau']+wc2['CS_bstautau']+wc2['CSp_bstautau']-wc2['CP_bstautau']-wc2['CPp_bstautau'], 
        'ledq_1131': wc2['CSR_buenue']+wc2['CSR_buenue']+wc2['CS_bdee']+wc2['CSp_bdee']-wc2['CP_bdee']-wc2['CPp_bdee'], 
        'ledq_1112': wc2['CSR_dcenue']+wc2['CSR_dcenue']+wc2['CS_sdee']+wc2['CSp_sdee']-wc2['CP_sdee']-wc2['CPp_sdee'], 
        'ledq_2212': wc2['CSR_dcmunumu']+wc2['CSR_dcmunumu']+wc2['CS_sdmumu']+wc2['CSp_sdmumu']-wc2['CP_sdmumu']-wc2['CPp_sdmumu'], 
        'ledq_3312': wc2['CSR_dctaunutau']+wc2['CSR_dctaunutau']+wc2['CS_sdtautau']+wc2['CSp_sdtautau']-wc2['CP_sdtautau']-wc2['CPp_sdtautau'], 
        'ledq_1122': wc2['CSR_scenue']+wc2['CSR_scenue'],
        'ledq_2222': wc2['CSR_scmunumu']+wc2['CSR_scmunumu'],
        'ledq_3322': wc2['CSR_sctaunutau']+wc2['CSR_sctaunutau'],
        'ledq_1121': wc2['CSR_suenue']+wc2['CSR_suenue']+wc2['CS_sdee']+wc2['CSp_sdee']-wc2['CP_sdee']-wc2['CPp_sdee'], 
        'ledq_2221': wc2['CSR_sumunumu']+wc2['CSR_sumunumu']+wc2['CS_sdmumu']+wc2['CSp_sdmumu']-wc2['CP_sdmumu']-wc2['CPp_sdmumu'], 
        'ledq_3321': wc2['CSR_sutaunutau']+wc2['CSR_sutaunutau']+wc2['CS_sdtautau']+wc2['CSp_sdtautau']-wc2['CP_sdtautau']-wc2['CPp_sdtautau'], 
        'ledq_2211': wc2['CSR_dumunumu']+wc2['CSR_dumunumu'],
        'ledq_3311': wc2['CSR_dutaunutau']+wc2['CSR_dutaunutau'],
        'lq1_2223': wc2['C9_bsmumu']+wc2['C9p_bsmumu'],'lq1_2213': wc2['C9_bdmumu']+wc2['C9p_bdmumu'],
        'qe_2322': wc2['C10_bsmumu']+wc2['C10p_bsmumu'],'qe_1322': wc2['C10_bdmumu']+wc2['C10p_bdmumu'],
        'dB_23': wc2['C7_bs']+wc2['C7p_bs'],'dB_13': wc2['C7_bd']+wc2['C7p_bd'],
        'dG_23': wc2['C8_bs']+wc2['C8p_bs'],'dG_13': wc2['C8_bd']+wc2['C8p_bd'],
        },scale=125,eft='SMEFT',basis='Warsaw')
    
    return FL2.log_likelihood(par,wc3)

#------------------------------
#   Get Contour Data
#------------------------------

sigmas = (1,2)
#sigmas = (1,2,3,4,5,6)

globo = partial(func,0) 
cdat = fpl.likelihood_contour_data(globo,-1,2,2.5,6, n_sigma=sigmas, threads=4, steps=60) 

#------------------------------
#   Print Out Values
#------------------------------

#app = 0#,1,2
#pval_func(cdat,app,obs_list)
#quit()

#------------------------------
#   Plotting
#------------------------------

plt.figure(figsize=(6,5))
fpl.contour(**cdat,col=4) 
plt.xlabel(r'$\log_{10}[\tan\beta]$') 
plt.ylabel(r'$\log_{10}[m_{H^+}/\text{GeV}]$')
plt.title(r'$m_{H^0}\sim m_{H^+}$',fontsize=18)
plt.savefig('smeft_test1.pdf')

quit()
