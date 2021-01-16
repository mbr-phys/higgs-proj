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

#np.seterr(divide='raise',over='raise',invalid='raise') 

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

#par = flavio.default_parameters.get_central_all()
#ckm_els = flavio.physics.ckm.get_ckm(par) # get out all the CKM elements
#mH0, mH = 3200, 3200
#tanb = 10
#C9_s, C9p_s, C10_s, C10p_s, CS_s, CSp_s, CP_s, CPp_s = bsll(par,ckm_els,['m_s','m_d',1],['m_mu','m_e',1],mH0,tanb,mH,0)
#C9_d, C9p_d, C10_d, C10p_d, CS_d, CSp_d, CP_d, CPp_d = bsll(par,ckm_els,['m_d','m_s',0],['m_mu','m_e',1],mH0,tanb,mH,0)
#print("For mH0 = mH+ = 3200 GeV, tanb = 10:")
#print()
#print("C9_bsmumu:",C9_s)
#print()
#print("C9'_bsmumu:",C9p_s)
#print()
#print("C10_bsmumu:",C10_s)
#print()
#print("C10'_bsmumu:",C10p_s)
#print()
#print("CS_bsmumu:",CS_s)
#print()
#print("CS'_bsmumu:",CSp_s)
#print()
#print("CP_bsmumu:",CP_s)
#print()
#print("CP'_bsmumu:",CPp_s)
#print()
#print("C9_bdmumu:",C9_d)
#print()
#print("C9'_bdmumu:",C9p_d)
#print()
#print("C10_bdmumu:",C10_d)
#print()
#print("C10'_bdmumu:",C10p_d)
#print()
#print("CS_bdmumu:",CS_d)
#print()
#print("CS'_bdmumu:",CSp_d)
#print()
#print("CP_bdmumu:",CP_d)
#print()
#print("CP'_bdmumu:",CPp_d)
#quit()

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
        'a_mu']

strings = 'B0topilnu'
#Fleps = FastLikelihood(observables=my_obs[:9]+[my_obs[38]],include_measurements=['Tree Level Leptonics'])
#Fleps = FastLikelihood(name="trees",observables=[obs2[1]],include_measurements=['Tree Level Semileptonics',])

ims = [
#       'LHCb-2003.04831 S 0.1-0.98','LHCb-2003.04831 S 1.1-2.5','LHCb-2003.04831 S 2.5-4.0',
#       'LHCb-2003.04831 S 4.0-6.0','LHCb-2003.04831 S 11.0-12.5','LHCb-2003.04831 S 15.0-17.0',
#       'LHCb-2003.04831 S 17.0-19.0','LHCb-2003.04831 S 1.1-6.0','LHCb-2003.04831 S 15.0-19.0',
       'LHCb-2012.13241 P 0.1-0.98','LHCb-2012.13241 P 1.1-2.5','LHCb-2012.13241 P 2.5-4.0',
       'LHCb-2012.13241 P 4.0-6.0','LHCb-2012.13241 P 15.0-17.0','LHCb-2012.13241 P 17.0-19.0',
       'LHCb-2003.04831 P 0.1-0.98','LHCb-2003.04831 P 1.1-2.5','LHCb-2003.04831 P 2.5-4.0',
       'LHCb-2003.04831 P 4.0-6.0','LHCb-2003.04831 P 11.0-12.5','LHCb-2003.04831 P 15.0-17.0',
       'LHCb-2003.04831 P 17.0-19.0','LHCb-2003.04831 P 1.1-6.0','LHCb-2003.04831 P 15.0-19.0',
#       'LHCb-1512.04442 S 1.1-2.5','LHCb-1512.04442 S 2.5-4.0',
#       'LHCb-1512.04442 S 4.0-6.0','LHCb-1512.04442 S 15.0-19.0',
#       'Belle-1612.05014 P45',
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
       'CDF 0.0-2.0','CDF 2.0-4.3','BaBar-1312.5364 Xs',
       'Belle-1908.01848','Belle-1908.01848 RKs','Belle-1904.02440','Belle-1904.02440 RKs 1','Belle-1904.02440 RKs 2',
       'Belle B->K*mumu LFU 2016']

obs6 = [#('<dBR/dq2>(B+->Kee)',1.1,6.0),('<dBR/dq2>(B+->Kee)',1.0,6.0),('<dBR/dq2>(B+->Kee)',0.1,4.0),
        #('<dBR/dq2>(B+->Kee)',4.0,8.12),
        #('<dBR/dq2>(B0->Kee)',1.0,6.0),('<dBR/dq2>(B0->Kee)',0.1,4.0),('<dBR/dq2>(B0->Kee)',4.0,8.12),
        #('<P4p>(B0->K*ee)',1.0,4.0),('<P4p>(B0->K*ee)',14.18,19.0),
        #('<P5p>(B0->K*ee)',1.0,4.0),('<P5p>(B0->K*ee)',14.18,19.0),
        ('<Dmue_P4p>(B0->K*ll)',0.1,4.0),('<Dmue_P4p>(B0->K*ll)',4.0,8.0),
        ('<Dmue_P4p>(B0->K*ll)',1.0,6.0),('<Dmue_P4p>(B0->K*ll)',14.18,19.0),
        ('<Dmue_P5p>(B0->K*ll)',0.1,4.0),('<Dmue_P5p>(B0->K*ll)',4.0,8.0),
        ('<Dmue_P5p>(B0->K*ll)',1.0,6.0),('<Dmue_P5p>(B0->K*ll)',14.18,19.0),
        ('<FL>(B0->K*ee)',0.002,1.12),#('<dBR/dq2>(B0->K*ee)',0.003,1.0),
        ('<BR>(B->Xsmumu)',14.2,25.0),('<BR>(B->Xsmumu)',0.1,2.0),#('<BR>(B->Xsmumu)',2.0,4.3),
        ('<BR>(B->Xsmumu)',4.3,6.8),
        ('<BR>(B->Xsee)',14.2,25.0),('<BR>(B->Xsee)',0.1,2.0),('<BR>(B->Xsee)',2.0,4.3),('<BR>(B->Xsee)',4.3,6.8)]

obs7 = [
#        ('<dBR/dq2>(B0->K*mumu)',1.0,2.0),('<dBR/dq2>(B0->K*mumu)',2.0,4.3),('<dBR/dq2>(B0->K*mumu)',4.3,6.0),
#        ('<dBR/dq2>(B0->K*mumu)',14.18,16.0),('<dBR/dq2>(B0->K*mumu)',16.0,19.0),
        ('<dBR/dq2>(B0->K*mumu)',1.1,6.0),('<dBR/dq2>(B0->K*mumu)',15.0,19.0),
        ('<AFB>(B0->K*mumu)',1.0,2.0),('<AFB>(B0->K*mumu)',2.0,4.3),('<AFB>(B0->K*mumu)',4.3,6.0),
        ('<AFB>(B0->K*mumu)',14.18,16.0),('<AFB>(B0->K*mumu)',16.0,19.0),
        ('<FL>(B0->K*mumu)',1.0,2.0),('<FL>(B0->K*mumu)',2.0,4.3),('<FL>(B0->K*mumu)',4.3,6.0),
        ('<FL>(B0->K*mumu)',14.18,16.0),('<FL>(B0->K*mumu)',16.0,19.0),
        ('<FL>(B0->K*mumu)',0.04,2.0),('<FL>(B0->K*mumu)',2.0,4.0),('<FL>(B0->K*mumu)',4.0,6.0),
        ('<FL>(B0->K*mumu)',0.1,0.98),('<FL>(B0->K*mumu)',1.1,2.5),('<FL>(B0->K*mumu)',2.5,4.0),
        ('<FL>(B0->K*mumu)',4.0,6.0),('<FL>(B0->K*mumu)',15.0,17.0),('<FL>(B0->K*mumu)',17.0,19.0),
        ('<P1>(B0->K*mumu)',1.0,2.0),('<P1>(B0->K*mumu)',2.0,4.3),('<P1>(B0->K*mumu)',4.3,6.0),
        ('<P1>(B0->K*mumu)',14.18,16.0),('<P1>(B0->K*mumu)',16.0,19.0),
        ('<P1>(B0->K*mumu)',0.04,2.0),('<P1>(B0->K*mumu)',2.0,4.0),('<P1>(B0->K*mumu)',4.0,6.0),
        ('<P1>(B0->K*mumu)',0.1,0.98),('<P1>(B0->K*mumu)',1.1,2.5),('<P1>(B0->K*mumu)',2.5,4.0),
        ('<P1>(B0->K*mumu)',4.0,6.0),('<P1>(B0->K*mumu)',15.0,17.0),('<P1>(B0->K*mumu)',17.0,19.0),
        ('<P2>(B0->K*mumu)',0.1,0.98),('<P2>(B0->K*mumu)',1.1,2.5),('<P2>(B0->K*mumu)',2.5,4.0),
        ('<P2>(B0->K*mumu)',4.0,6.0),('<P2>(B0->K*mumu)',15.0,17.0),('<P2>(B0->K*mumu)',17.0,19.0),
        ('<P3>(B0->K*mumu)',0.1,0.98),('<P3>(B0->K*mumu)',1.1,2.5),('<P3>(B0->K*mumu)',2.5,4.0),
        ('<P3>(B0->K*mumu)',4.0,6.0),('<P3>(B0->K*mumu)',15.0,17.0),('<P3>(B0->K*mumu)',17.0,19.0),
        ('<P4p>(B0->K*mumu)',0.04,2.0),('<P4p>(B0->K*mumu)',2.0,4.0),('<P4p>(B0->K*mumu)',4.0,6.0),
        ('<P4p>(B0->K*mumu)',0.1,0.98),('<P4p>(B0->K*mumu)',1.1,2.5),('<P4p>(B0->K*mumu)',2.5,4.0),
        ('<P4p>(B0->K*mumu)',4.0,6.0),('<P4p>(B0->K*mumu)',15.0,17.0),('<P4p>(B0->K*mumu)',17.0,19.0),
        ('<P5p>(B0->K*mumu)',1.0,2.0),('<P5p>(B0->K*mumu)',2.0,4.3),('<P5p>(B0->K*mumu)',4.3,6.0),
        ('<P5p>(B0->K*mumu)',14.18,16.0),('<P5p>(B0->K*mumu)',16.0,19.0),
        ('<P5p>(B0->K*mumu)',0.04,2.0),('<P5p>(B0->K*mumu)',2.0,4.0),('<P5p>(B0->K*mumu)',4.0,6.0),
        ('<P5p>(B0->K*mumu)',0.1,0.98),('<P5p>(B0->K*mumu)',1.1,2.5),('<P5p>(B0->K*mumu)',2.5,4.0),
        ('<P5p>(B0->K*mumu)',4.0,6.0),('<P5p>(B0->K*mumu)',15.0,17.0),('<P5p>(B0->K*mumu)',17.0,19.0),
        ('<P6p>(B0->K*mumu)',0.04,2.0),('<P6p>(B0->K*mumu)',2.0,4.0),('<P6p>(B0->K*mumu)',4.0,6.0),
        ('<P6p>(B0->K*mumu)',0.1,0.98),('<P6p>(B0->K*mumu)',1.1,2.5),('<P6p>(B0->K*mumu)',2.5,4.0),
        ('<P6p>(B0->K*mumu)',4.0,6.0),('<P6p>(B0->K*mumu)',15.0,17.0),('<P6p>(B0->K*mumu)',17.0,19.0),
        ('<P8p>(B0->K*mumu)',0.04,2.0),('<P8p>(B0->K*mumu)',2.0,4.0),('<P8p>(B0->K*mumu)',4.0,6.0),
        ('<P8p>(B0->K*mumu)',0.1,0.98),('<P8p>(B0->K*mumu)',1.1,2.5),('<P8p>(B0->K*mumu)',2.5,4.0),
        ('<P8p>(B0->K*mumu)',4.0,6.0),('<P8p>(B0->K*mumu)',15.0,17.0),('<P8p>(B0->K*mumu)',17.0,19.0),

        ('<dBR/dq2>(B0->Kmumu)',0.1,2.0),('<dBR/dq2>(B0->Kmumu)',2.0,4.0),('<dBR/dq2>(B0->Kmumu)',4.0,6.0),
        ('<dBR/dq2>(B0->Kmumu)',15.0,17.0),('<dBR/dq2>(B0->Kmumu)',17.0,19.0),
        #('<dBR/dq2>(B0->Kmumu)',1.0,6.0),('<dBR/dq2>(B0->Kmumu)',15.0,22.0),
        #('<dBR/dq2>(B0->Kmumu)',4.0,8.12),('<dBR/dq2>(B0->Kmumu)',0.1,4.0),
        #('<dBR/dq2>(B0->Kmumu)',0.1,2.0),('<dBR/dq2>(B0->Kmumu)',2.0,4.3),

        ('<dBR/dq2>(B+->Kmumu)',1.0,6.0),('<dBR/dq2>(B+->Kmumu)',21.0,22.0),
        ('<dBR/dq2>(B+->Kmumu)',0.1,4.0),('<dBR/dq2>(B+->Kmumu)',4.0,8.12),
        ('<dBR/dq2>(B+->Kmumu)',0.1,0.98),('<dBR/dq2>(B+->Kmumu)',1.1,2.0),('<dBR/dq2>(B+->Kmumu)',2.0,3.0),
        ('<dBR/dq2>(B+->Kmumu)',3.0,4.0),('<dBR/dq2>(B+->Kmumu)',4.0,5.0),('<dBR/dq2>(B+->Kmumu)',5.0,6.0),
        ('<dBR/dq2>(B+->Kmumu)',15.0,16.0),('<dBR/dq2>(B+->Kmumu)',16.0,17.0),('<dBR/dq2>(B+->Kmumu)',17.0,18.0),
        ('<dBR/dq2>(B+->Kmumu)',18.0,19.0),('<dBR/dq2>(B+->Kmumu)',19.0,20.0),('<dBR/dq2>(B+->Kmumu)',20.0,21.0),
        ('<AFB>(B+->Kmumu)',1.0,6.0),('<AFB>(B+->Kmumu)',16.0,18.0),('<AFB>(B+->Kmumu)',18.0,22.0),
        ('<FH>(B+->Kmumu)',1.0,6.0),('<FH>(B+->Kmumu)',16.0,18.0),('<FH>(B+->Kmumu)',18.0,22.0),
#        ('<dBR/dq2>(B+->Kmumu)',15.0,22.0),('<dBR/dq2>(B+->Kmumu)',0.0,2.0),('<dBR/dq2>(B+->Kmumu)',2.0,4.3),

        ('<dBR/dq2>(B+->K*mumu)',1.1,6.0),('<dBR/dq2>(B+->K*mumu)',15.0,19.0),
        ('<dBR/dq2>(B+->K*mumu)',0.1,2.0),('<dBR/dq2>(B+->K*mumu)',2.0,4.0),('<dBR/dq2>(B+->K*mumu)',4.0,6.0),
        ('<dBR/dq2>(B+->K*mumu)',15.0,19.0),('<dBR/dq2>(B+->K*mumu)',15.0,17.0),('<dBR/dq2>(B+->K*mumu)',17.0,19.0),
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
        ('<P8p>(B+->K*mumu)',4.0,6.0),('<P8p>(B+->K*mumu)',15.0,17.0),('<P8p>(B+->K*mumu)',17.0,19.0),
        ]

obs8 = [
        ('<dBR/dq2>(Bs->phimumu)',0.1,2.0),('<dBR/dq2>(Bs->phimumu)',2.0,5.0),
        ('<dBR/dq2>(Bs->phimumu)',15.0,17.0),('<dBR/dq2>(Bs->phimumu)',17.0,19.0),
        ('<FL>(Bs->phimumu)',0.1,2.0),('<FL>(Bs->phimumu)',2.0,5.0),('<FL>(Bs->phimumu)',15.0,17.0),('<FL>(Bs->phimumu)',17.0,19.0),
        ('<S3>(Bs->phimumu)',0.1,2.0),('<S3>(Bs->phimumu)',2.0,5.0),('<S3>(Bs->phimumu)',15.0,17.0),('<S3>(Bs->phimumu)',17.0,19.0),
        ('<S4>(Bs->phimumu)',0.1,2.0),('<S4>(Bs->phimumu)',2.0,5.0),('<S4>(Bs->phimumu)',15.0,17.0),('<S4>(Bs->phimumu)',17.0,19.0),
        ('<S7>(Bs->phimumu)',0.1,2.0),('<S7>(Bs->phimumu)',2.0,5.0),('<S7>(Bs->phimumu)',15.0,17.0),('<S7>(Bs->phimumu)',17.0,19.0),
        ('<dBR/dq2>(Lambdab->Lambdamumu)',1.1,6.0),('<dBR/dq2>(Lambdab->Lambdamumu)',15.0,16.0),
        ('<dBR/dq2>(Lambdab->Lambdamumu)',16.0,18.0),('<dBR/dq2>(Lambdab->Lambdamumu)',18.0,20.0),
        ('<AFBh>(Lambdab->Lambdamumu)',15.0,20.0),('<AFBl>(Lambdab->Lambdamumu)',15.0,20.0),
        ('<AFBlh>(Lambdab->Lambdamumu)',15.0,20.0)
        ]


angle_list = obs7 + obs8 + obs2[:-1] + obs6
print(len(angle_list))
quit()

#Fleps = Likelihood(observables=my_obs[:9]+my_obs[14:]+obs2[:2],include_measurements=['Tree Level Leptonics','LFU D Ratios','Tree Level Semileptonics']) 
#Fleps = FastLikelihood(name="trees",observables=my_obs[14:16],include_measurements=['LFU D Ratios',]) 
#Fleps = FastLikelihood(name="trees",observables=[my_obs[14]],include_measurements=['LFU D Ratios',]) 
#Fleps = FastLikelihood(name="trees",observables=[my_obs[15]],include_measurements=['LFU D Ratios',]) 
#------------------------------
#Fmix = Likelihood(observables=[my_obs[10]],include_measurements=['B Mixing',]) 
#Fmix = Likelihood(observables=[my_obs[11]],include_measurements=['B Mixing',]) 
Fmix = FastLikelihood(name='mix',observables=my_obs[10:12],include_measurements=['B Mixing',]) 
#Frad = FastLikelihood(name="rad",observables=[my_obs[9]],include_measurements=['Radiative Decays'])
#------------------------------
#Fmu = FastLikelihood(name="mu",observables=obs2[2:5],include_measurements=['LFU K Ratios 1','LFU K Ratios 2']) 
#Fmu = FastLikelihood(name="mu",observables=my_obs[12:14],include_measurements=['FCNC Leptonic Decays',]) 
#Fmu = FastLikelihood(name="mu",observables=obs3)
#------------------------------
#obs_list = my_obs+obs2[:2]+angle_list
#print(len(angle_list))
#print(len(obs_list))
#FL2 = FastLikelihood(name="glob",observables=obs_list,include_measurements=['Tree Level Leptonics','Radiative Decays','FCNC Leptonic Decays','B Mixing','LFU D Ratios','Tree Level Semileptonics','LFU K Ratios 1']+ims)
#------------------------------
#Fmuon = FastLikelihood(name="muons",observables=['a_mu'],include_measurements=['Anomalous Magnetic Moments'])
#Fmuon.make_measurement(N=500,threads=4)

#Fleps.make_measurement(N=500,threads=4)
Fmix.make_measurement(N=500,threads=4)
#Frad.make_measurement(N=500,threads=4)
#Fmu.make_measurement(N=500,threads=4)

#------------------------------
#   Leptonic and Semileptonic Tree Levels
#------------------------------

def leps(wcs):
    tanb, mH = wcs # state what the two parameters are going to be on the plot

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

    wc = flavio.WilsonCoefficients()
    wc.set_initial({ # tell flavio what WCs you're referring to with your variables
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

    C9_se, C9p_se, C10_se, C10p_se, CS_se, CSp_se, CP_se, CPp_se = bsll(par,ckm_els,['m_s','m_d',1],['m_e','m_mu',1],10**mH0,10**tanb,10**mH,align)
    C9_s, C9p_s, C10_s, C10p_s, CS_s, CSp_s, CP_s, CPp_s = bsll(par,ckm_els,['m_s','m_d',1],['m_mu','m_e',1],10**mH0,10**tanb,10**mH,align)
#    C9_d, C9p_d, C10_d, C10p_d, CS_d, CSp_d, CP_d, CPp_d = bsll(par,ckm_els,['m_d','m_s',0],['m_mu','m_e',1],10**mH0,10**tanb,10**mH,align)
    C7, C7p, C8, C8p = bsgamma2(par,ckm_els,flavio.config['renormalization scale']['bxgamma'],10**tanb,10**mH)

    wc = flavio.WilsonCoefficients()
    wc.set_initial({ # tell flavio what WCs you're referring to with your variables
           'C7_bs': C7,'C7p_bs': C7p, 
           'C8_bs': C8,'C8p_bs': C8p, 
           'C9_bsee': C9_se,'C9p_bsee': C9p_se,
           'C9_bsmumu': C9_s,'C9p_bsmumu': C9p_s,
           'C10_bsee': C10_se,'C10p_bsee': C10p_se,'CS_bsee': CS_se,'CSp_bsee': CSp_se,'CP_bsee': CP_se,'CPp_bsee': CPp_se, 
           'C10_bsmumu': C10_s,'C10p_bsmumu': C10p_s,'CS_bsmumu': CS_s,'CSp_bsmumu': CSp_s,'CP_bsmumu': CP_s,'CPp_bsmumu': CPp_s, # Bs->mumu
#           'C10_bdmumu': C10_d,'C10p_bdmumu': C10p_d,'CS_bdmumu': CS_d,'CSp_bdmumu': CSp_d,'CP_bdmumu': CP_d,'CPp_bdmumu': CPp_d, # B0->mumu
        }, scale=4.2, eft='WET', basis='flavio')
    return Fmu.log_likelihood(par,wc)

#mu_apx_plu = partial(mu,(0,1))
#mu_fix_plu = partial(mu,(1,1))
#mu_apx_min = partial(mu,(0,2))
#mu_fix_min = partial(mu,(1,2))

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

#def func(app,wcs):
def func(like,wcs):
    tanb, mH = wcs # state what the two parameters are going to be on the plot

#    if app == 0:
    mH0,mA0 = mH, mH
#    elif app == 2:
#        mH0,mA0 = mH, np.log10(1500)
#    elif app == 1:
#        mH0,mA0 = np.log10(1500), mH

    par = flavio.default_parameters.get_central_all()
    ckm_els = flavio.physics.ckm.get_ckm(par) # get out all the CKM elements

#    csev = a_mu2(par,'m_mu',10**tanb,10**mH0,10**mA0,10**mH)
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
    C9_se, C9p_se, C10_se, C10p_se, CS_se, CSp_se, CP_se, CPp_se = bsll(par,ckm_els,['m_s','m_d',1],['m_e','m_mu',1],10**mH0,10**tanb,10**mH,0)
    C9_s, C9p_s, C10_s, C10p_s, CS_s, CSp_s, CP_s, CPp_s = bsll(par,ckm_els,['m_s','m_d',1],['m_mu','m_e',1],10**mH0,10**tanb,10**mH,0)
    C9_d, C9p_d, C10_d, C10p_d, CS_d, CSp_d, CP_d, CPp_d = bsll(par,ckm_els,['m_d','m_s',0],['m_mu','m_e',1],10**mH0,10**tanb,10**mH,0)
    CVLL_bs, CVRR_bs, CSLL_bs, CSRR_bs, CSLR_bs, CVLR_bs = mixing(par,ckm_els,['m_s',1,'m_d'],10**tanb,10**mH)
    CVLL_bd, CVRR_bd, CSLL_bd, CSRR_bd, CSLR_bd, CVLR_bd = mixing(par,ckm_els,['m_d',0,'m_s'],10**tanb,10**mH)

    wc = flavio.WilsonCoefficients()
    wc.set_initial({ # tell flavio what WCs you're referring to with your variables
#            'C7_mumu': csev,
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
            'C10_bsee': C10_se,'C10p_bsee': C10p_se,'CS_bsee': CS_se,'CSp_bsee': CSp_se,'CP_bsee': CP_se,'CPp_bsee': CPp_se, 
            'C10_bsmumu': C10_s,'C10p_bsmumu': C10p_s,'CS_bsmumu': CS_s,'CSp_bsmumu': CSp_s,'CP_bsmumu': CP_s,'CPp_bsmumu': CPp_s, # Bs->mumu
            'C10_bdmumu': C10_d,'C10p_bdmumu': C10p_d,'CS_bdmumu': CS_d,'CSp_bdmumu': CSp_d,'CP_bdmumu': CP_d,'CPp_bdmumu': CPp_d, # B0->mumu
            'CVLL_bsbs': CVLL_bs,'CVRR_bsbs': CVRR_bs,'CSLL_bsbs': CSLL_bs,'CSRR_bsbs': CSRR_bs,'CSLR_bsbs': CSLR_bs,'CVLR_bsbs': CVLR_bs, # DeltaM_s
            'CVLL_bdbd': CVLL_bd,'CVRR_bdbd': CVRR_bd,'CSLL_bdbd': CSLL_bd,'CSRR_bdbd': CSRR_bd,'CSLR_bdbd': CSLR_bd,'CVLR_bdbd': CVLR_bd, # DeltaM_d
        }, scale=4.2, eft='WET', basis='flavio')
    return like.log_likelihood(par,wc)

def c2_test(app,wcs):
    tanb, mH = wcs
    ali,obs,m,u,l = app
    if ali == 0:
        mH0,mA0 = mH, mH
    elif ali == 1:
        mH0,mA0 = np.log10(1500), mH
    elif ali == 2:
        mH0,mA0 = mH, np.log10(1500)

    c2 = chi2_func(10**tanb,10**mH,10**mH0,10**mA0,obs,m,u,l)
    return c2

def test_func(ali,obs,m,u,l,tmin=-1.0,tmax=2.0,hmin=1.5,hmax=4.0,steps=100,n_sigma=1):
    tanb, mH = np.linspace(tmin,tmax,steps),np.linspace(hmin,hmax,steps)
    t,h = np.meshgrid(tanb,mH)
    th = np.array([t,h]).reshape(2,steps**2).T

    c2t = partial(c2_test,(ali,obs,m,u,l))

    pool = Pool()
    pred = np.array(pool.map(c2t,th)).reshape((steps,steps))
    pool.close()
    pool.join()

    if isinstance(n_sigma, Number):
        levels = [delta_chi2(n_sigma, dof=2)]
    else:
        levels = [delta_chi2(n, dof=2) for n in n_sigma]
    return {'x':t,'y':h,'z':pred,'levels':levels}

def sm_pull(obs):
    for i in obs:
        FL2 = FastLikelihood(name="glob",observables=[i],include_measurements=['Tree Level Leptonics','Radiative Decays','FCNC Leptonic Decays','B Mixing','LFU D Ratios','Tree Level Semileptonics','LFU K Ratios 1','LFU K Ratios 2']+ims)
        FL2.make_measurement(N=500,threads=4)
        glob0 = partial(func,FL2)
        cdat = fpl.likelihood_contour_data(glob0,-8,8,-8,8, n_sigma=(1,2), threads=4, steps=300) 

        chi_min = np.min(cdat['z'])

        wc = flavio.WilsonCoefficients()
        wc.set_initial({ # tell flavio what WCs you're referring to with your variables
                'CSR_bctaunutau': 0, 'CSL_bctaunutau': 0,
                'CSR_bcmunumu': 0, 'CSL_bcmunumu': 0,
                'CSR_bcenue': 0, 'CSL_bcenue': 0, 
                'CSR_butaunutau': 0, 'CSL_butaunutau': 0,
                'CSR_bumunumu': 0, 'CSL_bumunumu': 0,
                'CSR_buenue': 0, 'CSL_buenue': 0, 
                'CSR_dctaunutau': 0, 'CSL_dctaunutau': 0,
                'CSR_dcmunumu': 0, 'CSL_dcmunumu': 0,
                'CSR_dcenue': 0, 'CSL_dcenue': 0, 
                'CSR_sctaunutau': 0, 'CSL_sctaunutau': 0,
                'CSR_scmunumu': 0, 'CSL_scmunumu': 0,
                'CSR_scenue': 0, 'CSL_scenue': 0, 
                'CSR_sutaunutau': 0, 'CSL_sutaunutau': 0, 
                'CSR_sumunumu': 0, 'CSL_sumunumu': 0, 
                'CSR_suenue': 0, 'CSL_suenue': 0, 
                'CSR_dutaunutau': 0, 'CSL_dutaunutau': 0, 
                'CSR_dumunumu': 0, 'CSL_dumunumu': 0, 
                'C7_bs': 0,'C7p_bs': 0, 
                'C8_bs': 0,'C8p_bs': 0, 
                'C9_bsee': 0,'C9p_bsee': 0,
                'C9_bsmumu': 0,'C9p_bsmumu': 0,
                'C10_bsee': 0,'C10p_bsee': 0,'CS_bsee': 0,'CSp_bsee': 0,'CP_bsee': 0,'CPp_bsee': 0, 
                'C10_bsmumu': 0,'C10p_bsmumu': 0,'CS_bsmumu': 0,'CSp_bsmumu': 0,'CP_bsmumu': 0,'CPp_bsmumu': 0, 
                'C10_bdmumu': 0,'C10p_bdmumu': 0,'CS_bdmumu': 0,'CSp_bdmumu': 0,'CP_bdmumu': 0,'CPp_bdmumu': 0, 
                'CVLL_bsbs': 0,'CVRR_bsbs': 0,'CSLL_bsbs': 0,'CSRR_bsbs': 0,'CSLR_bsbs': 0,'CVLR_bsbs': 0, 
                'CVLL_bdbd': 0,'CVRR_bdbd': 0,'CSLL_bdbd': 0,'CSRR_bdbd': 0,'CSLR_bdbd': 0,'CVLR_bdbd': 0, 
            }, scale=4.2, eft='WET', basis='flavio')

        par = flavio.default_parameters.get_central_all()
        chi_sm = -2*FL2.log_likelihood(par,wc)

        print()
        if chi_sm > chi_min:
            print(i,' , ',pull(chi_sm-chi_min,2))
        elif chi_min > chi_sm:
            print(i,' , ',pull(chi_min-chi_sm,2))
    #    print(chi_sm-chi_min)
    return None

#sm_pull(angle_list)
#quit()

#glob0 = partial(func,0)
#glob1 = partial(func,1)
#glob2 = partial(func,2)

#------------------------------
#   Get Contour Data
#------------------------------

sigmas = (1,2)
#sigmas = (1,2,3,4,5,6)

#cmuon = fpl.likelihood_contour_data(muon,-1,2.5,-2,4, n_sigma=sigmas, threads=4, steps=60) 
#cmuon = fpl.likelihood_contour_data(muon,0,4,-1,4, n_sigma=sigmas, threads=4, steps=60) 
#cleps = fpl.likelihood_contour_data(leps,-1,2,1.5,4, n_sigma=sigmas, threads=4, steps=60) 
cmix = fpl.likelihood_contour_data(mix,-3,4,-2,6, n_sigma=sigmas, threads=4, steps=150) 
#crad = fpl.likelihood_contour_data(rad,-1,2,1.5,4, n_sigma=sigmas, threads=4, steps=80) 
#cmu = fpl.likelihood_contour_data(mu,2.5,4,-2,1, n_sigma=sigmas, threads=4, steps=60) 
#cmu = fpl.likelihood_contour_data(mu,-1,2,0,4, n_sigma=sigmas, threads=4, steps=60) 
#cmu0 = fpl.likelihood_contour_data(mu0,-1,2,1.5,4, n_sigma=sigmas, threads=4, steps=60) 
#cmu1 = fpl.likelihood_contour_data(mu1,-1,2,1.5,4, n_sigma=sigmas, threads=4, steps=60) 
#cdat = fpl.likelihood_contour_data(func,-1,2,1.5,4, n_sigma=sigmas, threads=4, steps=60) 
#cdat = fpl.likelihood_contour_data(func,-1,2,2.9,3.2, n_sigma=sigmas, threads=4, steps=400) 

#------------------------------
#   Print Out Values
#------------------------------

minz_bsgam = -19.066889111832868
minz_allsim = -759.1565659885084
minz_Hsim = -757.8282739220956
minz_Asim = -756.8886543429611

#app = 0#,1,2
#pval_func(cdat,app,obs_list)
#quit()

#------------------------------
#   Plotting
#------------------------------

#plt.figure(figsize=(6,5))
#fpl.contour(**cmuon,col=6) 
#plt.title(r'$m_{A^0}\sim m_{H^+}$ and $m_{H^0} = 1500\,$GeV',fontsize=18)
#plt.title(r'$m_{H^0}\sim m_{H^+}$ and $m_{A^0} = 1500\,$GeV',fontsize=18)
#plt.title(r'$m_{H^0},m_{A^0}\sim m_{H^+}$',fontsize=18)
#plt.axhline(y=np.log10(1220),color='black',linestyle='--') # Asim = 866, Hsim = 1220
#plt.axhline(y=np.log10(1660),color='black',linestyle='--') # Asim = 1660, Hsim = 1660
#plt.xlabel(r'$\log_{10}[\tan\beta]$') 
#plt.ylabel(r'$\log_{10}[m_{H^+}/\text{GeV}]$') 
#plt.savefig('muon_Asim.png') # 1,2 sig
#plt.savefig('muon_Hsim.png') # 3.3 sig
#plt.savefig('muon_allsim.png') # 3.3 sig
#plt.show()
#quit()

#plt.figure(figsize=(6,5))
#fpl.contour(**cleps,col=2)#,interpolation_factor=1.09,interpolation_order=1) 
###plt.title('Tree-Level Leptonic and Semileptonic Meson Decays and Hadronic Tau Decays')
###plt.title('Tree-Level Semileptonic Meson Decays')
###plt.title(r'$\mathcal{R}(D)$ and $\mathcal{R}(D^*)$')
###plt.title(strings)
#plt.xlabel(r'$\log_{10}[\tan\beta]$') 
#plt.ylabel(r'$\log_{10}[m_{H^+}/\text{GeV}]$') 
##plt.savefig(strings+'.png')
#plt.savefig('test_plot.pdf')
#quit()

plt.figure(figsize=(6,5))
fpl.contour(**cmix,col=0)#,interpolation_factor=1.015,interpolation_order=1) 
###plt.title(r'$\Delta M_{d,s}$')
plt.xlabel(r'$\log_{10}[\tan\beta]$') 
plt.ylabel(r'$\log_{10}[m_{H^+}/\text{GeV}]$') 
plt.savefig('mix_plot.pdf')
#plt.show()

#plt.figure(figsize=(6,5))
#fpl.contour(**crad,col=3)#,z_min=minz_bsgam)
##plt.title(r'$\bar{B}\to X_s\gamma$ Radiative Decay')
#plt.xlabel(r'$\log_{10}[\tan\beta]$') 
#plt.ylabel(r'$\log_{10}[m_{H^+}/\text{GeV}]$') 
#plt.savefig('bsgamma_plot2.pdf')
quit()

# (2.4,4.2,-2,1)
z_min1 = -5.182818890948422

#mt = [r'$m_{H^0}\sim m_{H^+},\,$',r'$m_{H^0}=1500\,$GeV,$\,$']
#at = [r'$\cos(\beta-\alpha)=0$',r'$\cos(\beta-\alpha)=0.05$',r'$\cos(\beta-\alpha)=-0.05$',r'$\cos(\beta-\alpha)=\sin2\beta$']
#bmu_mass = ['apx_','fix_']
#bmu_ali = ['cba02','cbap2','cbam2']
#for i in range(2):
#    for j in range(3):
#Fmu = FastLikelihood(name="mu",observables=my_obs[12:14],include_measurements=['FCNC Leptonic Decays',]) 
obs2_title = [r'$R(B^+\to K^+\ell\ell)$',r'$R(B^+\to K^+\ell\ell)$',r'$R(B^0\to K^{*0}\ell\ell)$',
              r'$R(B^0\to K^{*0}\ell\ell)$',r'$R(B^0\to K^{*0}\ell\ell)$',r'$R(B^+\to K^{*+}\ell\ell)$',
              r'$R(B^+\to K^{*+}\ell\ell)$',r'$R(B^+\to K^{*+}\ell\ell)$']
    
for i in range(8):
    ind = i + 2
    Fmu = FastLikelihood(name="mu",observables=[obs2[ind]],include_measurements=['LFU K Ratios 1','LFU K Ratios 2','Belle-1908.01848 RKs','Belle-1904.02440 RKs 1','Belle-1904.02440 RKs 2'])
    Fmu.make_measurement(N=500,threads=4)

    mu0 = partial(mu,(0,0))
    ##cmu = fpl.likelihood_contour_data(mu0,-1,2,0,4, n_sigma=(3,4), threads=4, steps=60) 
    cmu = fpl.likelihood_contour_data(mu0,-8,8,-8,8, n_sigma=sigmas, threads=4, steps=300)
    plt.figure(figsize=(6,5))
    fpl.contour(**cmu,col=9)#,z_min=z_min1)
    n,mini,maxi = obs2[ind]
    plt.title(obs2_title[i]+' '+str(mini)+'-'+str(maxi),fontsize=18)
    #plt.title(r'$1,2\sigma\,$ Contours for Original Three',fontsize=18)
    #plt.title(mt[i]+at[j],fontsize=18)
    #if i == 1:
    #    plt.axhline(y=np.log10(866),color='black',linestyle='--')
    #    plt.axhline(y=np.log10(1658),color='black',linestyle='--')
    plt.xlabel(r'$\log_{10}[\tan\beta]$') 
    plt.ylabel(r'$\log_{10}[m_{H^+}/\text{GeV}]$') 
    plt.savefig('rks_plot_big'+str(i+1)+'.pdf')
#    plt.savefig('rks_plot_og.pdf')

Fmu = FastLikelihood(name="mu",observables=obs2[2:10],include_measurements=['LFU K Ratios 1','LFU K Ratios 2','Belle-1908.01848 RKs','Belle-1904.02440 RKs 1','Belle-1904.02440 RKs 2'])
Fmu.make_measurement(N=500,threads=4)

mu0 = partial(mu,(0,0))
cmu = fpl.likelihood_contour_data(mu0,-8,8,-8,8, n_sigma=sigmas, threads=4, steps=300)
plt.figure(figsize=(6,5))
fpl.contour(**cmu,col=9)
plt.title(r'$1,2\sigma$',fontsize=18)
plt.xlabel(r'$\log_{10}[\tan\beta]$') 
plt.ylabel(r'$\log_{10}[m_{H^+}/\text{GeV}]$') 
plt.savefig('rks_plot_all_big.pdf')

quit()

#cmu = fpl.likelihood_contour_data(mu0,2.5,4,-2,1, n_sigma=(1,2), threads=4, steps=60) 
#plt.figure(figsize=(6,5))
#fpl.contour(**cmu,col=9,z_min=z_min1)
#plt.title(r'$1,2\sigma\,$ Contours',fontsize=18)
#plt.xlabel(r'$\log_{10}[\tan\beta]$') 
#plt.ylabel(r'$\log_{10}[m_{H^+}/\text{GeV}]$') 
#plt.savefig('rks_plot.png')
#
#quit()

#for i in range(len(obs3)):
#    Fmu = FastLikelihood(name="mu",observables=[obs3[i]],include_measurements=ims)
#    Fmu.make_measurement(N=500,threads=4)
#
#    cmu = fpl.likelihood_contour_data(mu0,-1,3,1.5,6, n_sigma=sigmas, threads=4, steps=80) 
#    plt.figure(figsize=(6,5))
#    fpl.contour(**cmu,col=9)
#    plt.title(r'$m_{H^0}\sim m_{H^+},$'+'\n'+obs4[i],fontsize=18)
#    plt.xlabel(r'$\log_{10}[\tan\beta]$') 
#    plt.ylabel(r'$\log_{10}[m_{H^+}/\text{GeV}]$') 
#    plt.savefig(obs3_png[i]+'_apx.png')
#
#    cmu = fpl.likelihood_contour_data(mu1,-1,3,1.5,6, n_sigma=sigmas, threads=4, steps=80) 
#    plt.figure(figsize=(6,5))
#    fpl.contour(**cmu,col=9)
#    plt.axhline(y=np.log10(866),color='black',linestyle='--')
#    plt.axhline(y=np.log10(1658),color='black',linestyle='--')
#    plt.title(r'$m_{H^0}=1500\,$GeV,'+'\n'+obs4[i],fontsize=18)
#    plt.xlabel(r'$\log_{10}[\tan\beta]$') 
#    plt.ylabel(r'$\log_{10}[m_{H^+}/\text{GeV}]$') 
#    plt.savefig(obs3_png[i]+'_fix.png')

#Fmu = FastLikelihood(name="mu",observables=angle_list,include_measurements=ims+['LFU K Ratios 1'])
#Fmu.make_measurement(N=500,threads=4)
#
#cmu = fpl.likelihood_contour_data(mu0,-1,2,1.5,20, n_sigma=sigmas, threads=4, steps=60) 
#plt.figure(figsize=(6,5))
#fpl.contour(**cmu,col=9)#,z_min=z_min1) 
#plt.title(r'$m_{H^0}\sim m_{H^+}$',fontsize=18)
#plt.xlabel(r'$\log_{10}[\tan\beta]$') 
#plt.ylabel(r'$\log_{10}[m_{H^+}/\text{GeV}]$') 
#plt.savefig('bkell_apx4.pdf')
#
#cmu = fpl.likelihood_contour_data(mu1,-1,2,1.5,20, n_sigma=sigmas, threads=4, steps=60) 
#plt.figure(figsize=(6,5))
#fpl.contour(**cmu,col=9)#,z_min=z_min1) 
#plt.axhline(y=np.log10(866),color='black',linestyle='--')
#plt.axhline(y=np.log10(1658),color='black',linestyle='--')
#plt.title(r'$m_{H^0}=1500\,$GeV',fontsize=18)
#plt.xlabel(r'$\log_{10}[\tan\beta]$') 
#plt.ylabel(r'$\log_{10}[m_{H^+}/\text{GeV}]$') 
#plt.savefig('bkell_fix4.pdf')

quit()
#print("bklls done")

for op in range(2):
    if op == 0:
        obs_list = my_obs+obs2[:2]+angle_list
    elif op == 1:
        obs_list = my_obs+obs2[:10]+angle_list
#    print(len(angle_list))
    print(len(obs_list))
    FL2 = FastLikelihood(name="glob",observables=obs_list,include_measurements=['Tree Level Leptonics','Radiative Decays','FCNC Leptonic Decays','B Mixing','LFU D Ratios','Tree Level Semileptonics','LFU K Ratios 1','LFU K Ratios 2']+ims)
    FL2.make_measurement(N=500,threads=4)
#    ms,us,ls = mk_measure(obs_list)

    for i in range(1):
        globo = partial(func,i) 
        cdat = fpl.likelihood_contour_data(globo,-1,2,1.5,20, n_sigma=sigmas, threads=4, steps=150) 
#        cdat = test_func(i,obs_list,ms,us,ls,n_sigma=sigmas,hmax=6.0,steps=50)
        pval_func(cdat,i,obs_list,sigmas)

        plt.figure(figsize=(6,5))
        fpl.contour(**cdat,col=4) 
        plt.xlabel(r'$\log_{10}[\tan\beta]$') 
        plt.ylabel(r'$\log_{10}[m_{H^+}/\text{GeV}]$')
        #plt.title(r'Combined Tree-Level Leptonics, $\Delta M_{d,s}$, $\bar{B}\to X_s\gamma$')
        if i == 0:
            plt.title(r'$m_{H^0}\sim m_{H^+}$',fontsize=18)
            plt.savefig('comb4_Hsim'+str(op)+'.pdf')
        elif i == 2:
            plt.title(r'$m_{H^0}\sim m_{H^+},\; m_{A^0}=1500\,$GeV',fontsize=18)
            plt.axhline(y=np.log10(1220),color='black',linestyle='--') # Asim = 866, Hsim = 1220
            plt.axhline(y=np.log10(1660),color='black',linestyle='--') # Asim = 1660, Hsim = 1660
            plt.savefig('comb4_Hsim'+str(op)+'.pdf')
        elif i == 1:
            plt.title(r'$m_{H^0}=1500\,$GeV',fontsize=18)
            plt.axhline(y=np.log10(866),color='black',linestyle='--') # Asim = 866, Hsim = 1220
            plt.axhline(y=np.log10(1660),color='black',linestyle='--') # Asim = 1660, Hsim = 1660
            plt.savefig('comb4_Hfix'+str(op)+'.pdf')
        #plt.savefig('comb1_plot.png')
        #plt.savefig('comb2_allsim.png')
        #plt.savefig('comb2_Hsim.png')
        #plt.savefig('comb2_Asim.png')

print("--- %s seconds ---" % (time.time() - start_time))
print(datetime.datetime.now())

#plt.show()
# colours : 0 - blue, 1 - orange, 2 - green, 3 - pink, 4 - purple, 5 - brown, 6 - bright pink, 7 - grey, 8 - yellow, 9 - cyan
