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
from numbers import Number
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
       #'CDF 0.0-2.0','CDF 2.0-4.3',
       'BaBar-1312.5364 Xs','BaBar-1204.3933 RKs',
       'Belle-1908.01848','Belle-1908.01848 RKs','Belle-1904.02440','Belle-1904.02440 RKs 1','Belle-1904.02440 RKs 2']

obs6 = [#('<dBR/dq2>(B+->Kee)',1.1,6.0),('<dBR/dq2>(B+->Kee)',1.0,6.0),('<dBR/dq2>(B+->Kee)',0.1,4.0),
        #('<dBR/dq2>(B+->Kee)',4.0,8.12),
        #('<dBR/dq2>(B0->Kee)',1.0,6.0),('<dBR/dq2>(B0->Kee)',0.1,4.0),('<dBR/dq2>(B0->Kee)',4.0,8.12),
        ('<P4p>(B->K*mumu)',1.0,6.0),('<P4p>(B->K*mumu)',14.18,19.0),
        ('<P5p>(B->K*mumu)',1.0,6.0),('<P5p>(B->K*mumu)',14.18,19.0),
        ('<P4p>(B->K*ee)',1.0,6.0),('<P4p>(B->K*ee)',14.18,19.0),
        ('<P5p>(B->K*ee)',1.0,6.0),('<P5p>(B->K*ee)',14.18,19.0),
#        ('<Dmue_P4p>(B0->K*ll)',0.1,4.0),('<Dmue_P4p>(B0->K*ll)',4.0,8.0),
#        ('<Dmue_P4p>(B0->K*ll)',1.0,6.0),('<Dmue_P4p>(B0->K*ll)',14.18,19.0),
#        ('<Dmue_P5p>(B0->K*ll)',0.1,4.0),('<Dmue_P5p>(B0->K*ll)',4.0,8.0),
#        ('<Dmue_P5p>(B0->K*ll)',1.0,6.0),('<Dmue_P5p>(B0->K*ll)',14.18,19.0),
        ('<FL>(B0->K*ee)',0.002,1.12),#('<dBR/dq2>(B0->K*ee)',0.003,1.0),
        ('<BR>(B->Xsmumu)',14.2,25.0),('<BR>(B->Xsmumu)',1.0,6.0),
        #('<BR>(B->Xsmumu)',0.1,2.0),#('<BR>(B->Xsmumu)',2.0,4.3),('<BR>(B->Xsmumu)',4.3,6.8),
        ('<BR>(B->Xsee)',14.2,25.0),('<BR>(B->Xsee)',0.1,2.0),('<BR>(B->Xsee)',2.0,4.3),('<BR>(B->Xsee)',4.3,6.8)]

obs7 = [
        ('<dBR/dq2>(B0->K*mumu)',1.0,2.0),('<dBR/dq2>(B0->K*mumu)',2.0,4.3),('<dBR/dq2>(B0->K*mumu)',4.3,6.0),
        ('<dBR/dq2>(B0->K*mumu)',14.18,16.0),('<dBR/dq2>(B0->K*mumu)',16.0,19.0),
#        ('<dBR/dq2>(B0->K*mumu)',1.1,6.0),('<dBR/dq2>(B0->K*mumu)',15.0,19.0),
        ('<AFB>(B0->K*mumu)',1.0,2.0),('<AFB>(B0->K*mumu)',2.0,4.3),('<AFB>(B0->K*mumu)',4.3,6.0),
        ('<AFB>(B0->K*mumu)',14.18,16.0),('<AFB>(B0->K*mumu)',16.0,19.0),
        ('<FL>(B0->K*mumu)',1.0,2.0),#('<FL>(B0->K*mumu)',2.0,4.3),('<FL>(B0->K*mumu)',4.3,6.0),
        ('<FL>(B0->K*mumu)',14.18,16.0),#('<FL>(B0->K*mumu)',16.0,19.0),
        ('<FL>(B0->K*mumu)',0.04,2.0),
        ('<FL>(B0->K*mumu)',0.1,0.98),('<FL>(B0->K*mumu)',1.1,2.5),('<FL>(B0->K*mumu)',2.5,4.0),
        ('<FL>(B0->K*mumu)',4.0,6.0),('<FL>(B0->K*mumu)',15.0,17.0),('<FL>(B0->K*mumu)',17.0,19.0),
        ('<P1>(B0->K*mumu)',1.0,2.0),#('<P1>(B0->K*mumu)',2.0,4.3),('<P1>(B0->K*mumu)',4.3,6.0),
        ('<P1>(B0->K*mumu)',14.18,16.0),#('<P1>(B0->K*mumu)',16.0,19.0),
        ('<P1>(B0->K*mumu)',0.04,2.0),
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
        ('<P5p>(B0->K*mumu)',14.18,16.0),
        ('<P5p>(B0->K*mumu)',0.04,2.0),('<P5p>(B0->K*mumu)',2.0,4.0),
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
        #('<dBR/dq2>(B0->Kmumu)',1.0,6.0),('<dBR/dq2>(B0->Kmumu)',15.0,22.0),
        #('<dBR/dq2>(B0->Kmumu)',4.0,8.12),('<dBR/dq2>(B0->Kmumu)',0.1,4.0),
        #('<dBR/dq2>(B0->Kmumu)',0.1,2.0),('<dBR/dq2>(B0->Kmumu)',2.0,4.3),

        #('<dBR/dq2>(B+->Kmumu)',1.0,6.0),
        ('<dBR/dq2>(B+->Kmumu)',21.0,22.0),
        #('<dBR/dq2>(B+->Kmumu)',0.1,4.0),
        ('<dBR/dq2>(B+->Kmumu)',4.0,8.12),
        ('<dBR/dq2>(B+->Kmumu)',0.1,0.98),('<dBR/dq2>(B+->Kmumu)',1.1,2.0),('<dBR/dq2>(B+->Kmumu)',2.0,3.0),
        ('<dBR/dq2>(B+->Kmumu)',3.0,4.0),('<dBR/dq2>(B+->Kmumu)',4.0,5.0),('<dBR/dq2>(B+->Kmumu)',5.0,6.0),
        ('<dBR/dq2>(B+->Kmumu)',15.0,16.0),('<dBR/dq2>(B+->Kmumu)',16.0,17.0),('<dBR/dq2>(B+->Kmumu)',17.0,18.0),
        ('<dBR/dq2>(B+->Kmumu)',18.0,19.0),('<dBR/dq2>(B+->Kmumu)',19.0,20.0),('<dBR/dq2>(B+->Kmumu)',20.0,21.0),
        ('<AFB>(B+->Kmumu)',1.0,6.0),('<AFB>(B+->Kmumu)',16.0,18.0),('<AFB>(B+->Kmumu)',18.0,22.0),
        ('<FH>(B+->Kmumu)',1.0,6.0),('<FH>(B+->Kmumu)',16.0,18.0),('<FH>(B+->Kmumu)',18.0,22.0),
#        ('<dBR/dq2>(B+->Kmumu)',15.0,22.0),('<dBR/dq2>(B+->Kmumu)',0.0,2.0),('<dBR/dq2>(B+->Kmumu)',2.0,4.3),

        #('<dBR/dq2>(B+->K*mumu)',1.1,6.0),('<dBR/dq2>(B+->K*mumu)',15.0,19.0),
        ('<dBR/dq2>(B+->K*mumu)',0.1,2.0),('<dBR/dq2>(B+->K*mumu)',2.0,4.0),('<dBR/dq2>(B+->K*mumu)',4.0,6.0),
        #('<dBR/dq2>(B+->K*mumu)',15.0,19.0),
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
        ('<AFBlh>(Lambdab->Lambdamumu)',15.0,20.0)]

angle_list = obs7 + obs8 + obs6 #+ obs2[2:-1]
#obs_list = my_obs+obs2[:2]+angle_list

def func(wcs):
    tanb, mH, mH0, mA0, cba = wcs # state what the two parameters are going to be on the plot

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
    C9_se, C9p_se, C10_se, C10p_se, CS_se, CSp_se, CP_se, CPp_se = bsll(par,ckm_els,2,1,0,10**mH0,10**tanb,10**mH,cba)
    C9_s, C9p_s, C10_s, C10p_s, CS_s, CSp_s, CP_s, CPp_s = bsll(par,ckm_els,2,1,1,10**mH0,10**tanb,10**mH,cba)
    C9_d, C9p_d, C10_d, C10p_d, CS_d, CSp_d, CP_d, CPp_d = bsll(par,ckm_els,2,0,1,10**mH0,10**tanb,10**mH,cba)
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
            'C7_bs': C7,'C7p_bs': C7p, 'C8_bs': C8,'C8p_bs': C8p, 
            'C9_bsee': C9_se,'C9p_bsee': C9p_se,'C9_bsmumu': C9_s,'C9p_bsmumu': C9p_s,
            'C10_bsee': C10_se,'C10p_bsee': C10p_se,'CS_bsee': CS_se,'CSp_bsee': CSp_se,'CP_bsee': CP_se,'CPp_bsee': CPp_se, 
            'C10_bsmumu': C10_s,'C10p_bsmumu': C10p_s,'CS_bsmumu': CS_s,'CSp_bsmumu': CSp_s,'CP_bsmumu': CP_s,'CPp_bsmumu': CPp_s, # Bs->mumu
            'C10_bdmumu': C10_d,'C10p_bdmumu': C10p_d,'CS_bdmumu': CS_d,'CSp_bdmumu': CSp_d,'CP_bdmumu': CP_d,'CPp_bdmumu': CPp_d, # B0->mumu
            'CVLL_bsbs': CVLL_bs,'CVRR_bsbs': CVRR_bs,'CSLL_bsbs': CSLL_bs,'CSRR_bsbs': CSRR_bs,'CSLR_bsbs': CSLR_bs,'CVLR_bsbs': CVLR_bs, # DeltaM_s
            'CVLL_bdbd': CVLL_bd,'CVRR_bdbd': CVRR_bd,'CSLL_bdbd': CSLL_bd,'CSRR_bdbd': CSRR_bd,'CSLR_bdbd': CSLR_bd,'CVLR_bdbd': CVLR_bd, # DeltaM_d
        }, scale=4.2, eft='WET', basis='flavio')
    return wc

def func2(wcs):
    tanb, mH, mH0, mA0, cba = wcs # state what the two parameters are going to be on the plot

    par = flavio.default_parameters.get_central_all()
    u_22, u_33, d_22, d_33, e_22, e_33, C_W, C_B, C_WB = Two_HDM_WC(par, 10**tanb, cba)

    wc = flavio.WilsonCoefficients()
    wc.set_initial({
        "uphi_33": u_33, "uphi_22": u_22,"dphi_33": d_33, "dphi_22": d_22,
        "ephi_33": e_33, "ephi_22": e_22,"phiW": C_W, "phiWB": C_WB, "phiB": C_B,
        }, scale=125,eft='SMEFT',basis='Warsaw')
    return wc

def c2_test(app,wcs):
    obs,m,u,l = app
    wc1 = func(wcs)
    wc2 = func2(wcs)
    c2 = chi2_func(obs,m,u,l,wc1,wc2)
    return c2

def test_func(obs,m,u,l,
        tmin=-1.0,tmax=2.0,hmin=2.0,hmax=5.0,omin=2.0,omax=5.0,amin=2.0,amax=5.0,cmin=-0.2,cmax=0.2,
        steps=10,n_sigma=1,threads=4):
    tanb = np.linspace(tmin,tmax,steps)
    mH = np.linspace(hmin,hmax,steps)
    mH0 = np.linspace(omin,omax,steps)
    mA0 = np.linspace(amin,amax,steps)
    cba = np.linspace(cmin,cmax,steps)
    t,h,o,a,c = np.meshgrid(tanb,mH,mH0,mA0,cba)
    th = np.array([t,h,o,a,c]).reshape(5,steps**5).T

    c2t = partial(c2_test,(obs,m,u,l))

    pool = Pool(threads)
    pred = np.array(pool.map(c2t,th)).reshape((steps,steps,steps,steps,steps))
    pool.close()
    pool.join()

    if isinstance(n_sigma, Number):
        levels = [delta_chi2(n_sigma, dof=5)]
    else:
        levels = [delta_chi2(n, dof=5) for n in n_sigma]
    return {'x':t,'y':h,'o':o,'a':a,'c':c,'z':pred,'levels':levels}


#------------------------------
#   Get Contour Data
#------------------------------

sigmas = (1,2)
#sigmas = (1,2,3,4,5,6)

for op in range(1):
    if op == 0:
        obs_list = my_obs+obs2[:2]+angle_list
    elif op == 1:
        obs_list = my_obs+obs2[:12]+angle_list
    print(len(angle_list))
    print(len(obs_list))
    FL2 = FastLikelihood(name="glob",observables=obs_list,include_measurements=['Tree Level Leptonics','Radiative Decays','FCNC Leptonic Decays','B Mixing','LFU D Ratios','Tree Level Semileptonics','LFU K Ratios 1','LFU K Ratios 2']+ims)
    FL2.make_measurement(N=500,threads=4)
    ms,us,ls = mk_measure(obs_list)

    cdat = test_func(obs_list,ms,us,ls,n_sigma=(1,2))
    pval_func2(cdat,obs_list,sigmas)

    np.save("cdat_5D.npy",cdat)
#        plt.figure(figsize=(6,5))
#        fpl.contour(**cdat,col=4) 
#        plt.xlabel(r'$\log_{10}[\tan\beta]$') 
#        plt.ylabel(r'$\log_{10}[m_{H^+}/\text{GeV}]$')
#        #plt.title(r'Combined Tree-Level Leptonics, $\Delta M_{d,s}$, $\bar{B}\to X_s\gamma$')
#        if i == 0:
#            plt.title(r'$m_{H^0}\sim m_{H^+},\,$ excl. $R_K$s',fontsize=18)
#            plt.savefig('comb4_Hsim_test'+str(op)+'.pdf')
#        elif i == 2:
#            plt.title(r'$m_{H^0}\sim m_{H^+},\; m_{A^0}=1500\,$GeV',fontsize=18)
#            plt.axhline(y=np.log10(1220),color='black',linestyle='--') # Asim = 866, Hsim = 1220
#            plt.axhline(y=np.log10(1660),color='black',linestyle='--') # Asim = 1660, Hsim = 1660
#            plt.savefig('comb4_Hsim'+str(op)+'.pdf')
#        elif i == 1:
#            plt.title(r'$m_{H^0}=1500\,$GeV',fontsize=18)
#            plt.axhline(y=np.log10(866),color='black',linestyle='--') # Asim = 866, Hsim = 1220
#            plt.axhline(y=np.log10(1660),color='black',linestyle='--') # Asim = 1660, Hsim = 1660
#            plt.savefig('comb4_Hfix'+str(op)+'.pdf')

print("--- %s seconds ---" % (time.time() - start_time))
print(datetime.datetime.now())

#plt.show()
# colours : 0 - blue, 1 - orange, 2 - green, 3 - pink, 4 - purple, 5 - brown, 6 - bright pink, 7 - grey, 8 - yellow, 9 - cyan
