import time
start_time = time.time()
import datetime
print(datetime.datetime.now())
import os
import numpy as np
import flavio
import flavio.plots as fpl
import matplotlib.pyplot as plt
from extra import *
import multiprocessing as mp
from multiprocessing import Pool
import ctypes
from functools import partial

par = flavio.default_parameters.get_central_all()
err = flavio.default_parameters.get_1d_errors_random()
ckm_els, ckm_errs = ckm_err(get_ckm_alex,par,err)
#mH = 3.5
#tanb = 1
#fac = 0.89
#par['Vub'] = par['Vub']*fac#rh([par['m_u'],par['m_b'],par['m_tau']],[tanb,mH])
#par['Vus'] = par['Vus']*fac#*rh([par['m_u'],par['m_s'],par['m_mu']],[tanb,mH])
#par['Vcb'] = par['Vcb']*fac#*rh([par['m_c'],par['m_b'],par['m_tau']],[tanb,mH])
#print(par['Vub'],par['Vus'],par['Vcb'])
ckm_test = get_ckm(par)
unit = ckm_test*np.conj(ckm_test.T)
#print(unit)
#print(abs(unit[0,0] + unit[0,1] + unit[0,2]))
r1_lim = 1 - abs(unit[0,0] + unit[0,1] + unit[0,2])
#print(abs(unit[1,0] + unit[1,1] + unit[1,2]))
r2_lim = 1 - abs(unit[1,0] + unit[1,1] + unit[1,2])
#print(abs(unit[2,0] + unit[2,1] + unit[2,2]))
r3_lim = 1 - abs(unit[2,0] + unit[2,1] + unit[2,2])

#heatmap = {'Vub':np.array([[0.9,0.99],[0.9,0.99]]),'Vus':np.array([[0.9,0.99],[0.9,0.99]]),'Vcb':np.array([[0.9,0.99],[0.99,0.99]])}
#heatmap = {'Vub':np.array([[0.1,0.1],[0.1,0.1]]),'Vus':np.array([[0.1,0.1],[0.1,0.1]]),'Vcb':np.array([[0.5,0.5],[0.5,0.5]])}
#errmap = {'Vub':np.array([[0,0],[0,0]]),'Vus':np.array([[0,0],[0,0]]),'Vcb':np.array([[0,0],[0,0]])}
#
#argus = [ckm_els,ckm_errs,par,err,heatmap,errmap,r1_lim,r2_lim,r3_lim]
#answer = testing(argus,(0,0))
#print(answer)
#
#quit()

def shared_zeros(n1,n2):
    ''' create a 2D numpy array which can be then changed in different threads '''
    shared_array_base = mp.Array(ctypes.c_double, n1 * n2)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(n1, n2)
    return shared_array

def vij_mult(args,ths):
    par, err, my_obs, steps, d, u, arr = args
    tanb, mH = ths
#    tb, ah = np.linspace(-1,2,steps), np.linspace(1,3.5,steps)
#    i, j  = np.where(tb==tanb)[0][0], np.where(ah==mH)[0][0]
    smp, npp, mod = [],[],[]
    smpe, nppe, mode = [],[],[]
    Vij = abs(par['V'+u+d])**2
    CSR_ijt, CSL_ijt = vcb(par['m_'+u],par['m_'+d],par['m_tau'],10**tanb,10**mH)
    CSR_ijm, CSL_ijm = vcb(par['m_'+u],par['m_'+d],par['m_mu'],10**tanb,10**mH)
    CSR_ije, CSL_ije = vcb(par['m_'+u],par['m_'+d],par['m_e'],10**tanb,10**mH)
    wc_np = flavio.WilsonCoefficients()
    wc_np.set_initial({'CSR_'+d+u+'taunutau': CSR_ijt, 'CSL_'+d+u+'taunutau': CSL_ijt,'CSR_'+d+u+'munumu': CSR_ijm,'CSL_'+d+u+'munumu': CSL_ijm,'CSR_'+d+u+'enue': CSR_ije,'CSL_'+d+u+'enue': CSL_ije,},scale=4.2,eft='WET',basis='flavio')
    for k in range(len(my_obs)):
        sm_pred = flavio.sm_prediction(my_obs[k])
        np_pred = flavio.np_prediction(my_obs[k],wc_obj=wc_np)
        smp.append(sm_pred/Vij)
        npp.append(np_pred/Vij)
        mod.append(np.sqrt(smp[k]/npp[k]))
        if arr == 2:
            sm_err = flavio.sm_uncertainty(my_obs[k])
            np_err = flavio.np_uncertainty(my_obs[k],wc_obj=wc_np)
            smpe.append(np.sqrt((sm_err/sm_pred)**2 + (err['V'+u+d]/Vij)**2)*smp[k])
            nppe.append(np.sqrt((np_err/np_pred)**2 + (err['V'+u+d]/Vij)**2)*npp[k])
            mode.append(np.sqrt((smpe[k]/(2*smp[k]))**2 + (nppe[k]/(2*npp[k]))**2)*mod[k])
    if arr == 1:
        return np.average(mod)
    else:
        return np.average(mode)
    #array_vs[j,i] = np.average(mod)
    #array_es[j,i] = np.average(mode)

steps = 100

tanb,mH = np.linspace(-1,2,steps), np.linspace(1.5,4,steps)
t, h = np.meshgrid(tanb,mH)
th = np.array([t,h]).reshape(2,steps**2).T

eyes, jays = np.arange(0,steps,1),np.arange(0,steps,1)
es, js = np.meshgrid(eyes,jays)
ej = np.array([es,js]).reshape(2,steps**2).T

my_obs = [['BR(B+->taunu)','BR(B+->munu)','BR(B0->pilnu)','BR(B0->rholnu)'],['BR(K+->munu)','BR(KL->pienu)','BR(KL->pimunu)','BR(KS->pienu)','BR(KS->pimunu)'],['Rtaul(B->Dlnu)', 'Rtaul(B->D*lnu)', 'BR(B0->Dlnu)', 'BR(B+->Dlnu)', 'BR(B0->D*lnu)', 'BR(B+->D*lnu)']]

#par = flavio.default_parameters.get_central_all()
#ckm_els = flavio.physics.ckm.get_ckm(par)
us = ['u','u','c']
ds = ['b','s','b']
mms = ['m_B+','m_D+','m_Ds','m_K+','m_pi+']
lats = ['$m_{B^+}$','$m_{D^+}$','$m_{D_s}$','$m_{K^+}$',r'$m_{\pi^+}$']
pngs = ['mB','mD','mDs','mK','mpi']
heatmap = {}
errmap = {}

for i in range(len(us)):
    args = [par,err,my_obs[i],steps,ds[i],us[i],1]
    arge = [par,err,my_obs[i],steps,ds[i],us[i],2]
    multy_vs = partial(vij_mult,args)
    multy_es = partial(vij_mult,arge)

#    heatmap_v = shared_zeros(steps,steps)
#    heatmap_e = shared_zeros(steps,steps)

    pool2 = Pool(processes=4)
    heatmap_v = np.array(pool2.map(multy_vs,th)).reshape((steps,steps))
    pool2.close()
    pool2.join()

    pool3 = Pool(processes=4)
    heatmap_e = np.array(pool3.map(multy_es,th)).reshape((steps,steps))
    pool3.close()
    pool3.join()

    heatmap['V'+us[i]+ds[i]] = heatmap_v
    errmap['V'+us[i]+ds[i]] = heatmap_e
    
#    vm = 0
#    if np.min(np.log10(heatmap_v)) < 0:
#        vm = np.min(np.log10(heatmap_v))
#
#    fig = plt.figure()
#    s = fig.add_subplot(1,1,1,xlabel=r"$\log_{10}[\tan\beta]$",ylabel=r"$\log_{10}[m_{H^+}/\text{GeV}]$")
#    im = s.imshow(np.log10(heatmap_v),extent=(tanb[0],tanb[-1],mH[0],mH[-1]),origin='lower',vmin=vm)#,vmax=1)
#    fig.colorbar(im)
#    plt.title("Heatmap of Modification Factor for V"+us[i]+ds[i])
#    plt.savefig("v"+us[i]+ds[i]+"_heatmap4.pdf")

    print("V"+us[i]+ds[i]+" done")
    print(datetime.datetime.now())

argus = [ckm_els,ckm_errs,par,err,heatmap,errmap,r1_lim,r2_lim,r3_lim]
testy = partial(testing,argus)
pool4 = Pool()
units = np.array(pool4.map(testy,ej)).reshape((steps,steps))
pool4.close()
pool4.join()

levs = (0.99,)
dat = {'x': t, 'y': h, 'z': units, 'levels': levs}

fig = plt.figure(figsize=(6,5))
fpl.contour(**dat,col=2)
plt.title("Modification Regions Allowed By Unitarity \n of full CKM Matrix")
plt.xlabel(r'$\log_{10}[\tan\beta]$')
plt.ylabel(r'$\log_{10}[m_{H^+}/\text{GeV}]$')
plt.savefig("ckm_full_mat7.pdf")
#plt.show()

fig = plt.figure()
s = fig.add_subplot(1,1,1,xlabel=r"$\log_{10}[\tan\beta]$",ylabel=r"$\log_{10}[m_{H^+}/\text{GeV}]$")
im = s.imshow(-1*units,extent=(tanb[0],tanb[-1],mH[0],mH[-1]),origin='lower',cmap='gray')
plt.title("Modification Regions Allowed By Unitarity \n of full CKM Matrix")
plt.savefig("ckm_full_mat8.pdf")

print("--- %s seconds ---" % (time.time() - start_time))
print(datetime.datetime.now())
