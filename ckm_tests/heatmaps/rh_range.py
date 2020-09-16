import time
start_time = time.time()
import os
import numpy as np
import flavio
import matplotlib.pyplot as plt
from extra import *
import multiprocessing as mp
from multiprocessing import Pool
import ctypes
from functools import partial

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
    CSL_ij, CSR_ij = vcb(par['m_'+u],par['m_'+d],10**tanb,10**mH)
    wc_np = flavio.WilsonCoefficients()
    wc_np.set_initial({'CSR_'+d+u+'taunutau': CSR_ij, 'CSL_'+d+u+'taunutau': CSL_ij,'CSR_'+d+u+'munumu': CSR_ij,'CSL_'+d+u+'munumu': CSL_ij,'CSR_'+d+u+'enue': CSR_ij,'CSL_'+d+u+'enue': CSL_ij,},scale=4.2,eft='WET',basis='flavio')
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

steps = 60

tanb,mH = np.linspace(-1,2,steps), np.linspace(1,3.5,steps)
t, h = np.meshgrid(tanb,mH)
th = np.array([t,h]).reshape(2,steps**2).T

eyes, jays = np.arange(0,steps,1),np.arange(0,steps,1)
es, js = np.meshgrid(eyes,jays)
ej = np.array([es,js]).reshape(2,steps**2).T

my_obs = [['BR(B+->taunu)','BR(B+->munu)','BR(B0->pilnu)','BR(B0->rholnu)'],['BR(K+->munu)','BR(KL->pienu)','BR(KL->pimunu)','BR(KS->pienu)','BR(KS->pimunu)'],['Rtaul(B->Dlnu)', 'Rtaul(B->D*lnu)', 'BR(B0->Dlnu)', 'BR(B+->Dlnu)', 'BR(B0->D*lnu)', 'BR(B+->D*lnu)']]

par = flavio.default_parameters.get_central_all()
err = flavio.default_parameters.get_1d_errors_random()
#ckm_els = flavio.physics.ckm.get_ckm(par)
ckm_els, ckm_errs = ckm_err(par,err)
us = ['u','u','c']
ds = ['b','s','b']
mms = ['m_B+','m_D+','m_Ds','m_K+','m_pi+']
lats = ['$m_{B^+}$','$m_{D^+}$','$m_{D_s}$','$m_{K^+}$',r'$m_{\pi^+}$']
pngs = ['mB','mD','mDs','mK','mpi']
heatmap = {}
errmap = {}

#for m in range(len(mus)):
#    pool1 = Pool()
#    data_list = [par[mus[m]],par[mds[m]],par[mms[m]]]#,10**tanb,10**mH]
#    errs_list = [par,err,mus,mds,mms,m]#tanb,mH,m]
#    rhth = partial(rh,data_list)
#    rh_err = partial(errors1,errs_list)
#    heatmap_i = np.array(pool1.map(rhth,th)).reshape((steps,steps))
#    heatmap_e = np.array(pool1.map(rh_err,th)).reshape((steps,steps))
#    pool1.close()
#    pool1.join()
#        #        fig_title = r"$\frac{|\tilde{V}_{ij}|}{|V_{ij}|}$ for " + lats[m]
#        #        fig_name = pngs[m] + "-" + "rH" + str(x) + ".png"
#        #        print("Range for "+fig_title+" is "+str(np.min(heatmap_i))+" to " +str(np.max(heatmap_i)))
#    heatmap[pngs[m]] = heatmap_i
#    errmap[pngs[m]] = heatmap_e
#
#        #        fig = plt.figure()
#        #        s = fig.add_subplot(1,1,1,xlabel=r"$\log_{10}[\tan\beta]$",ylabel=r"$\log_{10}[m_{H^+}]$")
#        #        im = s.imshow(heatmap_i,extent=(tanb[0],tanb[-1],mH[0],mH[-1]),origin='lower')
#        #        fig.colorbar(im)
#        #        plt.title(fig_title)
#        #        plt.savefig(fig_name)
#        #        plt.show()

for i in range(len(us)):
    args = [par,err,my_obs[i],steps,ds[i],us[i],1]
    arge = [par,err,my_obs[i],steps,ds[i],us[i],2]
    multy_vs = partial(vij_mult,args)
    multy_es = partial(vij_mult,arge)

#    heatmap_v = shared_zeros(steps,steps)
#    heatmap_e = shared_zeros(steps,steps)

    pool2 = Pool(processes=4)
    heatmap_v = np.array(pool2.map(multy_vs,th)).reshape((steps,steps))
    heatmap_e = np.array(pool2.map(multy_es,th)).reshape((steps,steps))
    pool2.close()
    pool2.join()

    heatmap['V'+us[i]+ds[i]] = heatmap_v
    errmap['V'+us[i]+ds[i]] = heatmap_e
    
    fig = plt.figure()
    s = fig.add_subplot(1,1,1,xlabel=r"$\log_{10}[\tan\beta]$",ylabel=r"$\log_{10}[m_{H^+} (\text{GeV})]$")
    im = s.imshow(heatmap_v,extent=(tanb[0],tanb[-1],mH[0],mH[-1]),origin='lower')
    fig.colorbar(im)
    plt.title("Heatmap of Modification Factor for V"+us[i]+ds[i])
    plt.savefig("v"+us[i]+ds[i]+"_heatmap.png")

pool3 = Pool()
argus = [ckm_els,ckm_errs,par,err,heatmap,errmap]
testy = partial(testing,argus)
units = np.array(pool3.map(testy,ej)).reshape((steps,steps))
pool3.close()
pool3.join()

fig = plt.figure()
s = fig.add_subplot(1,1,1,xlabel=r"$\log_{10}[\tan\beta]$",ylabel=r"$\log_{10}[m_{H^+} (\text{GeV})]$")
im = s.imshow(units,extent=(tanb[0],tanb[-1],mH[0],mH[-1]),origin='lower',cmap='gray')
plt.title("Modification Regions Allowed By Unitarity \n of full CKM Matrix")
plt.savefig("ckm_full_mat.png")
#plt.show()

print("--- %s seconds ---" % (time.time() - start_time))
