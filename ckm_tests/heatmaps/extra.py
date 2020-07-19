import flavio
import numpy as np
from flavio.physics.ckm import get_ckm

def ckm_err(par,err):
    '''
        Get out CKM elements and their (Gaussian) errors - can't see how to do this with flavio
    '''
    centrals = get_ckm(par)

    par_e1 = {'Vub':par['Vub']+err['Vub'],'Vus':par['Vus'],'Vcb':par['Vcb'],'delta':par['delta']}
    a1 = abs(get_ckm(par_e1)-centrals)**2

    par_e2 = {'Vub':par['Vub'],'Vus':par['Vus']+err['Vus'],'Vcb':par['Vcb'],'delta':par['delta']}
    a2 = abs(get_ckm(par_e2)-centrals)**2

    par_e3 = {'Vub':par['Vub'],'Vus':par['Vus'],'Vcb':par['Vcb']+err['Vcb'],'delta':par['delta']}
    a3 = abs(get_ckm(par_e3)-centrals)**2

    par_e4 = {'Vub':par['Vub'],'Vus':par['Vus'],'Vcb':par['Vcb'],'delta':par['delta']+err['delta']}
    a4 = abs(get_ckm(par_e4)-centrals)**2

    return abs(centrals), np.sqrt(a1 + a2 + a3 + a4)

def rh(mu,md,mm,tanb,mH):
    r = ((mu-md*tanb**2)/(mu+md))*(mm/mH)**2
    return 1/(1+r)

def errors1(par,err,mus,mds,mms,tanb,mH,i,j,m,hp):
    a1 = abs(rh(par[mus[m]]+err[mus[m]],par[mds[m]],par[mms[m]],10**tanb[j],10**mH[i])-hp)**2
    a2 = abs(rh(par[mus[m]],par[mds[m]]+err[mds[m]],par[mms[m]],10**tanb[j],10**mH[i])-hp)**2
    a3 = abs(rh(par[mus[m]],par[mds[m]],par[mms[m]]+err[mms[m]],10**tanb[j],10**mH[i])-hp)**2

    return np.sqrt(a1 + a2 + a3)

def vp(ckm_els,heatmap,i,j):
    Vub, Vcd, Vcs, Vus, Vud, Vcb = ckm_els[0,2], ckm_els[1,0], ckm_els[1,1], ckm_els[0,1], ckm_els[0,0], ckm_els[1,2]
    Vubp = heatmap['mB'][i,j]*Vub
    Vcdp = heatmap['mD'][i,j]*Vcd
    Vcsp = heatmap['mDs'][i,j]*Vcs
    Vusp = heatmap['mK'][i,j]*Vus
    Vudp = heatmap['mpi'][i,j]*Vud
    #row1 = abs(Vudp)**2 + abs(Vusp)**2 + abs(Vubp)**2
    #row2 = abs(Vcdp)**2 + abs(Vcsp)**2 + abs(Vcb)**2 
    row1 = Vudp**2 + Vusp**2 + Vubp**2
    row2 = Vcdp**2 + Vcsp**2 + Vcb**2 

    return row1, row2

def errors2(ckm_els,ckm_errs,heatmap,errmap,i,j):
    central = vp(ckm_els,heatmap,i,j)

    at1,at2 = 0,0
    for x in range(2):
        for y in range(3):
            c_errs = np.zeros((3,3))
            c_errs[x,y] = ckm_errs[x,y]
            ce = vp(ckm_els+c_errs,heatmap,i,j)
            at1 += abs(ce[0]-central[0])**2
            at2 += abs(ce[1]-central[1])**2

    heatmap1 = {}
    heatmap1['mB'] = heatmap['mB']+errmap['mB']
    heatmap1['mD'] = 1*heatmap['mD']
    heatmap1['mDs'] = 1*heatmap['mDs']
    heatmap1['mK'] = 1*heatmap['mK']
    heatmap1['mpi'] = 1*heatmap['mpi']
    ce1 = vp(ckm_els,heatmap1,i,j)
    at1 += abs(ce1[0]-central[0])**2
    at2 += abs(ce1[1]-central[1])**2

    heatmap2 = {}
    heatmap2['mB'] = 1*heatmap['mB']
    heatmap2['mD'] = heatmap['mD']+errmap['mD']
    heatmap2['mDs'] = 1*heatmap['mDs']
    heatmap2['mK'] = 1*heatmap['mK']
    heatmap2['mpi'] = 1*heatmap['mpi']
    ce2 = vp(ckm_els,heatmap2,i,j)
    at1 += abs(ce2[0]-central[0])**2
    at2 += abs(ce2[1]-central[1])**2

    heatmap3 = {}
    heatmap3['mB'] = 1*heatmap['mB']
    heatmap3['mD'] = 1*heatmap['mD']
    heatmap3['mDs'] = heatmap['mDs']+errmap['mDs']
    heatmap3['mK'] = 1*heatmap['mK']
    heatmap3['mpi'] = 1*heatmap['mpi']
    ce3 = vp(ckm_els,heatmap3,i,j)
    at1 += abs(ce3[0]-central[0])**2
    at2 += abs(ce3[1]-central[1])**2

    heatmap4 = {}
    heatmap4['mB'] = 1*heatmap['mB']
    heatmap4['mD'] = 1*heatmap['mD']
    heatmap4['mDs'] = 1*heatmap['mDs']
    heatmap4['mK'] = heatmap['mK']+errmap['mK']
    heatmap4['mpi'] = 1*heatmap['mpi']
    ce4 = vp(ckm_els,heatmap4,i,j)
    at1 += abs(ce4[0]-central[0])**2
    at2 += abs(ce4[1]-central[1])**2

    heatmap5 = {}
    heatmap5['mB'] = 1*heatmap['mB']
    heatmap5['mD'] = 1*heatmap['mD']
    heatmap5['mDs'] = 1*heatmap['mDs']
    heatmap5['mK'] = 1*heatmap['mK']
    heatmap5['mpi'] = heatmap['mpi']+errmap['mpi']
    ce5 = vp(ckm_els,heatmap5,i,j)
    at1 += abs(ce5[0]-central[0])**2
    at2 += abs(ce5[1]-central[1])**2

    a1, a2 = np.sqrt(at1), np.sqrt(at2)

    return 2*a1, 2*a2