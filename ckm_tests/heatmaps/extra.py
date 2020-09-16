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

def vcb(mu,md,tanb,mH):
    csr = mu/(mH**2)
    csl = md*(tanb/mH)**2
    return csr, csl

def rh(args,th):
    mu,md,mm = args
    tanb, mH = th
    r = ((mu-md*(10**tanb)**2)/(mu+md))*(mm/(10**mH))**2
    return 1/(1+r)

#def errors1(par,err,mus,mds,mms,tanb,mH,i,j,m,hp):
def errors1(args,th):
    par,err,mus,mds,mms,m = args
    tanb, mH = th
    hp = rh([par[mus[m]],par[mds[m]],par[mms[m]]],[tanb,mH])
    a1 = abs(rh([par[mus[m]]+err[mus[m]],par[mds[m]],par[mms[m]]],[tanb,mH])-hp)**2
    a2 = abs(rh([par[mus[m]],par[mds[m]]+err[mds[m]],par[mms[m]]],[tanb,mH])-hp)**2
    a3 = abs(rh([par[mus[m]],par[mds[m]],par[mms[m]]+err[mms[m]]],[tanb,mH])-hp)**2

    return np.sqrt(a1 + a2 + a3)

def vp(ckm_els,par,heatmap,i,j):
    Vub, Vus, Vcb = ckm_els[0,2], ckm_els[0,1], ckm_els[1,2]
    
    Vubp = heatmap['Vub'][i,j]*Vub
    Vusp = heatmap['Vus'][i,j]*Vus
    Vcbp = heatmap['Vcb'][i,j]*Vcb
    
    par['Vub'] = Vubp
    par['Vus'] = Vusp
    par['Vcb'] = Vcbp

    CKMs = get_ckm(par)
    r1,r2,r3 = 0,0,0
    for i in range(3):
        r1 += abs(CKMs[0,i])**2
        r2 += abs(CKMs[1,i])**2
        r3 += abs(CKMs[2,i])**2
    
    return r1, r2, r3

def errors2(ckm_els,ckm_errs,par,err,heatmap,errmap,i,j):
    central = vp(ckm_els,par,heatmap,i,j)

    at1,at2,at3 = 0,0,0
    for x in range(3):
        for y in range(3):
            c_errs = np.zeros((3,3))
            c_errs[x,y] = ckm_errs[x,y]
            ce = vp(ckm_els+c_errs,par,heatmap,i,j)
            at1 += abs(ce[0]-central[0])**2
            at2 += abs(ce[1]-central[1])**2
            at3 += abs(ce[2]-central[2])**2
            
    parl = {}
    parl['Vcb'] = 1*par['Vcb']
    parl['Vub'] = 1*par['Vub']
    parl['Vus'] = 1*par['Vus']
    parl['delta'] = par['delta']+err['delta']
    ced = vp(ckm_els,parl,heatmap,i,j)
    at1 += abs(ced[0]-central[0])**2
    at2 += abs(ced[1]-central[1])**2
    at3 += abs(ced[2]-central[2])**2

    heatmap1 = {}
    heatmap1['Vub'] = heatmap['Vub']+errmap['Vub']
    heatmap1['Vus'] = 1*heatmap['Vus']
    heatmap1['Vcb'] = 1*heatmap['Vcb']
    ce1 = vp(ckm_els,par,heatmap1,i,j)
    at1 += abs(ce1[0]-central[0])**2
    at2 += abs(ce1[1]-central[1])**2
    at3 += abs(ce1[2]-central[2])**2

    heatmap4 = {}
    heatmap4['Vub'] = 1*heatmap['Vub']
    heatmap4['Vus'] = heatmap['Vus']+errmap['Vus']
    heatmap4['Vcb'] = 1*heatmap['Vcb']
    ce4 = vp(ckm_els,par,heatmap4,i,j)
    at1 += abs(ce4[0]-central[0])**2
    at2 += abs(ce4[1]-central[1])**2
    at3 += abs(ce4[2]-central[2])**2

    heatmap6 = {}
    heatmap6['Vub'] = 1*heatmap['Vub']
    heatmap6['Vus'] = 1*heatmap['Vus']
    heatmap6['Vcb'] = 1*heatmap['Vcb']+errmap['Vcb']
    ce5 = vp(ckm_els,par,heatmap6,i,j)
    at1 += abs(ce5[0]-central[0])**2
    at2 += abs(ce5[1]-central[1])**2
    at3 += abs(ce5[2]-central[2])**2

    a1, a2, a3 = np.sqrt(at1), np.sqrt(at2), np.sqrt(at3)

    return 2*a1, 2*a2, 2*a3

def testing(args,ijs):
    ckm_els,ckm_errs,par,err,heatmap,errmap = args
    j,i = ijs
    r1, r2, r3 = vp(ckm_els,par,heatmap,i,j)
    re1, re2, re3 = errors2(ckm_els,ckm_errs,par,err,heatmap,errmap,i,j)
    rs1 = [r1,r1+re1,r1-re1]
    rs2 = [r2,r2+re2,r2-re2]
    rs3 = [r3,r3+re3,r3-re3]
    for i in rs1:
        for j in rs2:
            for k in rs3:
                if i < 1 and j < 1 and k < 1:
                    return 1
    return 0
#    if r1 < 1 and r2 < 1 and r3 < 1:
#        return 1
#    elif (r1 + re1) < 1 and r2 < 1 and r3 < 1:
#        return 1
#    elif (r1 - re1) < 1 and r2 < 1 and r3 < 1:
#        return 1
#    elif r1 < 1 and (r2 + re2) < 1 and r3 < 1:
#        return 1
#    elif r1 < 1 and (r2 - re2) < 1 and r3 < 1:
#        return 1
#    elif r1 < 1 and r2 < 1 and (r3 + re3) < 1:
#        return 1
#    elif r1 < 1 and r2 < 1 and (r3 - re3) < 1:
#        return 1
#    elif (r1 + re1) < 1 and (r2 + re2) < 1 and r3 < 1:
#        return 1
#    elif (r1 + re1) < 1 and (r2 - re2) < 1 and r3 < 1:
#        return 1
#    elif (r1 - re1) < 1 and (r2 + re2) < 1 and r3 < 1:
#        return 1
#    elif (r1 - re1) < 1 and (r2 - re2) < 1 and r3 < 1:
#        return 1
#    elif r1 < 1 and (r2 + re2) < 1 and (r3 + re3) < 1:
#        return 1
#    elif r1 < 1 and (r2 - re2) < 1 and (r3 + re3) < 1:
#        return 1
#    elif r1 < 1 and (r2 + re2) < 1 and (r3 - re3) < 1:
#        return 1
#    elif r1 < 1 and (r2 - re2) < 1 and (r3 - re3) < 1:
#        return 1
#    elif (r1 + re1) < 1 and r2 < 1 and (r3 + re3) < 1:
#        return 1
#    elif (r1 - re1) < 1 and r2 < 1 and (r3 + re3) < 1:
#        return 1
#    elif (r1 + re1) < 1 and r2 < 1 and (r3 - re3) < 1:
#        return 1
#    elif (r1 - re1) < 1 and r2 < 1 and (r3 - re3) < 1:
#        return 1
#    else: 
#        return 0
#
