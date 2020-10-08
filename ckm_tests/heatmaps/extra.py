import flavio
import numpy as np
from flavio.physics.ckm import get_ckm

def get_ckm_alex(par):

    vub = par['Vub']
    vcb = par['Vcb']
    vus = par['Vus']
    delta = par['delta']

    s13 = vub
    c13 = np.sqrt(1-s13**2)
    vvub = s13*np.exp(-1j*delta)
    s23 = vcb/c13
    c23 = np.sqrt(1-s23**2)
    s12 = vus/c13
    c12 = np.sqrt(1-s12**2)
#    if abs(s12) > 1:
#        message = "conflict"
#    else:
#        message = "fine"
    vvud = c12*c13;
    vvcd = -s12*c23 - c12*s23*s13*np.exp(1j*delta)
    vvcs = c12*c23 - s12*s23*s13*np.exp(1j*delta)
    vvtd = s12*s23 - c12*c23*s13*np.exp(1j*delta)
    vvts = -c12*s23 - s12*c23*s13*np.exp(1j*delta)
    vvtb = c23*c13

    CKM = np.array([[vvud,vus,vvub],[vvcd,vvcs,vcb],[vvtd,vvts,vvtb]])

    return CKM#, message
#    return s12

def ckm_err(t_ckm,par,err):
    '''
        Get out CKM elements and their (Gaussian) errors - can't see how to do this with flavio
    '''
    centrals = t_ckm(par)

    par_e1 = {'Vub':par['Vub']+err['Vub'],'Vus':par['Vus'],'Vcb':par['Vcb'],'delta':par['delta']}
    a1 = abs(t_ckm(par_e1)-centrals)**2

    par_e2 = {'Vub':par['Vub'],'Vus':par['Vus']+err['Vus'],'Vcb':par['Vcb'],'delta':par['delta']}
    a2 = abs(t_ckm(par_e2)-centrals)**2

    par_e3 = {'Vub':par['Vub'],'Vus':par['Vus'],'Vcb':par['Vcb']+err['Vcb'],'delta':par['delta']}
    a3 = abs(t_ckm(par_e3)-centrals)**2

    par_e4 = {'Vub':par['Vub'],'Vus':par['Vus'],'Vcb':par['Vcb'],'delta':par['delta']+err['delta']}
    a4 = abs(t_ckm(par_e4)-centrals)**2

    return abs(centrals), np.sqrt(a1 + a2 + a3 + a4)

def vcb(mu,md,ml,tanb,mH):
    csr = mu*ml/(mH**2)
    csl = md*ml*(tanb/mH)**2
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

    CKMs = get_ckm_alex(par)
    CKMt = np.conj(CKMs.T)
    unit = CKMt*CKMs
    r1,r2,r3 = 0,0,0
    for i in range(3):
        r1 += unit[0,i]
        r2 += unit[1,i]
        r3 += unit[2,i]

#    return s12
    return abs(r1), abs(r2), abs(r3), CKMs
#    return unit

def errors2(ckm_els,ckm_errs,par,err,heatmap,errmap,i,j):
    central = vp(ckm_els,par,heatmap,i,j)

    at1,at2,at3,at4 = 0,0,0,0
    for x in range(3):
        for y in range(3):
            c_errs = np.zeros((3,3))
            c_errs[x,y] = ckm_errs[x,y]
            ce = vp(ckm_els+c_errs,par,heatmap,i,j)
            at1 += abs(ce[0]-central[0])**2
            at2 += abs(ce[1]-central[1])**2
            at3 += abs(ce[2]-central[2])**2
            at4 += abs(ce[3]-central[3])**2

    parl = {}
    parl['Vcb'] = 1*par['Vcb']
    parl['Vub'] = 1*par['Vub']
    parl['Vus'] = 1*par['Vus']
    parl['delta'] = par['delta']+err['delta']
    ced = vp(ckm_els,parl,heatmap,i,j)
    at1 += abs(ced[0]-central[0])**2
    at2 += abs(ced[1]-central[1])**2
    at3 += abs(ced[2]-central[2])**2
    at4 += abs(ced[3]-central[3])**2

    heatmap1 = {}
    heatmap1['Vub'] = heatmap['Vub']+errmap['Vub']
    heatmap1['Vus'] = 1*heatmap['Vus']
    heatmap1['Vcb'] = 1*heatmap['Vcb']
    ce1 = vp(ckm_els,par,heatmap1,i,j)
    at1 += abs(ce1[0]-central[0])**2
    at2 += abs(ce1[1]-central[1])**2
    at3 += abs(ce1[2]-central[2])**2
    at4 += abs(ce1[3]-central[3])**2

    heatmap4 = {}
    heatmap4['Vub'] = 1*heatmap['Vub']
    heatmap4['Vus'] = heatmap['Vus']+errmap['Vus']
    heatmap4['Vcb'] = 1*heatmap['Vcb']
    ce4 = vp(ckm_els,par,heatmap4,i,j)
    at1 += abs(ce4[0]-central[0])**2
    at2 += abs(ce4[1]-central[1])**2
    at3 += abs(ce4[2]-central[2])**2
    at4 += abs(ce4[3]-central[3])**2

    heatmap6 = {}
    heatmap6['Vub'] = 1*heatmap['Vub']
    heatmap6['Vus'] = 1*heatmap['Vus']
    heatmap6['Vcb'] = 1*heatmap['Vcb']+errmap['Vcb']
    ce5 = vp(ckm_els,par,heatmap6,i,j)
    at1 += abs(ce5[0]-central[0])**2
    at2 += abs(ce5[1]-central[1])**2
    at3 += abs(ce5[2]-central[2])**2
    at4 += abs(ce5[3]-central[3])**2

    a1, a2, a3, a4 = np.sqrt(at1), np.sqrt(at2), np.sqrt(at3), np.sqrt(at4)
#    a1 = np.sqrt(at1)

    return 2*a1, 2*a2, 2*a3, 2*a4

def testing(args,ijs):
    ckm_els,ckm_errs,par,err,heatmap,errmap,r1_lim,r2_lim,r3_lim = args
    j,i = ijs
#    heatmap['Vub'] = 10**heatmap['Vub']
#    heatmap['Vus'] = 10**heatmap['Vus']
#    heatmap['Vcb'] = 10**heatmap['Vcb']
#    errmap['Vub'] = 10**errmap['Vub']
#    errmap['Vus'] = 10**errmap['Vus']
#    errmap['Vcb'] = 10**errmap['Vcb']
#    r1 = vp(ckm_els,par,heatmap,i,j)
#    re1 = errors2(ckm_els,ckm_errs,par,err,heatmap,errmap,i,j)
#    if r1 < 1 or (r1-re1) < 1 or (r1+re1) < 1:
#        return 1
#    else:
#        return 0
    r1, r2, r3, ckms = vp(ckm_els,par,heatmap,i,j)
    re1, re2, re3, c_err = errors2(ckm_els,ckm_errs,par,err,heatmap,errmap,i,j)
    ckms = abs(ckms)
    fac = np.min([ckms[0,2]/ckm_els[0,2],ckms[1,2]/ckm_els[1,2],ckms[0,1]/ckm_els[0,1]])
    fac2 = np.max([ckms[0,2]/ckm_els[0,2],ckms[1,2]/ckm_els[1,2],ckms[0,1]/ckm_els[0,1]])
    if fac2 > 1.0:
        fac = np.min([fac,2-fac2])
    rs1 = [r1,r1+re1,r1-re1]
    rs2 = [r2,r2+re2,r2-re2]
    rs3 = [r3,r3+re3,r3-re3]
    message = np.array([[False,False,False],[False,False,False],[False,False,False]])
    for i in rs1:
        for j in rs2:
            for k in rs3:
                if (1-r1_lim) < i < (1+r1_lim) and (1-r2_lim) < j < (1+r2_lim) and (1-r3_lim) < k < (1+r3_lim):
                    for u in range(3):
                        for d in range(3):
                            if ckm_els[u,d]*fac <= ckms[u,d] <= ckm_els[u,d]*(2-fac):
                                message[u,d] = message[u,d] or True
#                                print(u,d,1)
                            elif (ckm_els[u,d]-2*ckm_errs[u,d])*fac <= ckms[u,d] <= (ckm_els[u,d]-2*ckm_errs[u,d])*(2-fac):
                                message[u,d] = message[u,d] or True
#                                print(u,d,2)
                            elif (ckm_els[u,d]+2*ckm_errs[u,d])*fac <= ckms[u,d] <= (ckm_els[u,d]+2*ckm_errs[u,d])*(2-fac):
                                message[u,d] = message[u,d] or True
#                                print(u,d,3)
                            elif (ckm_els[u,d]+2*ckm_errs[u,d])*fac <= ckms[u,d] <= (ckm_els[u,d]-2*ckm_errs[u,d])*(2-fac):
                                message[u,d] = message[u,d] or True
#                                print(u,d,4)
                            elif (ckm_els[u,d]-2*ckm_errs[u,d])*fac <= ckms[u,d] <= (ckm_els[u,d]+2*ckm_errs[u,d])*(2-fac):
                                message[u,d] = message[u,d] or True
#                                print(u,d,5)
                            elif ckm_els[u,d]*fac <= (ckms[u,d]+c_err[u,d]) <= ckm_els[u,d]*(2-fac):
                                message[u,d] = message[u,d] or True
#                                print(u,d,6)
                            elif (ckm_els[u,d]-2*ckm_errs[u,d])*fac <= (ckms[u,d]+c_err[u,d]) <= (ckm_els[u,d]-2*ckm_errs[u,d])*(2-fac):
                                message[u,d] = message[u,d] or True
#                                print(u,d,7)
                            elif (ckm_els[u,d]+2*ckm_errs[u,d])*fac <= (ckms[u,d]+c_err[u,d]) <= (ckm_els[u,d]+2*ckm_errs[u,d])*(2-fac):
                                message[u,d] = message[u,d] or True
#                                print(u,d,8)
                            elif (ckm_els[u,d]+2*ckm_errs[u,d])*fac <= (ckms[u,d]+c_err[u,d]) <= (ckm_els[u,d]-2*ckm_errs[u,d])*(2-fac):
                                message[u,d] = message[u,d] or True
#                                print(u,d,9)
                            elif (ckm_els[u,d]-2*ckm_errs[u,d])*fac <= (ckms[u,d]+c_err[u,d]) <= (ckm_els[u,d]+2*ckm_errs[u,d])*(2-fac):
                                message[u,d] = message[u,d] or True
#                                print(u,d,10)
                            elif ckm_els[u,d]*fac <= (ckms[u,d]-c_err[u,d]) <= ckm_els[u,d]*(2-fac):
                                message[u,d] = message[u,d] or True
#                                print(u,d,11)
                            elif (ckm_els[u,d]-2*ckm_errs[u,d])*fac <= (ckms[u,d]-c_err[u,d]) <= (ckm_els[u,d]-2*ckm_errs[u,d])*(2-fac):
                                message[u,d] = message[u,d] or True
#                                print(u,d,12)
                            elif (ckm_els[u,d]+2*ckm_errs[u,d])*fac <= (ckms[u,d]-c_err[u,d]) <= (ckm_els[u,d]+2*ckm_errs[u,d])*(2-fac):
                                message[u,d] = message[u,d] or True
#                                print(u,d,13)
                            elif (ckm_els[u,d]+2*ckm_errs[u,d])*fac <= (ckms[u,d]-c_err[u,d]) <= (ckm_els[u,d]-2*ckm_errs[u,d])*(2-fac):
                                message[u,d] = message[u,d] or True
#                                print(u,d,14)
                            elif (ckm_els[u,d]-2*ckm_errs[u,d])*fac <= (ckms[u,d]-c_err[u,d]) <= (ckm_els[u,d]+2*ckm_errs[u,d])*(2-fac):
                                message[u,d] = message[u,d] or True
#                                print(u,d,15)
    answer = 0
    for m in range(3):
        for n in range(3):
            if message[m,n]:
                answer += 1
    result = 0
    if answer == 9:
        result = 1
    return result
