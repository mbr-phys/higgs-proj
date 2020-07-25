#!/bin/env python3

import flavio
from flavio.physics.running.running import get_mt, get_alpha
import matplotlib.pyplot as plt
import numpy as np

def mHmin(contour):
    '''
        Finding the minimum mH and tanb range in the contours
    '''
    x = contour['x']
    y = contour['y']
    z = contour['z']
    levels = contour['levels']

    xf, yf = np.where(z==np.min(z))
    xbf = 10**x[xf[0],yf[0]]
    ybf = 10**y[xf[0],yf[0]]

    minh_loc, mint_loc, maxt_loc = [],[],[]
    for i in levels:
        minh, mint, maxt = 100,100,-2
        x_loc, y_loc = np.where(z<i)
        for j in range(len(x_loc)):
            k = (x_loc[j],y_loc[j])
            if y[k] < minh:
                minh = y[k]
            if x[k] < mint:
                mint = x[k]
            if x[k] > maxt:
                maxt = x[k]
        minh_loc.append(10**minh)
        mint_loc.append(10**mint)
        maxt_loc.append(10**maxt)

    return [xbf,ybf], minh_loc, mint_loc, maxt_loc

def bsgamma(par,tanb,mH):
    '''
        Find BR[B->Xsgamma] 2HDM contributions to Wilson Coeffs C7 & C8 as fns of mH+ and tanb
    '''
    mtmu = get_mt(par,2*par['m_W'])

    xtH = (mtmu/mH)**2

    F1_tH = (xtH**3 - 6*(xtH**2) + 3*xtH + 2 + 6*xtH*np.log(xtH))/(12*((xtH-1)**4))
    F2_tH = (2*xtH**3 + 3*xtH**2 - 6*xtH + 1 - 6*np.log(xtH)*xtH**2)/(12*((xtH-1)**4))
    F3_tH = (xtH**2 - 4*xtH + 3 + 2*np.log(xtH))/(2*((xtH-1)**3))
    F4_tH = (xtH**2 - 1 - 2*xtH*np.log(xtH))/(2*((xtH-1)**3))
    C_7H = -(xtH/2)*((1/(tanb**2))*((2/3)*F1_tH + F2_tH) + (2/3)*F3_tH + F4_tH)
    C_8H = -(xtH/2)*(F1_tH/(tanb**2) + F3_tH)

    return C_7H, C_8H

def bsgamma2(par,CKM,mub1,tanb,mH):
    '''
        Finding C7 and C8 2HDM contributions using 1903.10440
    '''
    def f1(b):
        i = (12*b*(np.log(b)-1)-3*(6*np.log(b)+1)*b**2 + 8*b**3 + 7)/((1-b)**4)
        return i
    def f2(b):
        i = (4*np.log(b)+3-2*b*(3*np.log(b)+4)+5*b**2)/((1-b)**3)
        return i
    def f3(b):
        i = (3*b*(2*np.log(b)+1)-6*b**2 + b**3 + 2)/((1-b)**4)
        return i
    def f4(b):
        i = (2*np.log(b)+3-4*b+b**2)/((1-b)**3)
        return i

    Vus, Vub = CKM[0,1], CKM[0,2]
    Vcs, Vcb = CKM[1,1], CKM[1,2]
    Vts, Vtb = CKM[2,1], CKM[2,2]
    vev,mW,QCD = par['vev'],par['m_W'],par['lam_QCD']
    mu = [par['m_u'],par['m_c'],get_mt(par,par['m_t'])]
    md = [par['m_d'],par['m_s'],par['m_b']]

    y = (mW/mH)**2
    cob = 1/tanb
    eu = np.array([[cob*mu[0]/vev,0,0],[0,cob*mu[1]/vev,0],[0,0,cob*mu[2]/vev]])
    ed = np.array([[-tanb*md[0]/vev,0,0],[0,-tanb*md[1]/vev,0],[0,0,-tanb*md[2]/vev]])
    zs = [(mu[0]/mH)**2,(mu[1]/mH)**2,(mu[2]/mH)**2]
    mub2 = 4.2
    mul, mulb = (mub1/mH)**2, (mub2/mub1)**2

    def c7_1():
        p1,p2 = 0,0
        for k in range(3):
            for n in range(3):
                p1 += np.conj(CKM[k,1])*eu[k,2]*eu[n,2]*CKM[n,2]
                p2 += np.conj(CKM[k,2])*eu[k,2]*CKM[2,n]*ed[n,2]
        c7 = -y*p1*f1(zs[2])/18 - mu[2]*y*p2*f2(zs[2])/(3*md[2])
        return c7/(Vtb*np.conj(Vts)*0.65**2)

    def c7p_1():
        p1,p2 = 0,0
        for k in range(3):
            for n in range(3):
                p1 += ed[k,2]*np.conj(CKM[2,k])*CKM[2,n]*ed[n,2]
                p2 += ed[k,2]*np.conj(CKM[2,k])*eu[n,2]*CKM[n,2]
        c7p = -y*p1*f1(zs[2])/18 - mu[2]*y*p2*f2(zs[2])/(3*md[2])
        return c7p/(Vtb*np.conj(Vts)*0.65**2)

    def c8_1():
        p1,p2 = 0,0
        for k in range(3):
            for n in range(3):
                p1 += np.conj(CKM[k,1])*eu[k,2]*eu[n,2]*CKM[n,2]
                p2 += np.conj(CKM[k,1])*eu[k,2]*CKM[2,n]*ed[n,2]
        c8 = -y*p1*f3(zs[2])/6 - mu[2]*y*p2*f4(zs[2])/md[2]
        return c8/(Vtb*np.conj(Vts)*0.65**2)

    def c8p_1():
        p1,p2 = 0,0
        for k in range(3):
            for n in range(3):
                p1 += ed[k,1]*np.conj(CKM[2,k])*CKM[2,n]*ed[n,2]
                p2 += ed[k,2]*np.conj(CKM[2,k])*eu[n,2]*CKM[n,2]
        c8p = -y*p1*f3(zs[2])/6 - mu[2]*y*p2*f4(zs[2])/md[2]
        return c8p/(Vtb*np.conj(Vts)*0.65**2)

    def c7_2():
        p1,p2 = 0,0
        for k in range(3):
            for n in range(3):
                p1 += np.conj(CKM[k,1])*eu[k,1]*eu[n,1]*CKM[n,2]
                p2 += np.conj(CKM[k,1])*eu[k,1]*CKM[1,n]*ed[n,2]
        c7 = -7*y*p1/18 - mu[1]*y*p2*(3+4*np.log(mul))/(3*md[2])
        return c7/(Vtb*np.conj(Vts)*0.65**2)

    def c7p_2():
        p1,p2 = 0,0
        for k in range(3):
            for n in range(3):
                p1 += ed[k,1]*np.conj(CKM[1,k])*CKM[1,n]*ed[n,2]
                p2 += ed[k,1]*np.conj(CKM[1,k])*eu[n,1]*CKM[n,2]
        c7p = -7*y*p1/18 - mu[1]*y*p2*(3+4*np.log(mul))/(3*md[2])
        return c7p/(Vtb*np.conj(Vts)*0.65**2)

    def c8_2():
        p1,p2 = 0,0
        for k in range(3):
            for n in range(3):
                p1 += np.conj(CKM[k,1])*eu[k,1]*eu[n,1]*CKM[n,2]
                p2 += np.conj(CKM[k,1])*eu[k,1]*CKM[1,n]*ed[n,2]
        c8 = -y*p1/3 - mu[1]*y*p2*(3+2*np.log(mul))/(md[2])
        return c8/(Vtb*np.conj(Vts)*0.65**2)

    def c8p_2():
        p1,p2 = 0,0
        for k in range(3):
            for n in range(3):
                p1 += ed[k,1]*np.conj(CKM[1,k])*CKM[1,n]*ed[n,2]
                p2 += ed[k,1]*np.conj(CKM[1,k])*eu[n,1]*CKM[n,2]
        c8p = -y*p1/3 - mu[1]*y*p2*(3+2*np.log(mul))/(md[2])
        return c8p/(Vtb*np.conj(Vts)*0.65**2)

    def c7_3():
        p1,p2 = 0,0
        for k in range(3):
            for n in range(3):
                p1 += np.conj(CKM[k,1])*eu[k,1]*CKM[1,n]*ed[n,2]
        c7 = -4*mu[1]*y*p1*np.log(mulb)/3
        return c7/(Vtb*np.conj(Vts)*0.65**2)

    def c7p_3():
        p1,p2 = 0,0
        for k in range(3):
            for n in range(3):
                p1 += ed[k,1]*np.conj(CKM[1,k])*eu[n,1]*CKM[n,2]
        c7p = -4*mu[1]*y*p1*np.log(mulb)/3
        return c7p/(Vtb*np.conj(Vts)*0.65**2)

    def c8_3():
        return 3*c7_3()/2

    def c8p_3():
        return 3*c7p_3()/2

    C7 = c7_1() + c7_2() + c7_3() # non-effective WCs
    C7p = c7p_1() + c7p_2() + c7p_3()
    C8 = c8_1() + c8_2() + c8_3()
    C8p = c8p_1() + c8p_2() + c8p_3()

    return C7, C7p, C8, C8p

def bmumu(par,msd,CKM,ml,mH0,tanb,mH):
    '''
        Find BR[B(s/d)->mumu] 2HDM contributions to Wilson Coeffs C10, C10', CS (=CP), CS' (=CP')
    '''
    mmu,mW,mb,mc,mu = par[ml],par['m_W'],par['m_b'],par['m_c'],par['m_u']
    wangle,higgs,v,QCD = par['s2w'],par['m_h'],par['vev'],par['lam_QCD']
    ms = par[msd[0]] # picking either ms or md
    Vub, Vcb, Vtb = CKM[0,2], CKM[1,2], CKM[2,2]
    Vus, Vcs, Vts = CKM[0,msd[1]], CKM[1,msd[1]], CKM[2,msd[1]]
    def I0(b):
        i = (1-3*b)/(-1+b) + 2*(b**2)*np.log(b)/((b-1)**2)
        return i
    def I1(b):
        i = -1/(b-1) + b*np.log(b)/((b-1)**2)
        return i
    def I2(b):
        i = (1-b)*I1(b)-1
        return i
    def I3(a,b):
        if a == b:
            i = -2*b*(b-1+(b-3)*np.log(b))/(b-1)
        else:
            i = (7*a-b)*b/(a-b) + 2*(b**2)*np.log(b)*(2*a**2 -b**2 -6*a+3*b+2*a*b)/((b-1)*(a-b)**2) - 6*(a**2)*b*np.log(a)/(a-b)**2
        return i
    def I4(a,b):
        if a == b:
            i = b*(b-1-np.log(b))/((b-1)**2)
        else:
            i = np.sqrt(b*a**3)*np.log(a)/((a-1)*(a-b)) - np.sqrt(a*b**3)*np.log(b)/((b-1)*(a-b))
        return i
    def I5(a,b):
        if a == b:
            i = (b+1+(b-2)*b*np.log(b))/((b-1)**2)
        else:
            i = -1+(a**2)*np.log(a)/((a-1)*(a-b)) - (b**2)*np.log(b)/((b-1)*(a-b))
        return i
    def I6(b):
        i = b*(b-1)*I1(b)
        return i
    def I7(b):
        i = -b*I1(b)
        return i
    mtmu = get_mt(par,mW)
    mtmut = get_mt(par,par['m_t'])

    cob,g2,b = 1/tanb,0.65,np.arctan(tanb)
    a = b - np.pi/2 # alignment limit
    z1,z2,y,yh,yH0 = (mu/mH)**2,(mc/mH)**2,(mW/mH)**2,(mH/higgs)**2,(mH/mH0)**2
    z3t,z3w = (mtmut/mH)**2,(mtmu/mH)**2
    el = np.sqrt(4*np.pi/137)
    #cba,sba = np.sin(2*b),-np.sin(2*b) # wrong sign limit
    cba,sba = np.cos(b-a),np.sin(b-a) # alignment limit
    Lp = (yh*cba**2 + yH0*sba**2)*(-2*tanb*mmu/v)
    Lm = -1*Lp
    lam3 = 0.1
    lamh = v*sba*lam3
    lamH0 = v*cba*lam3

    C10_1 = (np.conj(Vts)*Vtb/(2*np.conj(Vts)*Vtb*el**2))*(abs(cob*mtmut/v)**2)*(I1(z3t)-1)
    C10P_1 = -(abs(tanb/v)**2)*(np.conj(Vts)*Vtb*mb*ms/(2*np.conj(Vts)*Vtb*el**2))*(I1(z3t)-1)
    CS_1 = -(np.conj(tanb*ms/v)/((g2**4)*wangle*np.conj(Vts)*Vtb))*(-(y/2)*Lp*(4*I1(z3w)*(mtmu/mb)*(z3w-1)-2*np.log((mb/mH)**2)*(np.conj(Vts)*Vtb*(abs(cob*mtmu/v)**2))-I0(z3w)*np.conj(Vts)*Vtb*(abs(cob*mtmu/v)**2)+4*I5(z3w,z3w)*np.conj(Vts)*Vtb*(abs(cob*mtmu/v)**2))+2*I4(z3w,z3w)*np.conj(Vts)*Vtb*(abs(cob*mtmu/v)**2)*Lm*y-np.conj(Vts)*Vtb*np.conj(cob*mtmu/v)*((y*z3w)**0.5)*(-(tanb*mmu/v)-np.conj(tanb*mmu/v))*(2*(1-I1(z3w))*cba*g2*sba*(yh-yH0)+I1(z3w)*(y**0.5)*(cba*yh*lamh/mH - sba*yH0*lamH0/mH))) #CP
    CSP_1 = (1/((g2**4)*wangle*np.conj(Vts)*Vtb))*(y*Lm*(-2*I1(z3w)*(mtmu/mb)*(z3w-1)*((tanb*mtmu*(mb**2)/v**3)*np.conj(Vts)*Vtb - (tanb*(ms**2)*mtmu/v**3)*np.conj(Vts)*Vtb)+2*np.log((mb/mH)**2)*((cob*(mtmu**2)*mb/v**3)*np.conj(Vts)*Vtb+((tanb*(mb**2)*mtmu/v**3)*np.conj(Vts)*Vtb-(tanb*(ms**2)*mtmu/v**3)*np.conj(Vts)*Vtb)*mtmu/mb)-(tanb*mb/v)*(I7(z3w)*(abs(tanb*ms/v)**2)*np.conj(Vts)*Vtb+2*I5(z3w,z3w)*(abs(cob*mtmu/v)**2)*np.conj(Vts)*Vtb))+2*I4(z3w,z3w)*(cob*(mtmu**2)*mb/v**3)*np.conj(Vts)*Vtb*Lp*y+(mtmu*mb/v**2)*np.conj(Vts)*Vtb*((y*z3w)**0.5)*(-(tanb*mmu/v)+np.conj(-tanb*mmu/v))*(2*(1-I1(z3w))*cba*g2*sba*(yh-yH0)+I1(z3w)*(y**0.5)*(cba*yh*lamh/mH - sba*yH0*lamH0/mH))) #CP
    CS_2 = (np.conj(-tanb*ms/v)/(wangle*g2**2))*((z3w/4)*np.log((mb/mH)**2)*Lp+(1/8)*I3(y,z3w)*Lp+I2(z3w)*(-tanb*mmu/v)) #CP
    CSP_2 = (-tanb*mb/(v*wangle*g2**2))*((z3w/2)*np.log((mb/mH)**2)*Lm-(1/2)*I6(z3w)*Lm+I2(z3w)*(-tanb*mmu/v)) #CP
    C10_2 = -(pow(mW*mmu,2)/(wangle*np.conj(Vts)*Vtb*(mH**2)*pow(g2*v,4)))*((mu**2)*np.conj(Vus)*Vub*I1(z1)+(mc**2)*np.conj(Vcs)*Vcb*I1(z2)+(mtmut**2)*np.conj(Vts)*Vtb*I1(z3t))
    C10P_2 = -(pow(mmu*mW,2)*ms*mb*(tanb**4)/(wangle*Vtb*np.conj(Vts)*(mH**2)*pow(g2*v,4)))*(np.conj(Vus)*Vub*I1(z1)+np.conj(Vcs)*Vcb*I1(z2)+np.conj(Vts)*Vtb*I1(z3t))

    C10 = C10_1+C10_2
    C10P = C10P_1+C10P_2
    CS = CS_1+CS_2
    CSP = CSP_1+CSP_2

    return C10, C10P, CS, CSP

def bsll_c9(par,CKM,mls,tanb,mH):
    '''
        Calculating C9 and C9' 2HDM WCs
    '''
    def I1(b):
        i = -1/(b-1) + b*np.log(b)/((b-1)**2) 
        return i
    def I8(a,b):
        i = -1/((1-a)*(1-b))
        if a != b:
            i += (np.log(b)*b**2)/((a-b)*(1-b)**2) + (np.log(a)*a**2)/((b-a)*(1-a)**2)
        return i
    def f5(b):
        i1 = 2*(12*np.log(b)+19)-9*b*(4*np.log(b)+13)
        i2 = 126*b**2 + (18*np.log(b)-47)*b**3
        return (i1+i2)/((1-b)**4)

    Vus, Vub = CKM[0,1], CKM[0,2]
    Vcs, Vcb = CKM[1,1], CKM[1,2]
    Vts, Vtb = CKM[2,1], CKM[2,2]
    vev,mW,QCD,s2w,e = par['vev'],par['m_W'],par['lam_QCD'],par['s2w'],1.6e-19
    mu = [par['m_u'],par['m_c'],get_mt(par,par['m_t'])]
    md = [par['m_d'],par['m_s'],par['m_b']]
    ml = [par[mls[1]],par[mls[0]],par['m_tau']]
    y = (mW/mH)**2
    cob = 1/tanb
    eu = np.array([[cob*mu[0]/vev,0,0],[0,cob*mu[1]/vev,0],[0,0,cob*mu[2]/vev]])
    ed = np.array([[-tanb*md[0]/vev,0,0],[0,-tanb*md[1]/vev,0],[0,0,-tanb*md[2]/vev]])
    el = np.array([[-tanb*ml[0]/vev,0,0],[0,-tanb*ml[1]/vev,0],[0,0,-tanb*ml[2]/vev]])
    zs = [(mu[0]/mH)**2,(mu[1]/mH)**2,(mu[2]/mH)**2]
    mul = (4.2/mH)**2

    def c9_1():
        p1 = 0
        for k in range(3):
            for n in range(3):
                p1 += np.conj(CKM[k,1])*eu[k,2]*eu[n,2]*CKM[n,2]
        c9 = -p1*(1-4*s2w)*(I1(zs[2])-1)/(2*np.conj(Vts)*Vtb*e**2)
        return c9

    def c9p_1():
        p1 = 0
        for k in range(3):
            for n in range(3):
                p1 += ed[k,1]*np.conj(CKM[2,k])*CKM[2,n]*ed[n,2]
        c9p = p1*(1-4*s2w)*(I1(zs[2])-1)/(2*np.conj(Vts)*Vtb*e**2)
        return c9p

    def c9_2():
        p1 = 0 
        for k in range(3):
            for n in range(3):
                p1 += np.conj(CKM[k,1])*eu[k,2]*eu[n,2]*CKM[n,2]
        c9 = p1*y*f5(zs[2])/(27*Vtb*np.conj(Vts)*0.65**2)
        return c9

    def c9p_2():
        p1 = 0 
        for k in range(3):
            for n in range(3):
                p1 += ed[k,1]*np.conj(CKM[2,k])*CKM[2,n]*ed[n,2]
        c9p = p1*y*f5(zs[2])/(27*Vtb*np.conj(Vts)*0.65**2)
        return c9p

    def c9_3():
        p1 = 0 
        for k in range(3):
            for n in range(3):
                p1 += np.conj(CKM[k,1])*eu[k,1]*eu[n,1]*CKM[n,2]
        c9 = 2*p1*y*(19+12*np.log(mul))/(27*Vtb*np.conj(Vts)*0.65**2)
        return c9

    def c9p_3():
        p1 = 0 
        for k in range(3):
            for n in range(3):
                p1 += ed[k,1]*np.conj(CKM[1,k])*CKM[1,n]*ed[n,2]
        c9p = 2*p1*y*(19+12*np.log(mul))/(27*Vtb*np.conj(Vts)*0.65**2)
        return c9p

    def c9_4():
        p1 = 0 
        for k in range(3):
            for n in range(3):
                for i in range(3):
                    for m in range(3):
                        p1 += np.conj(CKM[k,1])*eu[k,i]*eu[n,i]*CKM[n,2]*I1(zs[i])*el[m,mls[2]]**2
        c9 = -y*p1/(s2w*Vtb*np.conj(Vts)*0.65**4)
        return c9

    def c9p_4():
        p1 = 0 
        for k in range(3):
            for n in range(3):
                for i in range(3):
                    for m in range(3):
                        p1 += ed[k,1]*np.conj(CKM[i,k])*CKM[i,n]*ed[n,2]*I1(zs[i])*el[m,mls[2]]**2
        c9p = -y*p1/(s2w*Vtb*np.conj(Vts)*0.65**4)
        return c9p

    C9 = c9_1() + c9_2() + c9_3() + c9_4()
    C9p = c9p_1() + c9p_2() + c9p_3() + c9p_4()

    return C9, C9p

def mixing(par,CKM,mds,tanb,mH):
    '''
        Find DeltaMq 2HDM contributions to Wilson Coeffs C1, C1', C2, C2', C4, C5 using 1903.10440
        mds is list ['m_s',1,'m_d'] or ['m_d',0,'m_s'] depending on Bs or Bd
    '''
    Vus, Vub = CKM[0,mds[1]], CKM[0,2]
    Vcs, Vcb = CKM[1,mds[1]], CKM[1,2]
    Vts, Vtb = CKM[2,mds[1]], CKM[2,2]
    vev,mW,QCD = par['vev'],par['m_W'],par['lam_QCD']
    mu = [par['m_u'],par['m_c'],get_mt(par,par['m_t'])]
    md = [par[mds[2]],par[mds[0]],par['m_b']]
    def I1(b):
        i = -1/(b-1) + b*np.log(b)/((b-1)**2) 
        return i
    def I8(a,b):
        i = -1/((1-a)*(1-b))
        if a != b:
            i += (np.log(b)*b**2)/((a-b)*(1-b)**2) + (np.log(a)*a**2)/((b-a)*(1-a)**2)
        return i
    def I9(a,b):
        i = -a*b/((1-a)*(1-b)) 
        if a != b:
            i += a*b*np.log(b)/((a-b)*(1-b)**2) + a*b*np.log(a)/((b-a)*(1-a)**2)
        return i
    def I10(a,b):
        i = -1/((1-a)*(1-b)) 
        if a != b:
            i += a*np.log(a)/((b-a)*(1-a)**2) + b*np.log(b)/((a-b)*(1-b)**2)
        return i
    def I11(a,b,c):
        i = -3*(a**2)*np.log(a)/((a-1)*(a-b)*(a-c)) 
        if b != c: 
            i += b*(4*a-b)*np.log(b)/((b-1)*(a-b)*(b-c)) + c*(4*a-c)*np.log(c)/((c-1)*(a-c)*(c-b))
        return i
    def I12(a,b):
        if a != b:
            i = a*b*np.log(a)/((1-a)*(a-b)) - a*b*np.log(b)/((1-b)*(a-b)) 
        else:
            i = 0
        return i

    y = (mW/mH)**2
    cob = 1/tanb
    eu = [cob*mu[0]/vev,cob*mu[1]/vev,cob*mu[2]/vev]
    ed = [-tanb*md[0]/vev,-tanb*md[1]/vev,-tanb*md[2]/vev]
    zs = [(mu[0]/mH)**2,(mu[1]/mH)**2,(mu[2]/mH)**2]
    v2 = [np.conj(Vus),np.conj(Vcs),np.conj(Vts)]
    v3 = [Vub,Vcb,Vtb]

    def c1_1():
        pref = -1/(32*(np.pi*mH)**2)
        ai, bi, ci = 0,0,0
        for i in range(3):
            ai += v2[i]*v3[i]*(eu[i]**2)*I8(zs[0],zs[i])
            bi += v2[i]*v3[i]*(eu[i]**2)*I8(zs[1],zs[i])
            ci += v2[i]*v3[i]*(eu[i]**2)*I8(zs[2],zs[i])
        a = (v2[0]*v3[0]*eu[0]**2)*ai
        b = (v2[1]*v3[1]*eu[1]**2)*bi
        c = (v2[2]*v3[2]*eu[2]**2)*ci
        return pref*(a+b+c) 

    def c1p():
        pref = -((ed[1]*ed[2])**2)/(32*(np.pi*mH)**2)
        ai, bi, ci = 0,0,0
        for i in range(3):
            ai += v2[i]*v3[i]*I9(zs[0],zs[i])
            bi += v2[i]*v3[i]*I9(zs[1],zs[i])
            ci += v2[i]*v3[i]*I9(zs[2],zs[i])
        a = (v2[0]*v3[0])*ai
        b = (v2[1]*v3[1])*bi
        c = (v2[2]*v3[2])*ci
        return pref*(a+b+c) 

    def c2():
        pref = -(ed[1]**2)/(8*(np.pi*mH)**2)
        ai, bi, ci = 0,0,0
        for i in range(3):
            ai += v2[i]*v3[i]*eu[i]*np.sqrt(zs[i])*I10(zs[i],zs[0])
            bi += v2[i]*v3[i]*eu[i]*np.sqrt(zs[i])*I10(zs[i],zs[1])
            ci += v2[i]*v3[i]*eu[i]*np.sqrt(zs[i])*I10(zs[i],zs[2])
        a = v2[0]*v3[0]*eu[0]*np.sqrt(zs[0])*ai
        b = v2[1]*v3[1]*eu[1]*np.sqrt(zs[1])*bi
        c = v2[2]*v3[2]*eu[2]*np.sqrt(zs[2])*ci
        return pref*(a+b+c) 

    def c2p():
        pref = -(ed[2]**2)/(8*(np.pi*mH)**2)
        ai, bi, ci = 0,0,0
        for i in range(3):
            ai += v2[i]*v3[i]*eu[i]*np.sqrt(zs[i])*I10(zs[i],zs[0])
            bi += v2[i]*v3[i]*eu[i]*np.sqrt(zs[i])*I10(zs[i],zs[1])
            ci += v2[i]*v3[i]*eu[i]*np.sqrt(zs[i])*I10(zs[i],zs[2])
        a = v2[0]*v3[0]*eu[0]*np.sqrt(zs[0])*ai
        b = v2[1]*v3[1]*eu[1]*np.sqrt(zs[1])*bi
        c = v2[2]*v3[2]*eu[2]*np.sqrt(zs[2])*ci
        return pref*(a+b+c) 

    def c4_1():
        pref = -(ed[1]*ed[2]/(4*(np.pi*mH)**2))
        ai, bi, ci = 0,0,0
        for i in range(3):
            ai += v2[i]*v3[i]*eu[i]*np.sqrt(zs[i])*I10(zs[i],zs[0])
            bi += v2[i]*v3[i]*eu[i]*np.sqrt(zs[i])*I10(zs[i],zs[1])
            ci += v2[i]*v3[i]*eu[i]*np.sqrt(zs[i])*I10(zs[i],zs[2])
        a = v2[0]*v3[0]*eu[0]*np.sqrt(zs[0])*ai
        b = v2[1]*v3[1]*eu[1]*np.sqrt(zs[1])*bi
        c = v2[2]*v3[2]*eu[2]*np.sqrt(zs[2])*ci
        return pref*(a+b+c) 

    def c5():
        pref = ed[1]*ed[2]/(8*(np.pi*mH)**2)
        ai, bi, ci = 0,0,0
        for i in range(3):
            ai += v2[i]*v3[i]*(eu[i]**2)*(I8(zs[0],zs[i])+I1(zs[0]))
            bi += v2[i]*v3[i]*(eu[i]**2)*(I8(zs[1],zs[i])+I1(zs[1]))
            ci += v2[i]*v3[i]*(eu[i]**2)*(I8(zs[2],zs[i])+I1(zs[2]))
        a = v2[0]*v3[0]*ai
        b = v2[1]*v3[1]*bi
        c = v2[2]*v3[2]*ci
        return pref*(a+b+c) 

    def c1_2():
        pref = (0.65**2)/(64*(np.pi*mW)**2)
        ai, bi, ci = 0,0,0
        for i in range(3):
            ai += v2[i]*v3[i]*eu[i]*np.sqrt(zs[i])*I11(y,zs[0],zs[i])
            bi += v2[i]*v3[i]*eu[i]*np.sqrt(zs[i])*I11(y,zs[1],zs[i])
            ci += v2[i]*v3[i]*eu[i]*np.sqrt(zs[i])*I11(y,zs[2],zs[i])
        a = v2[0]*v3[0]*eu[0]*np.sqrt(zs[0])*ai
        b = v2[1]*v3[1]*eu[1]*np.sqrt(zs[1])*bi
        c = v2[2]*v3[2]*eu[2]*np.sqrt(zs[2])*ci
        return pref*(a+b+c) 

    def c4_2():
        pref = -(ed[1]*ed[2]*0.65**2)/(16*(np.pi*mW)**2)
        ai, bi, ci = 0,0,0
        for i in range(3):
            ai += v2[i]*v3[i]*I12(zs[0],zs[i])
            bi += v2[i]*v3[i]*I12(zs[1],zs[i])
            ci += v2[i]*v3[i]*I12(zs[2],zs[i])
        a = v2[0]*v3[0]*ai
        b = v2[1]*v3[1]*bi
        c = v2[2]*v3[2]*ci
        return pref*(a+b+c) 

    CVLL = (c1_1() + c1_2())
    CVRR = c1p()
    CSLL = c2()
    CSRR = c2p()
    CSLR = (c4_1() + c4_2())
    CVLR = 2*c5()

    return CVLL, CVRR, CSLL, CSRR, CSLR, CVLR

def mixing2(par,ckm_els,tanb,mH):
    '''
        LO Expressions for 2HDM contributions to DeltaM_q
    '''
    mtmu = get_mt(par,par['m_W'])
    mtmut = get_mt(par,par['m_t'])
    etat = (get_alpha(par,par['m_W'],nf_out=5)['alpha_s']/get_alpha(par,par['m_b'],nf_out=5)['alpha_s'])**(6/23)

    x_tH1 = (mtmut/mH)**2
    x_tW1 = (mtmut/par['m_W'])**2
    x_tH = (mtmu/mH)**2
    x_tW = (mtmu/par['m_W'])**2
    S_WH = (x_tH1*x_tW1/(4*tanb**2))*((2*x_tW1-8*x_tH1)*np.log(x_tH1)/((x_tH1-x_tW1)*(1-x_tH1)**2) + 6*x_tW1*np.log(x_tW1)/((x_tH1-x_tW1)*(1-x_tW1)**2) - (8-2*x_tW1)/((1-x_tW1)*(1-x_tH1)))
    S_HH = (x_tH1*x_tW1/(4*tanb**4))*((1+x_tH1)/((1-x_tH1)**2)+2*x_tH1*np.log(x_tH1)/((1-x_tH1)**3))

    pref = ((1/4)*(par['GF']/np.pi)**2)*(par['m_W']**2)*ckm_els*etat

    return pref*(S_WH + S_HH)

def rh(mu,md,tanb,mH):
    '''
        Function for M->lnu 2HDM WC contribution, based on rH we used from 0907.5135

        I think from looking at the operators, the 2HDM contributions appear in the CSR and CSL WCs, where (m_M**2/(m_l*(m_u+m_d))*(CSR-CSL) = rH

        Used https://github.com/flav-io/flavio/blob/master/flavio/physics/bdecays/blnu.py - line 22 - to figure this out
    '''
    csr = mu/(mH**2)
    csl = md*(tanb/mH)**2
    return csr, csl

#def rat_d(par,m_l,m_u,m_d,tanb,mH):
#    '''
#        Function for WCs of semileptonics in 2HDM
#
#        I think this is right for the WCs, I've tried to derive it from the Lagrangian given in
#        https://arxiv.org/pdf/1705.02465.pdf
#
#        I don't think we need this actually - the WCs should be the same for tree-level leps and semi-leps so will us rh above
#    '''
#    ml, mu, md = par[ml], par[m_u], par[m_d]
#    Gf, vev = par['GF'], par['vev']
#    csl = ml*mu/(np.sqrt(2)*Gf*(vev*mH**2))
#    csr = (ml*md*tanb**2)/(np.sqrt(2)*Gf*(vev*mH**2))
#
#    return csl, csr
