#!/bin/env python3

import flavio
from flavio.classes import Parameter
from flavio.physics.running.running import get_mt, get_alpha_e
from flavio.statistics.functions import pvalue
from scipy.integrate import quad
import matplotlib.pyplot as plt
import numpy as np

ims = ['BKll Observables 1','BKll Observables 2','BKll Observables 3','BKll Observables 4','BKll Observables 5','BKll Observables 6']

def mHmin(contour,minz=0):
    '''
        Finding the minimum mH and tanb range in the contours
    '''
    x = contour['x']
    y = contour['y']
    z = contour['z']
    levels = contour['levels']
    
    if minz == 0:
        mince = np.min(z)
    else:
        mince = minz
    z = z - mince

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

    return [xbf,ybf], minh_loc, mint_loc, maxt_loc, mince

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
    mub2 = 4.18
    mul, mulb = (mub1/mH)**2, (mub2/mub1)**2

    def c7_1():
        p1,p2 = 0,0
        for k in range(3):
            for n in range(3):
                p1 += np.conj(CKM[k,1])*eu[k,2]*eu[n,2]*CKM[n,2]
                p2 += np.conj(CKM[k,1])*eu[k,2]*CKM[2,n]*ed[n,2]
        c7 = -y*p1*f1(zs[2])/18 - mu[2]*y*p2*f2(zs[2])/(3*md[2])
        return c7/(Vtb*np.conj(Vts)*0.652**2)

    def c7p_1():
        p1,p2 = 0,0
        for k in range(3):
            for n in range(3):
                p1 += ed[k,1]*np.conj(CKM[2,k])*CKM[2,n]*ed[n,2]
                p2 += ed[k,1]*np.conj(CKM[2,k])*eu[n,2]*CKM[n,2]
        c7p = -y*p1*f1(zs[2])/18 - mu[2]*y*p2*f2(zs[2])/(3*md[2])
        return c7p/(Vtb*np.conj(Vts)*0.652**2)

    def c8_1():
        p1,p2 = 0,0
        for k in range(3):
            for n in range(3):
                p1 += np.conj(CKM[k,1])*eu[k,2]*eu[n,2]*CKM[n,2]
                p2 += np.conj(CKM[k,1])*eu[k,2]*CKM[2,n]*ed[n,2]
        c8 = -y*p1*f3(zs[2])/6 - mu[2]*y*p2*f4(zs[2])/md[2]
        return c8/(Vtb*np.conj(Vts)*0.652**2)

    def c8p_1():
        p1,p2 = 0,0
        for k in range(3):
            for n in range(3):
                p1 += ed[k,1]*np.conj(CKM[2,k])*CKM[2,n]*ed[n,2]
                p2 += ed[k,1]*np.conj(CKM[2,k])*eu[n,2]*CKM[n,2]
        c8p = -y*p1*f3(zs[2])/6 - mu[2]*y*p2*f4(zs[2])/md[2]
        return c8p/(Vtb*np.conj(Vts)*0.652**2)

    def c7_2():
        p1,p2 = 0,0
        for k in range(3):
            for n in range(3):
                p1 += np.conj(CKM[k,1])*eu[k,1]*eu[n,1]*CKM[n,2]
                p2 += np.conj(CKM[k,1])*eu[k,1]*CKM[1,n]*ed[n,2]
        c7 = -7*y*p1/18 - mu[1]*y*p2*(3+4*np.log(mul))/(3*md[2])
        return c7/(Vtb*np.conj(Vts)*0.652**2)

    def c7p_2():
        p1,p2 = 0,0
        for k in range(3):
            for n in range(3):
                p1 += ed[k,1]*np.conj(CKM[1,k])*CKM[1,n]*ed[n,2]
                p2 += ed[k,1]*np.conj(CKM[1,k])*eu[n,1]*CKM[n,2]
        c7p = -7*y*p1/18 - mu[1]*y*p2*(3+4*np.log(mul))/(3*md[2])
        return c7p/(Vtb*np.conj(Vts)*0.652**2)

    def c8_2():
        p1,p2 = 0,0
        for k in range(3):
            for n in range(3):
                p1 += np.conj(CKM[k,1])*eu[k,1]*eu[n,1]*CKM[n,2]
                p2 += np.conj(CKM[k,1])*eu[k,1]*CKM[1,n]*ed[n,2]
        c8 = -y*p1/3 - mu[1]*y*p2*(3+2*np.log(mul))/(md[2])
        return c8/(Vtb*np.conj(Vts)*0.652**2)

    def c8p_2():
        p1,p2 = 0,0
        for k in range(3):
            for n in range(3):
                p1 += ed[k,1]*np.conj(CKM[1,k])*CKM[1,n]*ed[n,2]
                p2 += ed[k,1]*np.conj(CKM[1,k])*eu[n,1]*CKM[n,2]
        c8p = -y*p1/3 - mu[1]*y*p2*(3+2*np.log(mul))/(md[2])
        return c8p/(Vtb*np.conj(Vts)*0.652**2)

    def c7_3():
        p1,p2 = 0,0
        for k in range(3):
            for n in range(3):
                p1 += np.conj(CKM[k,1])*eu[k,1]*CKM[1,n]*ed[n,2]
        c7 = -4*mu[1]*y*p1*np.log(mulb)/(3*md[2])
        return c7/(Vtb*np.conj(Vts)*0.652**2)

    def c7p_3():
        p1,p2 = 0,0
        for k in range(3):
            for n in range(3):
                p1 += ed[k,1]*np.conj(CKM[1,k])*eu[n,1]*CKM[n,2]
        c7p = -4*mu[1]*y*p1*np.log(mulb)/(3*md[2])
        return c7p/(Vtb*np.conj(Vts)*0.652**2)

    def c8_3():
        return 3*c7_3()/2

    def c8p_3():
        return 3*c7p_3()/2

    C7 = c7_1() + c7_2() + c7_3() # non-effective WCs
    C7p = c7p_1() + c7p_2() + c7p_3()
    C8 = c8_1() + c8_2() + c8_3()
    C8p = c8p_1() + c8p_2() + c8p_3()

    return C7, C7p, C8, C8p

def bsll(par,CKM,mss,mls,mH0,tanb,mH,ali):
    '''
        Calculating C9, C9', C10, C10', CS, CS', CP, CP' 2HDM WCs
    '''
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
            i = (b-1+(b-2)*b*np.log(b))/((b-1)**2)
        else:
            if b == 0:
                i = -1 + a*np.log(a)/(a-1)
            else:
                i = -1+(a**2)*np.log(a)/((a-1)*(a-b)) - (b**2)*np.log(b)/((b-1)*(a-b))
        return i
    def I6(b):
        i = b*(b-1)*I1(b)
        return i
    def I7(b):
        i = -b*I1(b)
        return i
    def I8(a,b):
        if a != b:
            i = -1/((1-a)*(1-b)) + (np.log(b)*b**2)/((a-b)*(1-b)**2) + (np.log(a)*a**2)/((b-a)*(1-a)**2)
        else:
            i = (1-(a**2)+2*a*np.log(a))/((a-1)**3)
        return i
    def f5(b):
        i1 = (2*(12*np.log(b)+19)-9*b*(4*np.log(b)+13))/((1-b)**4)
        i2 = (126*b**2 + (18*np.log(b)-47)*b**3)/((1-b)**4)
        return i1 + i2

    cob = 1/tanb
    b = np.arctan(tanb)
    if ali == 0:
        a = b - np.pi/2 # alignment limit
    elif ali == 1:
        a = b - np.arccos(0.05) 
    elif ali == 2:
        a = b - np.arccos(-0.05) 
#    cba,sba = np.sin(2*b),-np.sin(2*b) # wrong sign limit
    cba,sba = np.cos(b-a),np.sin(b-a) # alignment limit

    Vus, Vub = CKM[0,mss[2]], CKM[0,2]
    Vcs, Vcb = CKM[1,mss[2]], CKM[1,2]
    Vts, Vtb = CKM[2,mss[2]], CKM[2,2]
    vev,mW,QCD,s2w = par['vev'],par['m_W'],par['lam_QCD'],par['s2w']
    e = np.sqrt(4*np.pi*get_alpha_e(par,4.18))
    mu = [par['m_u'],par['m_c'],get_mt(par,par['m_t'])]
    muw = [par['m_u'],par['m_c'],get_mt(par,par['m_W'])]
    md = [par[mss[1]],par[mss[0]],par['m_b']]
    ml = [par[mls[1]],par[mls[0]],par['m_tau']]
    y,yh,yH0 = (mW/mH)**2,(mH/par['m_h'])**2,(mH/mH0)**2
    eu = np.array([[cob*mu[0]/vev,0,0],[0,cob*mu[1]/vev,0],[0,0,cob*mu[2]/vev]])
    ed = np.array([[-tanb*md[0]/vev,0,0],[0,-tanb*md[1]/vev,0],[0,0,-tanb*md[2]/vev]])
    el = np.array([[-tanb*ml[0]/vev,0,0],[0,-tanb*ml[1]/vev,0],[0,0,-tanb*ml[2]/vev]])
    zs = [(mu[0]/mH)**2,(mu[1]/mH)**2,(mu[2]/mH)**2]
    zsw = [(mu[0]/mH)**2,(mu[1]/mH)**2,(muw[2]/mH)**2]
    ts = [mu[0]/md[2],mu[1]/md[2],mu[2]/md[2]]
    mul = (4.18/mH)**2
    def Lp(yh,cba,yH0,sba,el):
        return (yh*cba**2 + yH0*sba**2)*(2*el[mls[2],mls[2]])
    def Lm(yh,cba,yH0,sba,el):
        return -1*Lp(yh,cba,yH0,sba,el)
    lam3 = 0.1
    lamh = vev*sba*lam3
    lamH0 = vev*cba*lam3

    def c9_1():
        p1 = 0
        for k in range(3):
            for n in range(3):
                p1 += np.conj(CKM[k,1])*eu[k,2]*eu[n,2]*CKM[n,2]
        c9 = -p1*(1-4*s2w)*(I1(zs[2])-1)/(2*np.conj(Vts)*Vtb*e**2)
        return c9

    def c10_1():
        p1 = 0
        for k in range(3):
            for n in range(3):
                #p1 += np.conj(CKM[k,1])*eu[k,2]*eu[n,2]*CKM[n,2]
                p1 += eu[k,2]*eu[n,2]
        #c10 = p1*(I1(zs[2])-1)/(2*np.conj(Vts)*Vtb*e**2)
        c10 = p1*(I1(zs[2])-1)/(2*e**2)
        return c10

    def c9p_1():
        p1 = 0
        for k in range(3):
            for n in range(3):
                p1 += ed[k,1]*np.conj(CKM[2,k])*CKM[2,n]*ed[n,2]
        c9p = p1*(1-4*s2w)*(I1(zs[2])-1)/(2*np.conj(Vts)*Vtb*e**2)
        return c9p

    def c10p_1():
        p1 = 0
        for k in range(3):
            for n in range(3):
                #p1 += ed[k,1]*np.conj(CKM[2,k])*CKM[2,n]*ed[n,2]
                p1 += ed[k,1]*ed[n,2]
        #c10p = -1*p1*(I1(zs[2])-1)/(2*np.conj(Vts)*Vtb*e**2)
        c10p = -1*p1*(I1(zs[2])-1)/(2*e**2)
        return c10p

    def c9_2():
        p1 = 0 
        for k in range(3):
            for n in range(3):
                p1 += np.conj(CKM[k,1])*eu[k,2]*eu[n,2]*CKM[n,2]
        c9 = p1*y*f5(zs[2])/(27*Vtb*np.conj(Vts)*0.652**2)
        return c9

    def c9p_2():
        p1 = 0 
        for k in range(3):
            for n in range(3):
                p1 += ed[k,1]*np.conj(CKM[2,k])*CKM[2,n]*ed[n,2]
        c9p = p1*y*f5(zs[2])/(27*Vtb*np.conj(Vts)*0.652**2)
        return c9p

    def c9_3():
        p1 = 0 
        for k in range(3):
            for n in range(3):
                p1 += np.conj(CKM[k,1])*eu[k,1]*eu[n,1]*CKM[n,2]
        c9 = 2*p1*y*(19+12*np.log(mul))/(27*Vtb*np.conj(Vts)*0.652**2)
        return c9

    def c9p_3():
        p1 = 0 
        for k in range(3):
            for n in range(3):
                p1 += ed[k,1]*np.conj(CKM[1,k])*CKM[1,n]*ed[n,2]
        c9p = 2*p1*y*(19+12*np.log(mul))/(27*Vtb*np.conj(Vts)*0.652**2)
        return c9p

    def c9_4():
        p1 = 0 
        for k in range(3):
            for n in range(3):
                for i in range(3):
                    for m in range(3):
                        p1 += np.conj(CKM[k,1])*eu[k,i]*eu[n,i]*CKM[n,2]*I1(zs[i])*el[m,mls[2]]**2
        c9 = -y*p1/(s2w*Vtb*np.conj(Vts)*0.652**4)
        return c9

    def c10_2():
        p1 = 0 
        for k in range(3):
            for n in range(3):
                for i in range(3):
                    for m in range(3):
                        p1 += np.conj(CKM[k,1])*eu[k,i]*eu[n,i]*CKM[n,2]*I1(zs[i])*el[m,mls[2]]**2
        c10 = -y*p1/(s2w*Vtb*np.conj(Vts)*0.652**4)
        return c10

    def c9p_4():
        p1 = 0 
        for k in range(3):
            for n in range(3):
                for i in range(3):
                    for m in range(3):
                        p1 += ed[k,1]*np.conj(CKM[i,k])*CKM[i,n]*ed[n,2]*I1(zs[i])*el[m,mls[2]]**2
        c9p = -y*p1/(s2w*Vtb*np.conj(Vts)*0.652**4)
        return c9p

    def c10p_2():
        p1 = 0 
        for k in range(3):
            for n in range(3):
                for i in range(3):
                    for m in range(3):
                        p1 += ed[k,1]*np.conj(CKM[i,k])*CKM[i,n]*ed[n,2]*I1(zs[i])*el[m,mls[2]]**2
        c10p = -y*p1/(s2w*Vtb*np.conj(Vts)*0.652**4)
        return c10p

    def cs_1(Lp,Lm,el):
        sq1, sq2, sq3 = 0,0,0
        for k in range(3):
            for n in range(3):
                sq1 += -(y/2)*Lp*(4*I1(zs[2])*ts[2]*(zs[2]-1)*(ed[2,2]*np.conj(CKM[k,1])*eu[k,2]*CKM[2,2] - ed[2,2]*np.conj(CKM[2,1])*eu[n,2]*CKM[n,2]) - 2*np.log(mul)*(2*(ed[2,2]*np.conj(CKM[k,1])*eu[k,2]*CKM[2,2] - ed[2,2]*np.conj(CKM[2,1])*eu[n,2]*CKM[n,2])*ts[2] + 2*np.conj(CKM[2,1])*eu[2,2]*eu[n,2]*CKM[n,2] - np.conj(CKM[k,1])*eu[k,2]*eu[n,2]*CKM[n,2]) - I0(zs[2])*np.conj(CKM[k,1])*eu[k,2]*eu[n,2]*CKM[n,2] + 4*I5(zs[2],zs[2])*np.conj(CKM[2,1])*eu[2,2]*eu[n,2]*CKM[n,2])
                sq2 += 2*I4(zs[2],zs[2])*np.conj(CKM[2,1])*eu[2,2]*eu[n,2]*CKM[n,2]*Lm*y
                sq3 += np.conj(CKM[2,1])*eu[n,2]*CKM[n,2]*np.sqrt(y*zs[2])*(el[mls[2],mls[2]]*2)*(2*(1-I1(zs[2]))*cba*0.652*sba*(yh-yH0) + I1(zs[2])*np.sqrt(y)*(cba*yh*lamh/mH - sba*yH0*lamH0/mH))
        p1 = (ed[1,1]/(s2w*np.conj(Vts)*Vtb*0.652**4))*(sq1+sq2-sq3)
        return p1

    def csp_1(Lp,Lm,el):
        sq1,sq2,sq3 = 0,0,0
        q = 0
        for k in range(3):
            for n in range(3):
                sq1 += y*Lm*(-2*I1(zs[2])*ts[2]*(zs[2]-1)*((ed[2,2]**2)*np.conj(CKM[k,1])*eu[k,2]*CKM[2,2] - (ed[1,1]**2)*np.conj(CKM[2,1])*eu[n,2]*CKM[n,2]) + 2*np.log(mul)*(-ed[2,2]*np.conj(CKM[k,1])*eu[k,2]*eu[2,2]*CKM[2,2] + ((ed[2,2]**2)*np.conj(CKM[k,1])*eu[k,2]*CKM[2,2] - (ed[1,1]**2)*np.conj(CKM[2,1])*eu[n,2]*CKM[n,2])*ts[2]) + ed[2,2]*(I7(zs[2])*(ed[1,1]**2)*np.conj(CKM[2,1])*Vtb + 2*I5(zs[2],zs[2])*np.conj(CKM[k,1])*eu[k,2]*eu[2,2]*Vtb))
                sq2 += -2*I4(zs[2],zs[2])*ed[2,2]*np.conj(CKM[k,1])*eu[k,2]*eu[2,2]*Vtb*Lp*y 
                sq3 += ed[2,2]*np.conj(CKM[k,1])*eu[k,2]*Vtb*np.sqrt(y*zs[2])*(el[mls[2],mls[2]]*2)*(2*(1-I1(zs[2]))*cba*0.652*sba*(yh-yH0) + I1(zs[2])*np.sqrt(y)*(cba*yh*lamh/mH - sba*yH0*lamH0/mH))
        p1 = (1/(s2w*np.conj(Vts)*Vtb*0.652**4))*(sq1+sq2-sq3)
        return p1

    def cs_2(Lp,Lm,el):
        p1 = (ed[1,1]/(s2w*0.652**2))*(zsw[2]*np.log(mul)*Lp/4 + I3(y,zsw[2])*Lp/8 + I2(zsw[2])*el[mls[2],mls[2]]) 
        return p1

    def csp_2(Lp,Lm,el):
        p1 = (ed[2,2]/(s2w*0.652**2))*(zsw[2]*np.log(mul)*Lm/2 - I6(zsw[2])*Lm/2 + I2(zsw[2])*el[mls[2],mls[2]]) 
        return p1

    def cs_3(Lp,Lm,el):
        sq1, sq2, sq3 = 0,0,0
        for k in range(3):
            for n in range(3):
                sq1 += 4*ts[1]*(ed[2,2]*np.conj(CKM[k,1])*eu[k,1]*CKM[1,2] - ed[2,2]*np.conj(CKM[1,1])*eu[n,1]*CKM[n,2]) + np.conj(CKM[k,1])*eu[k,1]*eu[n,1]*CKM[n,2]
                sq2 += 2*np.log(mul)*(2*(ed[2,2]*np.conj(CKM[k,1])*eu[k,1]*CKM[1,2] - ed[2,2]*np.conj(CKM[1,1])*eu[n,1]*CKM[n,2])*ts[1] + CKM[n,2]*(2*np.conj(CKM[1,1])*eu[n,1]*eu[1,1] + 2*np.conj(CKM[1,1])*eu[n,2]*eu[1,2] + 2*np.conj(CKM[2,1])*eu[n,1]*eu[2,1] - np.conj(CKM[k,1])*eu[n,1]*eu[k,1]))
                sq3 += 4*(np.conj(CKM[1,1])*eu[1,1]*eu[n,1]*CKM[n,2] - I5(zs[2],0)*(np.conj(CKM[1,1])*eu[1,2]*eu[n,2]*CKM[n,2] + np.conj(CKM[2,1])*eu[2,1]*eu[n,1]*CKM[n,2]))
        p1 = -y*ed[1,1]*Lp*(sq1-sq2-sq3)/(2*s2w*np.conj(Vts)*Vtb*0.652**4)
        return p1

    def csp_3(Lp,Lm,el):
        sq1, sq2, sq3 = 0,0,0
        for k in range(3):
            for n in range(3):
                sq1 += -2*ts[1]*((ed[2,2]**2)*np.conj(CKM[k,1])*eu[k,1]*CKM[1,2] - (ed[1,1]**2)*np.conj(CKM[1,1])*eu[n,1]*CKM[n,2])
                sq2 += 2*np.log(mul)*(-ed[2,2]*np.conj(CKM[k,1])*eu[k,1]*eu[1,1]*CKM[1,2] - ed[2,2]*np.conj(CKM[k,1])*eu[k,1]*eu[2,1]*CKM[2,2] - ed[2,2]*np.conj(CKM[k,1])*eu[k,2]*eu[1,2]*CKM[1,2] + ((ed[2,2]**2)*np.conj(CKM[k,1])*eu[k,1]*CKM[1,2] - (ed[1,1]**2)*np.conj(CKM[1,1])*eu[n,1]*CKM[n,2])*ts[1])
                sq3 += -2*ed[2,2]*np.conj(CKM[k,1])*eu[k,1]*eu[1,1]*CKM[1,2] + ed[2,2]*(-(ed[1,1]**2)*np.conj(CKM[1,1])*CKM[1,2] + 2*I5(zs[2],0)*np.conj(CKM[k,1])*(eu[k,2]*eu[1,2]*CKM[1,2] + eu[k,1]*eu[2,1]*CKM[2,2]))
        p1 = y*Lm*(sq1+sq2+sq3)/(s2w*np.conj(Vts)*Vtb*0.652**4)
        return p1

    C9 = c9_1() + c9_2() + c9_3() + c9_4()
    C9p = c9p_1() + c9p_2() + c9p_3() + c9p_4()
    C10 = c10_1() + c10_2()
    C10p = c10p_1() + c10p_2()
    CS = cs_1(Lp(yh,cba,yH0,sba,el),Lm(yh,cba,yH0,sba,el),el) + cs_2(Lp(yh,cba,yH0,sba,el),Lm(yh,cba,yH0,sba,el),el) + cs_3(Lp(yh,cba,yH0,sba,el),Lm(yh,cba,yH0,sba,el),el)
    CSP = csp_1(Lp(yh,cba,yH0,sba,el),Lm(yh,cba,yH0,sba,el),el) + csp_2(Lp(yh,cba,yH0,sba,el),Lm(yh,cba,yH0,sba,el),el) + csp_3(Lp(yh,cba,yH0,sba,el),Lm(yh,cba,yH0,sba,el),el)
    CP = cs_1(Lp(yh,cba,yH0,sba,-1*el),Lm(yh,cba,yH0,sba,-1*el),-1*el) + cs_2(Lp(yh,cba,yH0,sba,-1*el),Lm(yh,cba,yH0,sba,-1*el),-1*el) + cs_3(Lp(yh,cba,yH0,sba,-1*el),Lm(yh,cba,yH0,sba,-1*el),-1*el)
    CPP = csp_1(Lp(yh,cba,yH0,sba,-1*el),Lm(yh,cba,yH0,sba,-1*el),-1*el) + csp_2(Lp(yh,cba,yH0,sba,-1*el),Lm(yh,cba,yH0,sba,-1*el),-1*el) + csp_3(Lp(yh,cba,yH0,sba,-1*el),Lm(yh,cba,yH0,sba,-1*el),-1*el)

#    if mls[2] == 1:
#        print("CS HH:",cs_1(Lp(yh,cba,yH0,sba,el),Lm(yh,cba,yH0,sba,el),el) + cs_3(Lp(yh,cba,yH0,sba,el),Lm(yh,cba,yH0,sba,el),el))
#        print()
#        print("CS HW:",cs_2(Lp(yh,cba,yH0,sba,el),Lm(yh,cba,yH0,sba,el),el))
#        print()
#        print("CS prime HH:",csp_1(Lp(yh,cba,yH0,sba,el),Lm(yh,cba,yH0,sba,el),el) + csp_3(Lp(yh,cba,yH0,sba,el),Lm(yh,cba,yH0,sba,el),el))
#        print()
#        print("CS prime HW:",csp_2(Lp(yh,cba,yH0,sba,el),Lm(yh,cba,yH0,sba,el),el))
#        print()
#        print("CS:",CS)
#        print()
#        print("CS prime:",CSP)
#        print()
#        print("CP:",CP)
#        print()
#        print("CP prime:",CPP)
#        print()

    return C9, C9p, C10, C10p, CS, CSP, CP, CPP

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
        if a != b:
            i = -1/((1-a)*(1-b)) + (np.log(b)*b**2)/((a-b)*(1-b)**2) + (np.log(a)*a**2)/((b-a)*(1-a)**2)
        else:
            i = (1-(a**2)+2*a*np.log(a))/((a-1)**3)
        return i
    def I9(a,b):
        if a != b:
            i = -a*b/((1-a)*(1-b)) + a*b*np.log(b)/((a-b)*(1-b)**2) + a*b*np.log(a)/((b-a)*(1-a)**2)
        else:
            i = -a*(a**2 -1 - 2*a*np.log(a))/((a-1)**3)
        return i
    def I10(a,b):
        if a != b:
            i = -1/((1-a)*(1-b)) + a*np.log(a)/((b-a)*(1-a)**2) + b*np.log(b)/((a-b)*(1-b)**2)
        else:
            i = (2-2*a+(1+a)*np.log(a))/((a-1)**3)
        return i
    def I11(a,b,c):
        if b != c:
            i = -3*(a**2)*np.log(a)/((a-1)*(a-b)*(a-c)) + b*(4*a-b)*np.log(b)/((b-1)*(a-b)*(b-c)) + c*(4*a-c)*np.log(c)/((c-1)*(a-c)*(c-b))
        else:
            i = -3*(a**2)*np.log(a)/(a-1) + ((c-1)*(4*a**2 - 5*a*c + c**2)-(4*a**2 + c**2 - a*c*(2+3*c))*np.log(c))/((c-1)**2)
            i = i/((a-c)**2)
        return i
    def I12(a,b):
        if a != b:
            i = a*b*np.log(a)/((1-a)*(a-b)) - a*b*np.log(b)/((1-b)*(a-b)) 
        else:
            i = a*(1-a+a*np.log(a))/((a-1)**2)
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
        pref = (0.652**2)/(64*(np.pi*mW)**2)
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
        pref = -(ed[1]*ed[2]*0.652**2)/(16*(np.pi*mW)**2)
        ai, bi, ci = 0,0,0
        for i in range(3):
            ai += v2[i]*v3[i]*I12(zs[0],zs[i])
            bi += v2[i]*v3[i]*I12(zs[1],zs[i])
            ci += v2[i]*v3[i]*I12(zs[2],zs[i])
        a = v2[0]*v3[0]*ai
        b = v2[1]*v3[1]*bi
        c = v2[2]*v3[2]*ci
        return pref*(a+b+c) 

    CVLL = c1_1() + c1_2()
    CVRR = c1p()
    CSLL = c2()
    CSRR = c2p()
    CSLR = c4_1() + c4_2()
    CVLR = 2*c5()

#    print("CVLL:",CVLL)
#    print("CVRR:",CVRR)
#    print("CSLL:",CSLL)
#    print("CSRR:",CSRR)
#    print("CSLR:",CSLR)
#    print("CVLR:",CVLR)

    return CVLL, CVRR, CSLL, CSRR, CSLR, CVLR

def rh(mu,md,ml,tanb,mH):
    '''
        Function for M->lnu 2HDM WC contribution
    '''
    csl = -1*mu*ml/(mH**2)
    csr = -1*md*ml*(tanb/mH)**2
    return csr, csl

def chi2_func(tanb, mH, mH0, mA0, obs):
    '''
        Finding chisq value cause there's some extra factor going on in flavio's
    '''

    par = flavio.default_parameters.get_central_all()
    ckm_els = flavio.physics.ckm.get_ckm(par) # get out all the CKM elements

#    csev = a_mu2(par,'m_mu',tanb,mH0,mA0,mH)
    CSR_b_t, CSL_b_t = rh(par['m_u'],par['m_b'],par['m_tau'],tanb,mH)
    CSR_b_m, CSL_b_m = rh(par['m_u'],par['m_b'],par['m_mu'],tanb,mH)
    CSR_b_e, CSL_b_e = rh(par['m_u'],par['m_b'],par['m_e'],tanb,mH)
    CSR_d_t, CSL_d_t = rh(par['m_c'],par['m_d'],par['m_tau'],tanb,mH)
    CSR_d_m, CSL_d_m = rh(par['m_c'],par['m_d'],par['m_mu'],tanb,mH)
    CSR_d_e, CSL_d_e = rh(par['m_c'],par['m_d'],par['m_e'],tanb,mH)
    CSR_ds_t, CSL_ds_t = rh(par['m_c'],par['m_s'],par['m_tau'],tanb,mH)
    CSR_ds_m, CSL_ds_m = rh(par['m_c'],par['m_s'],par['m_mu'],tanb,mH)
    CSR_ds_e, CSL_ds_e = rh(par['m_c'],par['m_s'],par['m_e'],tanb,mH)
    CSR_k_t, CSL_k_t = rh(par['m_u'],par['m_s'],par['m_tau'],tanb,mH)
    CSR_k_m, CSL_k_m = rh(par['m_u'],par['m_s'],par['m_mu'],tanb,mH)
    CSR_k_e, CSL_k_e = rh(par['m_u'],par['m_s'],par['m_e'],tanb,mH)
    CSR_p_t, CSL_p_t = rh(par['m_u'],par['m_d'],par['m_tau'],tanb,mH)
    CSR_p_m, CSL_p_m = rh(par['m_u'],par['m_d'],par['m_mu'],tanb,mH)
    CSR_bc_t, CSL_bc_t = rh(par['m_c'],par['m_b'],par['m_tau'],tanb,mH)
    CSR_bc_m, CSL_bc_m = rh(par['m_c'],par['m_b'],par['m_mu'],tanb,mH)
    CSR_bc_e, CSL_bc_e = rh(par['m_c'],par['m_b'],par['m_e'],tanb,mH)
    C7, C7p, C8, C8p = bsgamma2(par,ckm_els,flavio.config['renormalization scale']['bxgamma'],tanb,mH)
    C9_se, C9p_se, C10_se, C10p_se, CS_se, CSp_se, CP_se, CPp_se = bsll(par,ckm_els,['m_s','m_d',1],['m_e','m_mu',1],mH0,tanb,mH)
    C9_s, C9p_s, C10_s, C10p_s, CS_s, CSp_s, CP_s, CPp_s = bsll(par,ckm_els,['m_s','m_d',1],['m_mu','m_e',1],mH0,tanb,mH,0)
    C9_d, C9p_d, C10_d, C10p_d, CS_d, CSp_d, CP_d, CPp_d = bsll(par,ckm_els,['m_d','m_s',0],['m_mu','m_e',1],mH0,tanb,mH,0)
    CVLL_bs, CVRR_bs, CSLL_bs, CSRR_bs, CSLR_bs, CVLR_bs = mixing(par,ckm_els,['m_s',1,'m_d'],tanb,mH)
    CVLL_bd, CVRR_bd, CSLL_bd, CSRR_bd, CSLR_bd, CVLR_bd = mixing(par,ckm_els,['m_d',0,'m_s'],tanb,mH)

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
    chisq = 0
    for i in obs:
        if type(i) == str:
            npp = flavio.np_prediction(i,wc_obj=wc) 
            npe = flavio.np_uncertainty(i,wc_obj=wc) 
        elif type(i) == tuple:
            ob, q1, q2 = i
            npp = flavio.np_prediction(ob,wc_obj=wc,q2min=q1,q2max=q2)
            npe = flavio.np_uncertainty(ob,wc_obj=wc,q2min=q1,q2max=q2)
        exp = flavio.combine_measurements(i,include_measurements=['Tree Level Leptonics','Radiative Decays','FCNC Leptonic Decays','B Mixing','LFU D Ratios','Tree Level Semileptonics','LFU K Ratios','Anomalous Magnetic Moments']+ims)
        expc = exp.central_value
        expr = exp.error_right
        expl = exp.error_left
        expp = ((expc+expr)+(expc-expl))/2
        expe = (expc+expr)-expp
        sig = np.sqrt(npe**2 + expe**2)
        chisq += ((npp-expp)/sig)**2

    return chisq

def pval_func(cdat,app,obs,sigmas):
    bf,minh,mint,maxt,minz = mHmin(cdat)#,minz_allsim)

    if app == 0:
        mHp,mH0,mA0 = bf[1],bf[1],bf[1]
        print('mH+ = mH0 = mA0')
    elif app == 2:
        mHp,mH0,mA0 = bf[1],bf[1],1500
        print('mH+ = mH0, mA0 = 1500 GeV')
    elif app == 1:
        mHp,mH0,mA0 = bf[1],1500,bf[1]
        print('mH+ = mA0, mH0 = 1500 GeV')

    print("Best fit value is found for (tanb,mH) =", bf)
    print("Print outs are lists for values at", sigmas, "sigmas")
    print("Minimum value of mH+ is:", minh) 
    print("Minimum value of tanb is:", mint)
    print("Maximum value of tanb is:", maxt)
    print(minz)

    chi2 = chi2_func(bf[0],mHp,mH0,mA0,obs)
    degs = len(obs)-2
    pval = pvalue(chi2,degs)
    print("chi2tilde_min is:",minz) 
    print("chi2_min is:",chi2)
    print("chi2_nu is:",chi2/degs)  
    print("2log(L(theta)) = ",chi2-minz)
    print("p-value at chi2_min point with dof =",degs," is",pval*100,"%")
    return None

def a_mu(par,ml,tanb,mH0,mA0,mH):
    '''
        Anomalous magnetic moment of the muon
    '''
    emu = -tanb*par[ml]/par['vev']
#    e = np.sqrt(4*np.pi*get_alpha_e(par,1.0))
    e = np.sqrt(4*np.pi*get_alpha_e(par,4.2))
    b = np.arctan(tanb)
    a = b - np.pi/2
#    a = b - np.arccos(0.05)
    cba, sba = np.cos(b-a),np.sin(b-a)
    
    def cr_1():
        cr = e*(par[ml]**3)*(tanb**2)/(192*(np.pi*par['vev']*mH)**2)
        return cr

    def cr_2():
        def gamHh():
            gam = cba*par[ml]/par['vev'] - sba*emu
            return gam
        def gamh0():
            gam = sba*par[ml]/par['vev'] + cba*emu
            return gam
        def gamA0():
            gam = 1j*emu
            return gam
        def eqn(gam,meh):
            eq = -2*e*par[ml]*(abs(gam())**2)/(192*(np.pi*meh)**2) 
            eq += e*par[ml]*(gam()**2)*(3+2*np.log((par[ml]/meh)**2))/(64*(np.pi*meh)**2)
            return eq

        cr = eqn(gamHh,mH0) + eqn(gamA0,mA0) #+ eqn(gamh0,par['m_h']) 

        return cr

    pref = e*par[ml]*par['GF']/(4*np.sqrt(2)*np.pi**2)
    csev = (cr_1()+cr_2())/pref
    return csev

def a_mu2(par,lep,tanb,mH0,mA0,mH):
    #----------------------------------------------
    def fH(r):
        def integ(x):
            z = (x**2)*(2-x)/(1-x+r*x**2)
            return z
        inte, err = quad(integ,0,1)
        return inte
    def fA(r):
        def integ(x):
            z = -x**3/(1-x+r*x**2)
            return z
        inte, err = quad(integ,0,1)
        return inte
    def fHp(r):
        def integ(x):
            z = -x*(1-x)/(1-(1-x)*r)
            return z
        inte, err = quad(integ,0,1)
        return inte

    GF, mW, s2w, vev = par['GF'],par['m_W'],par['s2w'],par['vev']
    b = np.arctan(tanb)
    a = b - np.pi/2
    ca, cb, sa, sb = np.cos(a), np.cos(b), np.sin(a), np.sin(b)
    sba, cba = np.sin(b-a), np.cos(b-a)

    mmu = par[lep]
    rH, rA, rHp = (mmu/mH0)**2, (mmu/mA0)**2, (mmu/mH)**2
    yHl, yAl = (cba + sba*tanb), -1*tanb
    yhsl = sba - cba*tanb

    def int1():
        def fu(x):
            z = (2-x)*(x**2)/(rH*x**2 - x + 1)
            return z
        integ,err = quad(fu,0,1)
        return integ
    def int2():
        def fu(x):
            z = -(x**3)/(rA*x**2 - x + 1)
            return z
        integ,err = quad(fu,0,1)
        return integ
    def int3():
        def fu(x):
            z = (1-x)*(x**2)/(rHp*x*(1-x)-x)
            return z
        integ,err = quad(fu,0,1)
        return integ

    da = (rH*(yHl**2)*int1() + rA*(yAl**2)*int2() + rHp*(yAl**2)*int3())*(mmu**2)/(8*pow(np.pi*vev,2))

    aem = get_alpha_e(par,4.2)
    yHu, yHd, yAu, yAd = (cba - sba/tanb), yHl, -1/tanb, -1*tanb
    mu, md, mc, ms, mb, mt = par['m_u'], par['m_d'], par['m_c'], par['m_s'], par['m_b'], par['m_t']
    Qu, Qd = (2/3)**2, (-1/3)**2
    #I don't know why I'm not just summing over lists/arrays, but oh well
    rHu, rHd, rHc, rHs, rHb, rHt = (mu/mH0)**2, (md/mH0)**2, (mc/mH0)**2, (ms/mH0)**2, (mb/mH0)**2, (mt/mH0)**2
    rAu, rAd, rAc, rAs, rAb, rAt = (mu/mA0)**2, (md/mA0)**2, (mc/mA0)**2, (ms/mA0)**2, (mb/mA0)**2, (mt/mA0)**2
    
    def F1(r):
        def integ(x):
            z = (2*x*(1-x)-1)*np.log(r/(x*(1-x)))/(r - x*(1-x))
            return z
        inte,err = quad(integ,0,1)
        return inte*r/2
    def Ft1(r):
        def integ(x):
            z = np.log(r/(x*(1-x)))/(r - x*(1-x))
            return z
        inte,err = quad(integ,0,1)
        return inte*r/2
    
    pref = aem*(mmu**2)*3/(4*(vev**2)*np.pi**3)
    d1 = yHu*yHl*Qu*F1(rHu) + yAu*yAl*Qu*Ft1(rAu)
    d1 += yHu*yHl*Qu*F1(rHc) + yAu*yAl*Qu*Ft1(rAc)
    d1 += yHu*yHl*Qu*F1(rHt) + yAu*yAl*Qu*Ft1(rAt)
    d1 += yHd*yHl*Qd*F1(rHd) + yAd*yAl*Qd*Ft1(rAd)
    d1 += yHd*yHl*Qd*F1(rHs) + yAd*yAl*Qd*Ft1(rAs)
    d1 += yHd*yHl*Qd*F1(rHb) + yAd*yAl*Qd*Ft1(rAb)
    d1 = d1*pref

    #----------------------------------------------
    RH1, RH2 = cba, -1*sba
    rWH0 = (mW/mH0)**2
    mhs = par['m_h']
    rWHp, rHHp, rSMHp, rHW, rSMW = (mW/mH)**2, (mH0/mH)**2, (mhs/mH)**2, (mH0/mW)**2, (mhs/mW)**2
    lam3 = (mhs**2 + 2*mH**2 - 2*mH0**2)/vev**2
    lamhs = vev*lam3

    def F2(r):
        def integ(x):
            z = x*(x-1)*np.log(r/(x*(1-x)))/(r - x*(1-x))
            return z
        inte, err = quad(integ,0,1)
        return inte/2

    d2 = aem*(mmu**2)*yhsl*lamhs*F2(1/rSMHp)/(8*(mhs**2)*np.pi**3)
    
#    def F3(r):
#        def integ(x):
#            z = (x*(3*x*(4*x-1)+10)*r - x*(1-x))*np.log(r/(x*(1-x)))/(r-x*(1-x))
#            return z
#        inte, err = quad(integ,0,1)
#        return inte/2
#
#    d3 = aem*(mmu**2)*yHl*RH1*F3(rWH0)/(8*(vev**2)*np.pi**3)

    #----------------------------------------------
    ckms = flavio.physics.ckm.get_ckm(par)
    Vtb2 = abs(ckms[2,2])**2
    rtHp, rbHp, rtW, rbW = (mt/mH)**2, (mb/mH)**2, (mt/mW)**2, (mb/mW)**2
    
    def G(r1,r2,x):
        z = np.log((x*r1 + (1-x)*r2)/(x*(1-x)))/(x*(1-x) - x*r1 - (1-x)*r2)
        return z
    def I4():
        def integ(x):
            z = (2*x/3 - (1-x)/3)*(pow(tanb*mb,2)*x*(1-x) - (mt**2)*x*(1+x))*(G(rtHp,rbHp,x)-G(rtW,rbW,x))
            return z
        inte, err = quad(integ,0,1)
        return inte

    d4 = aem*(mmu**2)*3*Vtb2*I4()/(32*(mH**2 - mW**2)*s2w*pow(vev,2)*np.pi**3)
    
    #----------------------------------------------
    RSM1, RSM2 = sba, cba

#    def I5(m):
#        def integ(x):
#            z = (x**2)*((mH**2 + mW**2 - m)*(1-x) - 4*mW**2)*(G(rWHp,m/(mH**2),x)-G(1,m/(mW**2),x))
#            return z
#        inte, err = quad(integ,0,1)
#        return inte
#
#    d5 = -tanb*RSM1*RSM2*I5(mhs**2) - tanb*RH1*RH2*I5(mH0**2)
#    d5 = d5*aem*(mmu**2)/(64*(mH**2 - mW**2)*pow(vev*s2w,2)*np.pi**3)

    #----------------------------------------------
#
#    def I6(m):
#        def integ(x):
#            z = (x**2)*(x-1)*(G(1,m/(mH**2),x)-G(1/rWHp,m/(mW**2),x))
#            return z
#        inte, err = quad(integ,0,1)
#        return inte
#
#    d6 = -tanb*RSM2*lamhs*I6(mhs**2)
#    d6 = d6*aem*(mmu**2)/(64*(mH**2 - mW**2)*(vev**2)*np.pi**3)

    #----------------------------------------------
    cr = (da + d1 + d2 + d4)*np.sqrt(2)*(np.pi**2)/(GF*mmu**2)

    return cr

