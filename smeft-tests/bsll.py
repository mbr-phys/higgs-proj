#!/bin/env python3

import flavio
from flavio.classes import Parameter, Observable, Measurement
from flavio.physics.running.running import get_mt, get_alpha_e
from flavio.statistics.functions import pvalue
from flavio.statistics.likelihood import *
from flavio.statistics.probability import NormalDistribution, MultivariateNormalDistribution
from scipy.integrate import quad
import matplotlib.pyplot as plt
import numpy as np

def bsll(par,CKM,d,q,l,mH0,tanb,mH,cba):
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
    sba = np.sqrt(1-cba**2)

    vev,mW,QCD,s2w = par['vev'],par['m_W'],par['lam_QCD'],par['s2w']
    e = np.sqrt(4*np.pi*get_alpha_e(par,4.2))
    mu = [par['m_u'],par['m_c'],get_mt(par,par['m_t'])]
    muw = [par['m_u'],par['m_c'],get_mt(par,par['m_W'])]
    md = [par['m_d'],par['m_s'],par['m_b']]
    ml = [par['m_e'],par['m_mu'],par['m_tau']]
    y,yh,yH0 = (mW/mH)**2,(mH/par['m_h'])**2,(mH/mH0)**2
    eu = np.array([[cob*mu[0]/vev,0,0],[0,cob*mu[1]/vev,0],[0,0,cob*mu[2]/vev]])
    ed = np.array([[-tanb*md[0]/vev,0,0],[0,-tanb*md[1]/vev,0],[0,0,-tanb*md[2]/vev]])
    el = np.array([[-tanb*ml[0]/vev,0,0],[0,-tanb*ml[1]/vev,0],[0,0,-tanb*ml[2]/vev]])
    zs = [(mu[0]/mH)**2,(mu[1]/mH)**2,(mu[2]/mH)**2]
    zsw = [(mu[0]/mH)**2,(mu[1]/mH)**2,(muw[2]/mH)**2]
    ts = [mu[0]/md[d],mu[1]/md[d],mu[2]/md[d]]
    mul = (4.2/mH)**2
    def Lp(yh,cba,yH0,sba,el):
        return (yh*cba**2 + yH0*sba**2)*(2*el[l,l])
    def Lm(yh,cba,yH0,sba,el):
        return -1*Lp(yh,cba,yH0,sba,el)
    lam3 = 0.1
    lamh = vev*sba*lam3
    lamH0 = vev*cba*lam3

    def c9_1():
        p1 = 0
        for k in range(3):
            for n in range(3):
                p1 += np.conj(CKM[k,q])*eu[k,2]*eu[n,2]*CKM[n,d]
        c9 = -p1*(1-4*s2w)*(I1(zs[2])-1)/(2*np.conj(CKM[2,q])*CKM[2,d]*e**2)
        return c9

    def c10_1():
        p1 = 0
        for k in range(3):
            for n in range(3):
                p1 += np.conj(CKM[k,q])*eu[k,2]*eu[n,2]*CKM[n,d]
                #p1 += eu[k,2]*eu[n,2]
        c10 = p1*(I1(zs[2])-1)/(2*np.conj(CKM[2,q])*CKM[2,d]*e**2)
        #c10 = p1*(I1(zs[2])-1)/(2*e**2)
        return c10

    def c9p_1():
        p1 = 0
        for k in range(3):
            for n in range(3):
                p1 += ed[k,1]*np.conj(CKM[2,k])*CKM[2,n]*ed[n,2]
        c9p = p1*(1-4*s2w)*(I1(zs[2])-1)/(2*np.conj(CKM[2,q])*CKM[2,d]*e**2)
        return c9p

    def c10p_1():
        p1 = 0
        for k in range(3):
            for n in range(3):
                p1 += ed[k,1]*np.conj(CKM[2,k])*CKM[2,n]*ed[n,d]
                #p1 += ed[k,1]*ed[n,2]
        c10p = -1*p1*(I1(zs[2])-1)/(2*np.conj(CKM[2,q])*CKM[2,d]*e**2)
        #c10p = -1*p1*(I1(zs[2])-1)/(2*e**2)
        return c10p

    def c9_2():
        p1 = 0 
        for k in range(3):
            for n in range(3):
                p1 += np.conj(CKM[k,q])*eu[k,2]*eu[n,2]*CKM[n,d]
        c9 = p1*y*f5(zs[2])/(27*CKM[2,d]*np.conj(CKM[2,q])*0.652**2)
        return c9

    def c9p_2():
        p1 = 0 
        for k in range(3):
            for n in range(3):
                p1 += ed[k,1]*np.conj(CKM[2,k])*CKM[2,n]*ed[n,2]
        c9p = p1*y*f5(zs[2])/(27*CKM[2,d]*np.conj(CKM[2,q])*0.652**2)
        return c9p

    def c9_3():
        p1 = 0 
        for k in range(3):
            for n in range(3):
                p1 += np.conj(CKM[k,q])*eu[k,1]*eu[n,1]*CKM[n,d]
        c9 = 2*p1*y*(19+12*np.log(mul))/(27*CKM[2,d]*np.conj(CKM[2,q])*0.652**2)
        return c9

    def c9p_3():
        p1 = 0 
        for k in range(3):
            for n in range(3):
                p1 += ed[k,1]*np.conj(CKM[1,k])*CKM[1,n]*ed[n,2]
        c9p = 2*p1*y*(19+12*np.log(mul))/(27*CKM[2,d]*np.conj(CKM[2,q])*0.652**2)
        return c9p

    def c9_4():
        p1 = 0 
        for k in range(3):
            for n in range(3):
                for i in range(3):
                    for m in range(3):
                        p1 += np.conj(CKM[k,q])*eu[k,i]*eu[n,i]*CKM[n,d]*I1(zs[i])*el[m,l]**2
        c9 = -y*p1/(s2w*CKM[2,d]*np.conj(CKM[2,q])*0.652**4)
        return c9

    def c10_2():
        p1 = 0 
        for k in range(3):
            for n in range(3):
                for i in range(3):
                    for m in range(3):
                        p1 += np.conj(CKM[k,q])*eu[k,i]*eu[n,i]*CKM[n,d]*I1(zs[i])*el[m,l]**2
        c10 = -y*p1/(s2w*CKM[2,d]*np.conj(CKM[2,q])*0.652**4)
        return c10

    def c9p_4():
        p1 = 0 
        for k in range(3):
            for n in range(3):
                for i in range(3):
                    for m in range(3):
                        p1 += ed[k,q]*np.conj(CKM[i,k])*CKM[i,n]*ed[n,d]*I1(zs[i])*el[m,l]**2
        c9p = -y*p1/(s2w*CKM[2,d]*np.conj(CKM[2,q])*0.652**4)
        return c9p

    def c10p_2():
        p1 = 0 
        for k in range(3):
            for n in range(3):
                for i in range(3):
                    for m in range(3):
                        p1 += ed[k,q]*np.conj(CKM[i,k])*CKM[i,n]*ed[n,d]*I1(zs[i])*el[m,l]**2
        c10p = -y*p1/(s2w*CKM[2,d]*np.conj(CKM[2,q])*0.652**4)
        return c10p

    def cs_1(Lp,Lm,el):
        sq1, sq2, sq3 = 0,0,0
        for k in range(3):
            for n in range(3):
                sq1 += -(y/2)*Lp*(4*I1(zs[2])*ts[2]*(zs[2]-1)*(ed[d,d]*np.conj(CKM[k,q])*eu[k,2]*CKM[2,d] - ed[d,d]*np.conj(CKM[2,q])*eu[n,2]*CKM[n,d]) - 2*np.log(mul)*(2*(ed[d,d]*np.conj(CKM[k,q])*eu[k,2]*CKM[2,d] - ed[d,d]*np.conj(CKM[2,q])*eu[n,2]*CKM[n,d])*ts[2] + 2*np.conj(CKM[2,q])*eu[2,2]*eu[n,2]*CKM[n,d] - np.conj(CKM[k,q])*eu[k,2]*eu[n,2]*CKM[n,d]) - I0(zs[2])*np.conj(CKM[k,q])*eu[k,2]*eu[n,2]*CKM[n,d] + 4*I5(zs[2],zs[2])*np.conj(CKM[2,q])*eu[2,2]*eu[n,2]*CKM[n,d])
                sq2 += 2*I4(zs[2],zs[2])*np.conj(CKM[2,q])*eu[2,2]*eu[n,2]*CKM[n,d]*Lm*y
                sq3 += np.conj(CKM[2,q])*eu[n,2]*CKM[n,d]*np.sqrt(y*zs[2])*(el[l,l]*2)*(2*(1-I1(zs[2]))*cba*0.652*sba*(yh-yH0) + I1(zs[2])*np.sqrt(y)*(cba*yh*lamh/mH - sba*yH0*lamH0/mH))
        p1 = (ed[q,q]/(s2w*np.conj(CKM[2,q])*CKM[2,d]*0.652**4))*(sq1+sq2-sq3)
        return p1

    def csp_1(Lp,Lm,el):
        sq1,sq2,sq3 = 0,0,0
        q = 0
        for k in range(3):
            for n in range(3):
                sq1 += y*Lm*(-2*I1(zs[2])*ts[2]*(zs[2]-1)*((ed[d,d]**2)*np.conj(CKM[k,q])*eu[k,2]*CKM[2,d] - (ed[q,q]**2)*np.conj(CKM[2,q])*eu[n,2]*CKM[n,d]) + 2*np.log(mul)*(-ed[d,d]*np.conj(CKM[k,q])*eu[k,2]*eu[2,2]*CKM[2,d] + ((ed[d,d]**2)*np.conj(CKM[k,q])*eu[k,2]*CKM[2,d] - (ed[q,q]**2)*np.conj(CKM[2,q])*eu[n,2]*CKM[n,d])*ts[2]) + ed[d,d]*(I7(zs[2])*(ed[q,q]**2)*np.conj(CKM[2,q])*CKM[2,d] + 2*I5(zs[2],zs[2])*np.conj(CKM[k,q])*eu[k,2]*eu[2,2]*CKM[2,d]))
                sq2 += -2*I4(zs[2],zs[2])*ed[d,d]*np.conj(CKM[k,q])*eu[k,2]*eu[2,2]*CKM[2,d]*Lp*y 
                sq3 += ed[d,d]*np.conj(CKM[k,q])*eu[k,2]*CKM[2,d]*np.sqrt(y*zs[2])*(el[l,l]*2)*(2*(1-I1(zs[2]))*cba*0.652*sba*(yh-yH0) + I1(zs[2])*np.sqrt(y)*(cba*yh*lamh/mH - sba*yH0*lamH0/mH))
#        print('Lines 1-3: ',sq1)
#        print('Line 4: ',sq2)
#        print('Lines 5-6: ',sq3)
        p1 = (1/(s2w*np.conj(CKM[2,q])*CKM[2,d]*0.652**4))*(sq1+sq2-sq3)
        return p1

    def cs_2(Lp,Lm,el):
        p1 = (ed[q,q]/(s2w*0.652**2))*(zsw[2]*np.log(mul)*Lp/4 + I3(y,zsw[2])*Lp/8 + I2(zsw[2])*el[l,l]) 
        return p1

    def csp_2(Lp,Lm,el):
        p1 = (ed[d,d]/(s2w*0.652**2))*(zsw[2]*np.log(mul)*Lm/2 - I6(zsw[2])*Lm/2 + I2(zsw[2])*el[l,l]) 
        return p1

    def cs_3(Lp,Lm,el):
        sq1, sq2, sq3 = 0,0,0
        for k in range(3):
            for n in range(3):
                sq1 += 4*ts[1]*(ed[d,d]*np.conj(CKM[k,q])*eu[k,1]*CKM[1,d] - ed[d,d]*np.conj(CKM[1,q])*eu[n,1]*CKM[n,d]) + np.conj(CKM[k,q])*eu[k,1]*eu[n,1]*CKM[n,d]
                sq2 += 2*np.log(mul)*(2*(ed[d,d]*np.conj(CKM[k,q])*eu[k,1]*CKM[1,d] - ed[d,d]*np.conj(CKM[1,q])*eu[n,1]*CKM[n,d])*ts[1] + CKM[n,d]*(2*np.conj(CKM[1,q])*eu[n,1]*eu[1,1] + 2*np.conj(CKM[1,q])*eu[n,2]*eu[1,2] + 2*np.conj(CKM[2,q])*eu[n,1]*eu[2,1] - np.conj(CKM[k,q])*eu[n,1]*eu[k,1]))
                sq3 += 4*(np.conj(CKM[1,q])*eu[1,1]*eu[n,1]*CKM[n,d] - I5(zs[2],0)*(np.conj(CKM[1,q])*eu[1,2]*eu[n,2]*CKM[n,d] + np.conj(CKM[2,q])*eu[2,1]*eu[n,1]*CKM[n,d]))
        p1 = -y*ed[q,q]*Lp*(sq1-sq2-sq3)/(2*s2w*np.conj(CKM[2,q])*CKM[2,d]*0.652**4)
        return p1

    def csp_3(Lp,Lm,el):
        sq1, sq2, sq3 = 0,0,0
        for k in range(3):
            for n in range(3):
                sq1 += -2*ts[1]*((ed[d,d]**2)*np.conj(CKM[k,q])*eu[k,1]*CKM[1,d] - (ed[q,q]**2)*np.conj(CKM[1,q])*eu[n,1]*CKM[n,d])
                sq2 += 2*np.log(mul)*(-ed[d,d]*np.conj(CKM[k,q])*eu[k,1]*eu[1,1]*CKM[1,d] - ed[d,d]*np.conj(CKM[k,q])*eu[k,1]*eu[2,1]*CKM[2,d] - ed[d,d]*np.conj(CKM[k,q])*eu[k,2]*eu[1,2]*CKM[1,d] + ((ed[d,d]**2)*np.conj(CKM[k,q])*eu[k,1]*CKM[1,d] - (ed[q,q]**2)*np.conj(CKM[1,q])*eu[n,1]*CKM[n,d])*ts[1])
                sq3 += -2*ed[d,d]*np.conj(CKM[k,q])*eu[k,1]*eu[1,1]*CKM[1,d] + ed[d,d]*(-(ed[q,q]**2)*np.conj(CKM[1,q])*CKM[1,d] + 2*I5(zs[2],0)*np.conj(CKM[k,q])*(eu[k,2]*eu[1,2]*CKM[1,d] + eu[k,1]*eu[2,1]*CKM[2,d]))
        p1 = y*Lm*(sq1+sq2+sq3)/(s2w*np.conj(CKM[2,q])*CKM[2,d]*0.652**4)
        return p1

    C9 = c9_1() + c9_2() + c9_3() + c9_4()
    C9p = c9p_1() + c9p_2() + c9p_3() + c9p_4()
    C10 = c10_1() + c10_2()
    C10p = c10p_1() + c10p_2()
    CS = cs_1(Lp(yh,cba,yH0,sba,el),Lm(yh,cba,yH0,sba,el),el) + cs_2(Lp(yh,cba,yH0,sba,el),Lm(yh,cba,yH0,sba,el),el) + cs_3(Lp(yh,cba,yH0,sba,el),Lm(yh,cba,yH0,sba,el),el)
    CSP = csp_1(Lp(yh,cba,yH0,sba,el),Lm(yh,cba,yH0,sba,el),el) + csp_2(Lp(yh,cba,yH0,sba,el),Lm(yh,cba,yH0,sba,el),el) + csp_3(Lp(yh,cba,yH0,sba,el),Lm(yh,cba,yH0,sba,el),el)
    CP = cs_1(Lp(yh,cba,yH0,sba,-1*el),Lm(yh,cba,yH0,sba,-1*el),-1*el) + cs_2(Lp(yh,cba,yH0,sba,-1*el),Lm(yh,cba,yH0,sba,-1*el),-1*el) + cs_3(Lp(yh,cba,yH0,sba,-1*el),Lm(yh,cba,yH0,sba,-1*el),-1*el)
    CPP = csp_1(Lp(yh,cba,yH0,sba,-1*el),Lm(yh,cba,yH0,sba,-1*el),-1*el) + csp_2(Lp(yh,cba,yH0,sba,-1*el),Lm(yh,cba,yH0,sba,-1*el),-1*el) + csp_3(Lp(yh,cba,yH0,sba,-1*el),Lm(yh,cba,yH0,sba,-1*el),-1*el)

    print('Eq. 3.34: ',csp_1(Lp(yh,cba,yH0,sba,el),Lm(yh,cba,yH0,sba,el),el))
    print('Eq. 3.35: ',csp_2(Lp(yh,cba,yH0,sba,el),Lm(yh,cba,yH0,sba,el),el))
    print('Eq. 6.5: ',csp_3(Lp(yh,cba,yH0,sba,el),Lm(yh,cba,yH0,sba,el),el))

#    print('C9_'+str(q),C9,'\n')
#    print('C9p_'+str(q),C9p,'\n')
#    print('C10_'+str(q),C10,'\n')
#    print('C10p_'+str(q),C10p,'\n')
#    print('CS_'+str(q),CS,'\n')
#    print('CSp_'+str(q),CSP,'\n')
#    print('CP_'+str(q),CP,'\n')
#    print('CPp_'+str(q),CPP,'\n')

    return C9, C9p, C10, C10p, CS, CSP, CP, CPP

