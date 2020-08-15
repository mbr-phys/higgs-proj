import numpy as np
from fitting import *

# oblique parameter functions 

sigma = 1.96

def t(x,y,z):
    t=x+y-z
    return t
def r(x,y,z):
    r=z**2-2*z*(x+y)+(x-y)**2
    return r

def f(t,r):
    if r>0:
        f=(r**0.5)*np.log(np.absolute((t-r**0.5)/(t+r**0.5)))
    if r==0:
        f=0
    if r<0:
        f=2*(-r)**0.5*np.arctan((-r)**0.5/t)
    return f

def gxneqy(x,y,z):
    gxneqy=-16/3+5*(x+y)/z-(2*(x-y)**2)*z**(-2)+(3/z)*((x**2+y**2)/(x-y)-((x**2-y**2)/z)+(x-y)**3/(3*z**2))*np.log(x/y)+r(x,y,z)*f(t(x,y,z),r(x,y,z))/(z**3)
    return gxneqy

def g(x,y,z):
    if x!=y and x+y!=z:
        g=gxneqy(x,y,z)
    if x==y:
        g=gxneqy(x+1,x,z)  #small numerical aproximation to take limits
    elif x+y==z:
        g=gxneqy(x,y,x+y+1)
    elif y==z:
        g=gxneqy(x,z+1,z)
    return g

def gTilde(x,y,z):
    gTilde=-2+((x-y)/z-(x+y)/(x-y))*np.log(x/y)+f(t(x,y,z),r(x,y,z))/z
    return gTilde

def gTildeFixed(x,y,z):
    if x!=y:
        gTildeFixed=gTilde(x,y,z)
    if x==y:
        gTildeFixed=gTilde(x,x+0.1,z)
    return gTildeFixed

def gHat(x,z):
    gHat=g(x,z,z)+12*gTildeFixed(x,z,z)
    return gHat

def S2HDMofTheta (mHpm,mA0,mH0,Theta,mW,mZ,mh,Gf,alphaem,wangle):
    S2HDMofTheta=(wangle*Gf*mW**2/(alphaem*12*2**(0.5)*np.pi**2))\
    *(((2*wangle-1)**2)*g(mHpm**2,mHpm**2,mZ**2)\
    +(np.sin(Theta))**2*g(mA0**2,mH0**2,mZ**2)\
    +(np.cos(Theta))**2*g(mA0**2,mh**2,mZ**2)\
    +2*np.log(mA0*mH0*(mHpm**(-2)))\
    +(np.cos(Theta))**2*(gHat(mH0**2,mZ**2)-gHat(mh**2,mZ**2)))
    #S2HDMofTheta=(wangle*Gf*mW**2/(alphaem*12*2**(0.5)*np.pi**2))\
    #*(((2*wangle-1)**2)*g(mHpm**2,mHpm**2,mZ**2)\
    #+(-np.sin(2*Theta))**2*g(mA0**2,mH0**2,mZ**2)\
    #+(np.sin(2*Theta))**2*g(mA0**2,mh**2,mZ**2)\
    #+2*np.log(mA0*mH0*(mHpm**(-2)))\
    #+(np.sin(2*Theta))**2*(gHat(mH0**2,mZ**2)-gHat(mh**2,mZ**2)))
    return S2HDMofTheta

def S2HDMofAlphaBeta (mHpm,mA0,mH0,Alpha,Beta,mW,mZ,mh,Gf,alphaem,wangle):
    S2HDMofAlphaBeta=S2HDMofTheta(mHpm,mA0,mH0,Beta-Alpha,mW,mZ,mh,Gf,alphaem,wangle)
    #S2HDMofAlphaBeta=S2HDMofTheta(mHpm,mA0,mH0,Beta,mW,mZ,mh,Gf,alphaem,wangle)
    return S2HDMofAlphaBeta

def SOb_err(mHp,mA0,mH0,alpha,beta,mW,mW_err,mZ,mZ_err,mh,mh_err,Gf,alphaem,wangle,wan_err):
    S = S2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW,mZ,mh,Gf,alphaem,wangle)

    err1_up = abs(S2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW+mW_err,mZ,mh,Gf,alphaem,wangle)-S)
    err2_up = abs(S2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW,mZ+mZ_err,mh,Gf,alphaem,wangle)-S)
    err3_up = abs(S2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW,mZ,mh+mh_err,Gf,alphaem,wangle)-S)
    err4_up = abs(S2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW,mZ,mh,Gf,alphaem,wangle+wan_err)-S)

    err = sigma*np.sqrt(err1_up**2 + err2_up**2 + err3_up**2 + err4_up**4)
    upper = S+err
    lower = S-err

    return S, upper, lower

def U2HDMofTheta (mHpm,mA0,mH0,Theta,mW,mZ,mh,Gf,alphaem,wangle):
    U2HDMofTheta=((Gf*mW**2)/(48*2**0.5*np.pi**2*alphaem))*(g(mHpm**2,mA0**2,mW**2)\
    +(np.sin(Theta))**2*g(mHpm**2,mH0**2,mW**2)+(np.cos(Theta))**2*g(mHpm**2,mh**2,mW**2)\
    -(2*wangle-1)**2*g(mHpm**2,mHpm**2,mZ**2)\
    -(np.sin(Theta))**2*g(mA0**2,mH0**2,mZ**2)\
    -(np.cos(Theta))**2*g(mA0**2,mh**2,mZ**2)\
    +(np.cos(Theta))**2*(gHat(mH0**2,mW**2)-gHat(mH0**2,mZ**2))\
    -(np.cos(Theta))**2*(gHat(mh**2,mW**2)-gHat(mh**2,mZ**2)))
    #U2HDMofTheta=((Gf*mW**2)/(48*2**0.5*np.pi**2*alphaem))*(g(mHpm**2,mA0**2,mW**2)\
    #+(-np.sin(2*Theta))**2*g(mHpm**2,mH0**2,mW**2)+(np.sin(2*Theta))**2*g(mHpm**2,mh**2,mW**2)\
    #-(2*wangle-1)**2*g(mHpm**2,mHpm**2,mZ**2)\
    #-(-np.sin(2*Theta))**2*g(mA0**2,mH0**2,mZ**2)\
    #-(np.sin(2*Theta))**2*g(mA0**2,mh**2,mZ**2)\
    #+(np.sin(2*Theta))**2*(gHat(mH0**2,mW**2)-gHat(mH0**2,mZ**2))\
    #-(np.sin(2*Theta))**2*(gHat(mh**2,mW**2)-gHat(mh**2,mZ**2)))
    return U2HDMofTheta

def U2HDMofAlphaBeta (mHpm,mA0,mH0,Alpha,Beta,mW,mZ,mh,Gf,alphaem,wangle):
    U2HDMofAlphaBetta=U2HDMofTheta(mHpm,mA0,mH0,Beta-Alpha,mW,mZ,mh,Gf,alphaem,wangle)
    #U2HDMofAlphaBetta=U2HDMofTheta(mHpm,mA0,mH0,Beta,mW,mZ,mh,Gf,alphaem,wangle)
    return U2HDMofAlphaBetta

def UOb_err(mHp,mA0,mH0,alpha,beta,mW,mW_err,mZ,mZ_err,mh,mh_err,Gf,alphaem,wangle,wan_err):
    U = U2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW,mZ,mh,Gf,alphaem,wangle)

    err1_up = abs(U2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW+mW_err,mZ,mh,Gf,alphaem,wangle)-U)
    err2_up = abs(U2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW,mZ+mZ_err,mh,Gf,alphaem,wangle)-U)
    err3_up = abs(U2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW,mZ,mh+mh_err,Gf,alphaem,wangle)-U)
    err4_up = abs(U2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW,mZ,mh,Gf,alphaem,wangle+wan_err)-U)

    err = sigma*np.sqrt(err1_up**2 + err2_up**2 + err3_up**2 + err4_up**4)
    upper = U+err
    lower = U-err

    return U, upper, lower

def F(x,y):
    if x!=y:
        F=(x+y)/2-((x*y)*(x-y)**(-1))*np.log(x/y)
    elif x==y:
        F=0
    return F

def T(MHpm,MA0,MH0,Theta,mW,mZ,mh,Gf,alphaem):
    T=(Gf/(8*(2**0.5)*alphaem*(np.pi)**2))*(F(MHpm**2,MA0**2)+(np.sin(Theta))**2*F(MHpm**2,MH0**2)\
    +((np.cos(Theta))**2)*F(MHpm**2,mh**2)-((np.sin(Theta))**2)*F(MA0**2,MH0**2)\
    -((np.cos(Theta))**2)*F(MA0**2,mh**2)\
    +3*((np.cos(Theta))**2)*(F(mZ**2,MH0**2)-F(mW**2,MH0**2))\
    -3*((np.cos(Theta))**2)*(F(mZ**2,mh**2)-F(mW**2,mh**2)))
    #T=(Gf/(8*(2**0.5)*alphaem*(np.pi)**2))*(F(MHpm**2,MA0**2)+(-np.sin(2*Theta))**2*F(MHpm**2,MH0**2)\
    #+((np.sin(2*Theta))**2)*F(MHpm**2,mh**2)-((-np.sin(2*Theta))**2)*F(MA0**2,MH0**2)\
    #-((np.sin(2*Theta))**2)*F(MA0**2,mh**2)\
    #+3*((np.sin(2*Theta))**2)*(F(mZ**2,MH0**2)-F(mW**2,MH0**2))\
    #-3*((np.sin(2*Theta))**2)*(F(mZ**2,mh**2)-F(mW**2,mh**2)))
    return T

def T2HDMofAlphaBeta(MHpm,MA0,MH0,Alpha,Beta,mW,mZ,mh,Gf,alphaem):
    Tofalphabeta = T(MHpm,MA0,MH0,Beta-Alpha,mW,mZ,mh,Gf,alphaem)
    #Tofalphabeta = T(MHpm,MA0,MH0,Beta,mW,mZ,mh,Gf,alphaem)
    return Tofalphabeta

def TOb_err(mHp,mA0,mH0,alpha,beta,mW,mW_err,mZ,mZ_err,mh,mh_err,Gf,alphaem):
    T = T2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW,mZ,mh,Gf,alphaem)

    err1_up = abs(T2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW+mW_err,mZ,mh,Gf,alphaem)-T)
    err2_up = abs(T2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW,mZ+mZ_err,mh,Gf,alphaem)-T)
    err3_up = abs(T2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW,mZ,mh+mh_err,Gf,alphaem)-T)

    err = sigma*np.sqrt(err1_up**2 + err2_up**2 + err3_up**2)
    upper = T+err
    lower = T-err

    return T, upper, lower

def fit(args,ms):
    par, err, Sce, Supe, Sloe, Tce, Tupe, Tloe, Uce, Uupe, Uloe = args
    mHp = 500
    mH0,mA0 = ms

    alpha, beta = 0, np.pi/2
    mW, mW_err, mZ, mZ_err, mh, mh_err = par['m_W'], err['m_W'], par['m_Z'], err['m_Z'], par['m_h'], err['m_h']
    Gf, alphaem, wangle, wan_err = par['GF'], par['alpha_e'], par['s2w'], err['s2w']

    Sc, Sup, Slo = SOb_err(mHp,mA0,mH0,alpha,beta,mW,mW_err,mZ,mZ_err,mh,mh_err,Gf,alphaem,wangle,wan_err)
    Tc, Tup, Tlo = TOb_err(mHp,mA0,mH0,alpha,beta,mW,mW_err,mZ,mZ_err,mh,mh_err,Gf,alphaem)
    Uc, Uup, Ulo = 0,0,0 #UOb_err(mHp,mA0,mH0,alpha,beta,mW,mW_err,mZ,mZ_err,mh,mh_err,Gf,alphaem,wangle,wan_err)
    chi = chisq_simp([Sce,Tce,Uce],[Sc,Tc,Uc],[Supe-Sce,Tupe-Tce,Uupe-Uce],[Sup-Sc,Tup-Tc,Uup-Uc],2)

    return chi

#    S_bool=((Sce >= Sc and Sup >= Sloe) or (Sce <= Sc and Slo <= Supe))
#    T_bool=((Tce >= Tc and Tup >= Tloe) or (Tce <= Tc and Tlo <= Tupe))
#    U_bool=((Uce >= Uc and Uup >= Uloe) or (Uce <= Uc and Ulo <= Uupe))

#    if S_bool and T_bool:# and U_bool:
#        #return [mHp, mH0, mA0]
#        return 1
#    else:
#        return 0
#        #return [float('nan'), float('nan'), float('nan')]
