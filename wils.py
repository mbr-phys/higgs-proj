import numpy as np
from scipy.integrate import quad

lam = 0.2275
mW = 80.379
s2w = 0.23155
mu0 = 80.379
mub = 4.18
mt = 172.9
g = (0.65/(4*np.pi))**2
mmu = 0.1056583745
s = (mmu/mub)**2
A10, R10 = -3.1, -0.23

def cten():
    def Li2(x):
        def func(t):
            z = np.log(1-t)/t
            return z
        inte, err = quad(func,0,x)
        return -1*inte
    def bo(x):
        box = (x*np.log(x))/(4*(1-x)**2) + 1/(4*(1-x))
        return box
    def co(x):
        cox = (3*x**2 +2*x)*np.log(x)/(8*(1-x)**2) + (-x**2 +6*x)/(8*(1-x))
        return cox
    def bi(x):
        bix = (-2*x/((1-x)**2))*Li2(1-1/x) + (-x**2 +17*x)*np.log(x)/(3*(1-x)**3) + (13*x+3)/(3*(1-x)**2) + ((2*x**2 +2*x)*np.log(x)/((1-x)**3) + 4*x/((1-x)**2))*np.log((mu0/mt)**2)
        return bix
    def ci(x):
        cix = ((-x**3 -4*x)/((1-x)**2))*Li2(1-1/x) + (3*x**3 +14*x**2 +23*x)*np.log(x)/(3*(1-x)**3) + (4*x**3 +7*x**2 +29*x)/(3*(1-x)**2) + ((8*x**2 +2*x)*np.log(x)/((1-x)**3) + (x**3+x**2+8*x)/((1-x)**2))*np.log((mW/mt)**2)
        return cix
    def om(s):
        ohm = -(4/3)*Li2(s) - (2/3)*np.log(1-s)*np.log(s) - (2/9)*(np.pi**2) - (5+4*s)*np.log(1-s)/(3*(1+2*s)) - (2*s*(1+s)*(1-2*s))*np.log(s)/(3*(1+2*s)*(1-s)**2) + (5+9*s-6*s**2)/(6*(1-s)*(1+2*s))
        return ohm

    lmu = 2*np.log(mu0/lam)
    lmub = 2*np.log(mub/lam)
    asm0 = (12*np.pi/(23*lmu))*(1 - (348/529)*(np.log(lmu)/lmu))
    asmb = (12*np.pi/(23*lmub))*(1 - (348/529)*(np.log(lmub)/lmub))
    eta = asm0/asmb

    lmuM = 2*np.log(mu0/mt)
    amu = asm0/np.pi
    factor = 1 - (4/3 + lmuM)*amu - (9.125 + 419*lmuM/72 + (2/9)*lmuM**2)*amu**2 - (0.3125*lmuM**3 + 4.5937*lmuM**2 + 25.3188*lmuM + 81.825)*amu**3
    mtmu = mt*factor

    x = (mtmu/mW)**2

    c10_1 = (1/s2w)*(bo(x)-co(x))
    c10_2 = (asm0/(s2w*4*np.pi))*(bi(x)-ci(x)+(4*om(s)/eta)*(bo(x)-co(x)))
    c10 = c10_1 + c10_2

    c10c = (1/(4*s2w))*(1+(asm0/np.pi)*(1+om(s)/eta))

    return c10-c10c

if __name__ == '__main__':
    coeff = cten()
    print(coeff)
