import flavio
import numpy as np
import pkg_resources
from flavio.physics.running.running import get_alpha, get_mt
from scipy.integrate import quad
import scipy

def Cgen(C00,C10,C20,C11,C21,C22,ats,kap):
    i = C00 + ats*C10 + C20*ats**2 + ats*kap*C11 + C21*kap*ats**2 + C22*(kap*ats)**2
    return i

def C1(par,mub):
    a_es = get_alpha(par,mub,nf_out=5)
    a_s = a_es['alpha_s']
    kaps = a_es['alpha_e']/a_s
    eta = get_alpha(par,par['m_W'],nf_out=5)['alpha_s']/a_s
    mtmu = get_mt(par,par['m_t'])
    x = (mtmu/par['m_W'])**2

    C10 = 15 + 6*np.log((mub/par['m_W'])**2)
    L = np.log((mub/par['m_W'])**2)
    C20 = -(T(x)- 7987/72 - 17*(np.pi**2)/3 - 475*L/6 - 17*L**2)

    Cone = Cgen(0,C10,C20,0,0,0,a_s/(4*np.pi),kaps)

    return Cone

def C2(par,mub):
    a_es = get_alpha(par,mub,nf_out=5)
    a_s = a_es['alpha_s']
    kaps = a_es['alpha_e']/a_s
    eta = get_alpha(par,par['m_W'],nf_out=5)['alpha_s']/a_s
    mtmu = get_mt(par,par['m_t'])
    x = (mtmu/par['m_W'])**2

    C00 = 1
    C11 = -7/3 - (4/3)*np.log((mub/par['m_Z'])**2)
    L = np.log((mub/par['m_W'])**2)
    C20 = -(-127/18 - 4*(np.pi**2)/3 - 46/3*L - 4*L**2)
    
    Ctwo = Cgen(C00,0,C20,C11,0,0,a_s/(4*np.pi),kaps)

    return Ctwo

def C3(par,mub):
    a_es = get_alpha(par,mub,nf_out=5)
    a_s = a_es['alpha_s']
    kaps = a_es['alpha_e']/a_s
    eta = get_alpha(par,par['m_W'],nf_out=5)['alpha_s']/a_s
    mtmu = get_mt(par,par['m_t'])
    x = (mtmu/par['m_W'])**2

    C11 = (2/(9*par['s2w']))*(X(x)-2*Y(x))
    L = np.log((mub/par['m_W'])**2)
    C20 = Gt1(x,(mub/mtmu)**2) - (680/243 + (20/81)*np.pi**2 + 68*L/81 + (20/27)*L**2)

    Cthree = Cgen(0,0,C20,C11,0,0,a_s/(4*np.pi),kaps)

    return Cthree

def C4(par,mub):
    a_es = get_alpha(par,mub,nf_out=5)
    a_s = a_es['alpha_s']
    kaps = a_es['alpha_e']/a_s
    eta = get_alpha(par,par['m_W'],nf_out=5)['alpha_s']/a_s
    mtmu = get_mt(par,par['m_t'])
    x = (mtmu/par['m_W'])**2
    L = np.log((mub/par['m_W'])**2)

    C10 = E(x) - 2/3 + (2/3)*np.log((mub/par['m_W'])**2)
    C20 = Et1(x,(mub/mtmu)**2) - (-950/243-(10/81)*np.pi**2-124*L/27-(10/27)*L**2)

    Cfour = Cgen(0,C10,C20,0,0,0,a_s/(4*np.pi),kaps)

    return Cfour

def C5(par,mub):
    a_es = get_alpha(par,mub,nf_out=5)
    a_s = a_es['alpha_s']
    kaps = a_es['alpha_e']/a_s
    eta = get_alpha(par,par['m_W'],nf_out=5)['alpha_s']/a_s
    mtmu = get_mt(par,par['m_t'])
    x = (mtmu/par['m_W'])**2
    L = np.log((mub/par['m_W'])**2)

    C11 = -(1/(18*par['s2w']))*(X(x)-2*Y(x))
    C20 = -Gt1(x,(mub/mtmu)**2)/10 + 2*Et0(x)/15 - (68/243 - (2/81)*np.pi**2 - 14*L/81 -(2/27)*L**2)

    Cfive = Cgen(0,0,C20,C11,0,0,a_s/(4*np.pi),kaps)

    return Cfive

def C6(par,mub):
    a_es = get_alpha(par,mub,nf_out=5)
    a_s = a_es['alpha_s']
    kaps = a_es['alpha_e']/a_s
    eta = get_alpha(par,par['m_W'],nf_out=5)['alpha_s']/a_s
    mtmu = get_mt(par,par['m_t'])
    x = (mtmu/par['m_W'])**2
    L = np.log((mub/par['m_W'])**2)

    C20 = -3*Gt1(x,(mub/mtmu)**2)/16 + Et0(x)/4 - (-85/162 - (5/108)*np.pi**2 - 35*L/108 - (5/36)*L**2)

    Csix = Cgen(0,0,C20,0,0,0,a_s/(4*np.pi),kaps)

    return Csix

def C9(par,mub):
    a_es = get_alpha(par,mub,nf_out=5)
    a_s = a_es['alpha_s']
    kaps = a_es['alpha_e']/a_s
    eta = get_alpha(par,par['m_W'],nf_out=5)['alpha_s']/a_s
    mtmu = get_mt(par,par['m_t'])
    x = (mtmu/par['m_W'])**2
    xht = (par['m_h']/mtmu)**2
    L = np.log((mub/par['m_W'])**2)

    C11 = Y(x)/par['s2w'] + W(x) + 4/9 - (4/9)*np.log((mub/mtmu)**2)
    C21 = (1-4*par['s2w'])*Ct1(x,(mub/mtmu)**2)/par['s2w'] - Bt1(x,(mub/mtmu)**2)/par['s2w'] - Dt1(x,(mub/mtmu)**2) - (-1/par['s2w'] - 524/729 + (128/243)*np.pi**2 + 16*L/3 + (128/81)*L**2)
    C22 = (-x**2)/(32*par['s2w']**2) * (4*par['s2w']-1)*(3+taub2(xht)-delt(mub,xht,mtmu))

    Cnine = (a_s/(4*np.pi))*Cgen(0,0,0,C11,C21,C22,a_s/(4*np.pi),kaps)

    return Cnine

def C10(par,mub):
    a_es = get_alpha(par,mub,nf_out=5)
    a_s = a_es['alpha_s']
    kaps = a_es['alpha_e']/a_s
    eta = get_alpha(par,par['m_W'],nf_out=5)['alpha_s']/a_s
    mtmu = get_mt(par,par['m_t'])
    x = (mtmu/par['m_W'])**2
    xht = (par['m_h']/mtmu)**2
    L = np.log((mub/par['m_W'])**2)

    C11 = -Y(x)/par['s2w']
    C21 = (Bt1(x,(mub/mtmu)**2)-Ct1(x,(mub/mtmu)**2))/par['s2w'] - 1/par['s2w']
    C22 = (-x**2)/(32*par['s2w']**2) * (3 + taub2(xht) - delt(mub,xht,mtmu))

    Cten = (a_s/(4*np.pi))*Cgen(0,0,0,C11,C21,C22,a_s/(4*np.pi),kaps)

    print(Cten)
    return Cten

def Csev(par,mub):
    a_es = get_alpha(par,mub,nf_out=5)
    a_s = a_es['alpha_s']
    kaps = a_es['alpha_e']/a_s
    eta = get_alpha(par,par['m_W'],nf_out=5)['alpha_s']/a_s

    mtmu = get_mt(par,par['m_W'])

    hi = [626126/272277,-56281/51730,-3/7,-1/14,-0.6494,-0.0380,-0.0186,-0.0057]
    hpi = [50090080/131509791,-107668/646625,-3254504085930274/23167509579260865,34705151/143124975,-0.2502,0.1063,-0.0525,0.0213]
    hppi = [10974039505456/21104973066375,-13056852574/29922509799,-718812/6954395,-154428730/12196819523,-0.1374,-0.0078,-0.0023,-0.0001]
    ai = [14/23,16/23,6/23,-12/23,0.4086,-0.4230,-0.8994,0.1456]
    fi = [-17.3023,8.5027,4.5508,0.7519,2.0040,0.7476,-0.5385,0.0914]
    ei = [4661194/816831,-8516/2217,0,0,-1.9043,-0.1008,0.1216,0.0183]
    gi = [14.8088,-10.8090,-0.8740,0.4218,-2.9347,0.3971,0.1600,0.0225]

    x = (mtmu/par['m_W'])**2

    hi_et = 0
    for i in range(8):
        hi_et += hi[i]*eta**ai[i]

    fi_et = 0
    for i in range(8):
        fi_et += (ei[i]*eta*E(x) + fi[i] + gi[i]*eta)*eta**ai[i]

    pi_et = 0
    for i in range(8):
        pi_et += hpi[i]*eta**ai[i] + hppi[i]*eta**(ai[i]-1)

    C_70 = (3*x**3 - 2*x**2)*np.log(x)/(4*(x-1)**4) + (-8*x**3 - 5*x**2+7*x)/(24*(x-1)**3)
    C_80 = -3*(x**2)*np.log(x)/(4*(x-1)**4) + (-x**3+5*x**2+2*x)/(8*(x-1)**3)
    C7eff_0 = (eta**(16/23))*(C_70) + (8/3)*(eta**(14/23) - eta**(16/23))*(C_80) + hi_et

    C_71 = (-16*x**4 - 122*x**3 + 80*x**2 - 8*x)*Li2(1-1/x)/(9*(x-1)**4) + (6*x**4 + 46*x**3 - 28*x**2)*(np.log(x)**2)/(3*(x-1)**5) + (-102*x**5 - 588*x**4 - 2282*x**3 + 3244*x**2 - 1364*x + 208)*np.log(x)/(81*(x-1)**5) + (1646*x**4 + 12205*x**3 - 10740*x**2 + 2509*x - 436)/(436*(x-1)**4)
    C_81 = (-4*x**4 + 40*x**3 + 41*x**2 + x)*Li2(1-1/x)/(6*(x-1)**4) + (-17*x**3 - 31*x**2)*(np.log(x)**2)/(2*(x-1)**5) + (-210*x**5 + 1086*x**4 + 4893*x**3 + 2857*x**2 - 1994*x + 280)*np.log(x)/(216*(x-1)**5) + (737*x**4 - 14102*x**3 - 28209*x**2 + 610*x - 508)/(1296*(x-1)**4)
    C7eff_1 = (eta**(39/23))*(C_71) + (8/3)*(eta**(37/23) - eta**(39/23))*(C_81) + ((297664/14283)*eta**(16/23) - (7164416/257075)*eta**(14/23) + (256868/14283)*eta**(37/23) - (6698884/357075)*eta**(39/23))*C_80 + (37208/4761)*(eta**(39/23) - eta**(16/23))*C_70 + fi_et
    C7eff_01 = ((88/575)*eta**(16/23) - (40/69)*eta**(-7/23) + (32/75)*eta**(-9/23))*C_70 + (-(704/1725)*eta**(16/23) + (640/1449)*eta**(14/23) + (32/1449)*eta**(-7/23) - (32/575)*eta**(-9/23))*C_80 - (526074716/4417066408125)*eta**(-47/23) + (65590/1686113)*eta**(-20/23) + pi_et

    C7game = (1/par['s2w'])*(1.11 - 1.15*(1-(mtmu/170)**2) - 0.44*np.log(par['m_h']/100) - 0.21*np.log(par['m_h']/100)**2 - 0.513*np.log(par['m_h']/100)*np.log(mtmu/170)) #+ ((8/9)*C7g - 104/243)*np.log(
    C8game = (1/par['s2w'])*(-0.143 - 0.156*(1-(mtmu/170)**2) - 0.129*np.log(par['m_h']/100) - 0.0244*np.log(par['m_h']/100)**2 - 0.037*np.log(par['m_h']/100)*np.log(mtmu/170)) #+ ((8/9)*C7g - 104/243)*np.log(
    C2e = -22/9 + (4/3)*np.log((par['m_Z']/par['m_W'])**2) + 1/9
    C3e = -(1/par['s2w'])*((4/9)*Box(x) + (2/9)*Cox(x))
    C5e = (1/par['s2w'])*((1/9)*Box(x) + (1/18)*Cox(x))
    C7e = 4*Cox(x) + Dox(x) - (1/par['s2w'])*((10/3)*Box(x) - (4/3)*Cox(x))
    C9e = (1/par['s2w'])*((5/6)*Box(x) - (1/3)*Cox(x))
    C7eff_11 = C7game*eta**(16/23) + (8/3)*(eta**(14/23)-eta**(16/23))*C8game - (0.448-0.49*eta)*C2e + (0.362-0.454*eta)*C3e + (5.57-5.86*eta)*C5e + (0.321-0.47*eta)*C7e + (1.588-2.89*eta)*C9e

    C7eff = C7eff_0 + C7eff_1*a_s/(4*np.pi) + kaps*C7eff_01 + C7eff_11*kaps*a_s/(4*np.pi)
    C8eff = (C_80 + 313063/363036)*eta**(14/23) - 0.9135*eta**0.4086 + 0.0873*eta**(-0.4230) - 0.0571*eta**(-0.8994) + 0.0209*eta**0.1456

    return C7eff, C8eff

def C3Q(par,mub):
    a_es = get_alpha(par,mub,nf_out=5)
    a_s = a_es['alpha_s']
    kaps = a_es['alpha_e']/a_s
    eta = get_alpha(par,par['m_W'],nf_out=5)['alpha_s']/a_s
    mtmut = get_mt(par,par['m_t'])

    x = (mtmut/par['m_W'])**2

    C11 = (2/(3*par['s2w']))*(X(x)+Y(x)) - W(x) - 4/9 + (4/9)*np.log((mub/mtmut)**2)

    Ctq = Cgen(0,0,0,C11,0,0,a_s/(4*np.pi),kaps)

    return Ctq

def C5Q(par,mub):
    a_es = get_alpha(par,mub,nf_out=5)
    a_s = a_es['alpha_s']
    kaps = a_es['alpha_e']/a_s
    eta = get_alpha(par,par['m_W'],nf_out=5)['alpha_s']/a_s
    mtmu = get_mt(par,par['m_t'])

    x = (mtmu/par['m_W'])**2

    C11 = -(1/(6*par['s2w']))*(X(x)+Y(x))

    Cfq = Cgen(0,0,0,C11,0,0,a_s/(4*np.pi),kaps)

    return Cfq

def Cb(par,mub):
    a_es = get_alpha(par,mub,nf_out=5)
    a_s = a_es['alpha_s']
    kaps = a_es['alpha_e']/a_s
    eta = get_alpha(par,par['m_W'],nf_out=5)['alpha_s']/a_s
    mtmu = get_mt(par,par['m_t'])
    x = (mtmu/par['m_W'])**2

    C11 = -S(x)/(2*par['s2w'])

    Cbee = Cgen(0,0,0,C11,0,0,a_s/(4*np.pi),kaps)

    return Cbee

def delt(mu,a,mt):
    i = 18*np.log(mu/mt) + 11 - a/2 + a*(a-6)*np.log(a)/2 + (a-4)*np.sqrt(a)*g(a)/2
    return i

def taub2(a):
    i = 9 - 13*a/4 - 2*a**2 - a*(19+6*a)*np.log(a)/4 - (1/4)*(7-6*a)*(np.log(a)*a)**2 - (1/4 + (7/2)*a**2 - 3*a**3)*(np.pi**2)/6 + (a/2 - 2)*np.sqrt(a)*g(a) + ((a-1)**2)*(4*a-7/4)*Li2(1-a)-(a**3 - 33*(a**2)/4 + 18*a - 7)*f(a)
    return i

def g(a):
    if 0 <= a <= 4:
        return 2*np.sqrt(4-a)*np.arccos(np.sqrt(a/4))
    elif a > 4: 
        return np.sqrt(a-4)*np.log((1-np.sqrt(1-4/a))/(1+np.sqrt(1-4/a)))
    else:
        raise ValueError("a can't be negative")

def f(a):
    def r(t,a):
        return (1+(a-1)*t)/(t*(1-t))
    def func(t,a):
        return Li2(1-r(t,a)) + (r(t,a)/(r(t,a)-1))*np.log(r(t,a))
    inte, err = quad(func,0,1,args=(a))
    return inte

def Box(x):
    i = (-x/(4*(x-1)) + x*np.log(x)/(4*(x-1)**2))
    return i

def Cox(x):
    i = (x*(x-6)/(8*(x-1)) + x*(2+3*x)*np.log(x)/(8*(x-1)**2))
    return i

def Dox(x):
    i = (16-48*x+73*x**2-35*x**3)/(36*(x-1)**3) + (-8+32*x-54*x**2+30*x**3-3*x**4)*np.log(x)/(18*(x-1)**4)
    return i

def E(x):
    ex = x*(18-11*x-x**2)/(12*(1-x)**3) + (15-16*x+4*x**2)*(x**2)*np.log(x)/(6*(1-x)**4) - 2*np.log(x)/3
    return ex

def Li2(x):
    def func(t):
        return np.log(1-t)/t
    inte, err = quad(func,0,x)
    return -1*inte

def Y(x):
    i = 3*np.log(x)*(x**2)/(8*(x-1)**2) + (x**2 - 4*x)/(8*(x-1))
    return i

def X(x):
    i = (3*x**2 - 6*x)*np.log(x)/(8*(x-1)**2) + (x**2 + 2*x)/(8*(x-1))
    return i

def W(x):
    i = (-32*x**4 + 38*x**3 + 15*x**2 - 18*x)*np.log(x)/(18*(x-1)**4) + (-18*x**4 + 163*x**3 - 259*x**2 + 108*x)/(36*(x-1)**3) 
    return i

def S(x):
    i = 3*(x**3)*np.log(x)/(2*(x-1)**3) + (x**3 - 11*x**2 + 4*x)/(4*(x-1)**2)
    return i

def T(x):
    i = -(16*x+8)*np.sqrt(4*x-1)*Cl2(2*np.arcsin(1/(2*np.sqrt(x))))+(16*x+20/3)*np.log(x)+32*x+112/9
    return i

def Cl2(x):
    return np.imag(Li2(np.exp(1j*x)))

def Gt1(x,mul):
    i = (10*x**4 - 100*x**3 + 30*x**2 + 160*x -40)*Li2(1-1/x)/(27*(1-x)**4) + (30*x**3 - 42*x**2 - 332*x + 68)*np.log(x)/(81*(1-x)**4) + (-6*x**3 - 293*x**2 + 161*x + 42)/(81*(1-x)**3) + ((90*x**2-160*x+40)*np.log(x)/(27*(1-x)**4) + (35*x**3 + 105*x**2 - 210*x - 20)/(81*(1-x)**3))*np.log(mul)
    return i

def Et1(x,mul):
    i = (515*x**4 - 614*x**3 - 81*x**2 - 190*x + 40)*Li2(1-1/x)/(54*(1-x)**4) + (-1030*x**4 + 435*x**3 + 1373*x**2 + 1950*x-424)*np.log(x)/(108*(1-x)**5) + (-29467*x**4 + 45604*x**3 - 30237*x**2 + 66532*x - 10960)/(1944*(1-x)**4) + ((-1125*x**3 + 1685*x**2 + 380*x - 76)*np.log(x)/(54*(1-x)**5) + (133*x**4 - 2758*x**3 - 2061*x**2 + 11522*x - 1652)/(324*(1-x)**4))*np.log(mul)
    return i

def Et0(x):
    i = (-9*x**2 + 16*x - 4)*np.log(x)/(6*(1-x)**4) + (-7*x**3 - 23*x**2 + 42*x + 4)/(36*(1-x)**3)
    return i

def Ct1(x,mul):
    i = (-x**3-4*x)*Li2(1-1/x)/((1-x)**2) + (3*x**3+14*x**2+23*x)*np.log(x)/(3*(1-x)**3) + (4*x**3+7*x**2+29*x)/(3*(1-x)**2) + ((8*x**2+2*x)*np.log(x)/((1-x)**3) + (x**3+x**2+8*x)/((1-x)**2))*np.log(mul)
    return i 

def Bt1(x,mul):
    i = -2*x*Li2(1-1/x)/((1-x)**2) + (-x**2+17*x)*np.log(x)/(3*(1-x)**3) + (13*x+3)/(3*(1-x)**2) + ((2*x**2+2*x)*np.log(x)/((1-x)**3) + 4*x/((1-x)**2))*np.log(mul)
    return i

def Dt1(x,mul):
    i = (380*x**4 - 1352*x**3 + 1656*x**2 - 784*x+256)*Li2(1-1/x)/(81*(1-x)**4) + (304*x**4+1716*x**3 - 4644*x**2 + 2768*x - 720)*np.log(x)/(81*(1-x)**5) + (-6175*x**4 + 41608*x**3 - 66723*x**2 + 33106*x - 7000)/(729*(1-x)**4) + ((648*x**4 - 720*x**3 - 232*x**2 - 160*x + 32)*np.log(x)/(81*(1-x)**5) + (-352*x**4 + 4912*x**3 - 8280*x**2 + 3304*x - 880)/(243*(1-x)**4))*np.log(mul)
    return i 

pars = flavio.default_parameters.get_central_all()
mubs = np.arange(2,6,0.5)

c1, c2, c3, c4, c5, c6 = [],[],[],[],[],[]
c7, c8, c9, c10, c3Q, c5Q, cb = [],[],[],[],[],[],[]

for i in mubs:
    c1 = np.append(c1,C1(pars,i))
    c2 = np.append(c2,C2(pars,i))
    c3 = np.append(c3,C3(pars,i))
    c4 = np.append(c4,C4(pars,i))
    c5 = np.append(c5,C5(pars,i))
    c6 = np.append(c6,C6(pars,i))
    ct7,ct8 = Csev(pars,i)
    c7 = np.append(c7,ct7)
    c8 = np.append(c8,ct8)
    c9 = np.append(c9,C9(pars,i))
    c10 = np.append(c10,C10(pars,i))
    c3Q = np.append(c3Q,C3Q(pars,i))
    c5Q = np.append(c5Q,C5Q(pars,i))
    cb = np.append(cb,Cb(pars,i))

data = np.load(pkg_resources.resource_filename('flavio.physics', 'data/wcsm/wc_sm_dB1_2_55.npy'))

print("Their calc for C10 is",data[9])

data[0] = c1
data[1] = c2
data[2] = c3
data[3] = c4
data[4] = c5
data[5] = c6
data[6] = c7
data[7] = c8
data[8] = c9
data[9] = c10
data[10] = c3Q
data[12] = c5Q
data[14] = cb

scales = (2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5)
wcsm = scipy.interpolate.interp1d(scales,data,fill_value='extrapolate')
wcsm_mub = wcsm(4.18)
print(wcsm_mub[9])

#np.save("wcs_gam.npy",data)
#np.save(pkg_resources.resource_filename('flavio.physics', 'data/wcsm/wc_sm_dB1_2_55.npy'),data)

