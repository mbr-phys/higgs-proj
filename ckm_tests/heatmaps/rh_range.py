import numpy as np 
import flavio
import matplotlib.pyplot as plt

def rh(mu,md,mm,tanb,mH):
    r = ((mu-md*tanb**2)/(mu+md))*(mm/mH)**2
    return 1/(1+r)

tanbs = [np.linspace(-1,2,300),np.linspace(-1,1,200)]
mHs = [np.linspace(1,3.5,250),np.linspace(2,3.5,150)]

par = flavio.default_parameters.get_central_all()
ckm_els = flavio.physics.ckm.get_ckm(par)
mus = ['m_u','m_c','m_c','m_u','m_u']
mds = ['m_b','m_d','m_s','m_s','m_d']
mms = ['m_B+','m_D+','m_Ds','m_K+','m_pi+']
lats = ['$m_{B^+}$','$m_{D^+}$','$m_{D_s}$','$m_{K^+}$',r'$m_{\pi^+}$']
pngs = ['mB','mD','mDs','mK','mpi']
heatmap = {}

for m in range(len(mus)):
    for x in range(2):
        tanb, mH = tanbs[x], mHs[x]
        heatmap_i = np.empty((mH.size,tanb.size))
        for i in range(mH.size):
            for j in range(tanb.size):
                heatmap_i[i,j] = rh(par[mus[m]],par[mds[m]],par[mms[m]],10**tanb[j],10**mH[i])

        fig_title = r"$\frac{|\tilde{V}_{ij}|}{|V_{ij}|}$ for " + lats[m]
        fig_name = pngs[m] + "-" + "rH" + str(x) + ".png"
#        print("Range for "+fig_title+" is "+str(np.min(heatmap_i))+" to " +str(np.max(heatmap_i)))
        if x == 1:
            heatmap[pngs[m]] = heatmap_i 

#        fig = plt.figure()
#        s = fig.add_subplot(1,1,1,xlabel=r"$\log_{10}[\tan\beta]$",ylabel=r"$\log_{10}[m_{H^+}]$")
#        im = s.imshow(heatmap_i,extent=(tanb[0],tanb[-1],mH[0],mH[-1]),origin='lower')
#        fig.colorbar(im)
#        plt.title(fig_title)
#        plt.savefig(fig_name)
#        plt.show()

Vub, Vcd, Vcs, Vus, Vud, Vcb = ckm_els[0,2], ckm_els[1,0], ckm_els[1,1], ckm_els[0,1], ckm_els[0,0], ckm_els[1,2]
units = np.zeros((mHs[1].size,tanbs[1].size))
for i in range(mHs[1].size):
    for j in range(tanbs[1].size):
        Vubp = heatmap['mB'][i,j]*Vub 
        Vcdp = heatmap['mD'][i,j]*Vcd 
        Vcsp = heatmap['mDs'][i,j]*Vcs 
        Vusp = heatmap['mK'][i,j]*Vus 
        Vudp = heatmap['mpi'][i,j]*Vud 
        r1 = Vudp**2 + Vusp**2 + Vubp**2
        r2 = Vcdp**2 + Vcsp**2 + Vcb**2 # not got mod factor for Vcb yet - maybe incoming?
        if r1 < 1 and r2 < 1: 
            units[i,j] = 1

fig = plt.figure()
s = fig.add_subplot(1,1,1,xlabel=r"$\log_{10}[\tan\beta]$",ylabel=r"$\log_{10}[m_{H^+}]$")
im = s.imshow(units,extent=(tanbs[1][0],tanbs[1][-1],mHs[1][0],mHs[1][-1]),origin='lower',cmap='gray')
#fig.colorbar(im)
plt.title("Modification Regions Allowed By Unitarity using CKM first two rows")
plt.savefig("ckm_mod.png")
#plt.show()
