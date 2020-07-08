import numpy as np 
import flavio
import matplotlib.pyplot as plt

def rh(mu,md,mm,tanb,mH):
    r = ((mu-md*tanb**2)/(mu+md))*(mm/mH)**2
    return 1/(1+r)

tanbs = [np.linspace(-1,2,300),np.linspace(-1,1,200)]
mHs = [np.linspace(1,3.5,250),np.linspace(2,3.5,150)]

par = flavio.default_parameters.get_central_all()
mus = ['m_u','m_c','m_c','m_u','m_u']
mds = ['m_b','m_d','m_s','m_s','m_d']
mms = ['m_B+','m_D+','m_Ds','m_K+','m_pi+']
lats = ['$m_{B^+}$','$m_{D^+}$','$m_{D_s}$','$m_{K^+}$',r'$m_{\pi^+}$']
pngs = ['mB','mD','mDs','mK','mpi']

for m in range(len(mus)):
    for x in range(2):
        tanb, mH = tanbs[x], mHs[x]
        heatmap = np.empty((mH.size,tanb.size))
        for i in range(mH.size):
            for j in range(tanb.size):
                heatmap[i,j] = rh(par[mus[m]],par[mds[m]],par[mms[m]],10**tanb[j],10**mH[i])

        fig = plt.figure()
        s = fig.add_subplot(1,1,1,xlabel=r"$\log_{10}[\tan\beta]$",ylabel=r"$\log_{10}[m_{H^+}]$")
        im = s.imshow(heatmap,extent=(tanb[0],tanb[-1],mH[0],mH[-1]),origin='lower')
        fig.colorbar(im)
        fig_title = r"$\frac{|\tilde{V}_{ij}|}{|V_{ij}|}$ for " + lats[m]
        fig_name = pngs[m] + "-" + "rH" + str(x) + ".png"
        plt.title(fig_title)
        plt.savefig(fig_name)
#        plt.show()

