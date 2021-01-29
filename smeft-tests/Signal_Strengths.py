import flavio
from flavio.statistics.likelihood import FastLikelihood
from flavio.classes import Parameter
from flavio.physics.running import running
import flavio.plots as fpl
import matplotlib.pyplot as plt
import wilson
from wilson import wcxf
import numpy as np

pars = flavio.default_parameters

#Including and updating parameters
lambda_QCD = Parameter('lambda_QCD')
lambda_QCD.tex = r"$\Lambda_{QCD}$"
lambda_QCD.description = "QCD Lambda scale"
pars.set_constraint('lambda_QCD','0.2275 + 0.01433 - 0.01372')

vev = Parameter('vev')
vev.tex = r"$v$"
vev.description = "Vacuum Expectation Value of the SM Higgs"
pars.set_constraint('vev','246')

par = flavio.default_parameters.get_central_all()
scale = flavio.config['renormalization scale']['hdecays']
print(scale)

flavio.measurements.read_file('world_avgs.yml')

# Lists of observables for the fit in various scenarios
Comb_meas = [
    'mu_tth(h->VV)',
    'mu_VBF(h->WW)', 'mu_gg(h->WW)', 'mu_tth(h->WW)','mu_Wh(h->WW)', 'mu_Zh(h->WW)',
    'mu_VBF(h->ZZ)', 'mu_gg(h->ZZ)', 'mu_Vh(h->ZZ)', 'mu_tth(h->ZZ)',  
    'mu_gg(h->Zgamma)',
    'mu_VBF(h->gammagamma)', 'mu_Vh(h->gammagamma)', 'mu_gg(h->gammagamma)', 'mu_tth(h->gammagamma)', 'mu_Wh(h->gammagamma)', 'mu_Zh(h->gammagamma)',
    'mu_VBF(h->mumu)', 'mu_gg(h->mumu)',
    'mu_gg(h->tautau)', 'mu_VBF(h->tautau)', 'mu_tth(h->tautau)', 'mu_Wh(h->tautau)', 'mu_Zh(h->tautau)',
    'mu_Vh(h->bb)', 'mu_VBF(h->bb)', 'mu_tth(h->bb)', 'mu_Wh(h->bb)', 'mu_Zh(h->bb)', 'mu_gg(h->bb)', 
    'mu_Zh(h->cc)'
]

CEPC_obs_meas = ['mu_Zh(h->WW)', 'mu_Zh(h->tautau)', 'mu_Zh(h->bb)']

CEPC_obs_SM = ['mu_Zh(h->WW)', 'mu_Zh(h->tautau)', 'mu_Zh(h->bb)', "mu_Zh(h->cc)", "mu_Zh(h->ZZ)", "mu_Zh(h->gammagamma)","mu_Zh(h->mumu)"]

ATLAS_future_meas = ["mu_gg(h->gammagamma)", "mu_gg(h->ZZ)", "mu_gg(h->WW)", "mu_VBF(h->gammagamma)",
    "mu_VBF(h->ZZ)", "mu_VBF(h->WW)", "mu_VBF(h->tautau)", "mu_Vh(h->gammagamma)", "mu_Vh(h->ZZ)", "mu_Vh(h->bb)", "mu_tth(h->gammagamma)"]

ATLAS_future_SM = ["mu_gg(h->gammagamma)", "mu_gg(h->ZZ)", "mu_gg(h->WW)", "mu_gg(h->tautau)", "mu_gg(h->mumu)", "mu_gg(h->Zgamma)",
    "mu_VBF(h->gammagamma)","mu_VBF(h->ZZ)", "mu_VBF(h->WW)", "mu_VBF(h->tautau)", "mu_Vh(h->gammagamma)", "mu_Vh(h->ZZ)", "mu_Zh(h->bb)", "mu_Wh(h->bb)",
    "mu_tth(h->gammagamma)","mu_tth(h->ZZ)"]

# If you do multiple fits in one program, make sure the names are different!,
FL = FastLikelihood(name = "likelihood test", observables = CEPC_obs_SM, include_measurements=["Higgs Future CEPC SM"])

#include_measurements=['ATLAS Run 2 Higgs 139/fb', 'CMS Run 2 Higgs 137/fb','ATLAS Run 2 Higgs 139/fb mumu', 'ATLAS Run 2 Higgs 139/fb Zgamma', 'CMS h->cc 2019', 'CMS Run 2 Higgs 36/fb NR'])
#, 'CMS Run 2 Higgs 36/fb' - Superseded by the 137/fb CMS data; also slightly different channels (36/fb includes:
#' mu_Zh(h->ZZ)', 'mu_Wh(h->gammagamma)', 'mu_Wh(h->ZZ)', 'mu_Zh(h->gammagamma)', but only 'mu_Wh(h->gammagamma)' is of any merit)

# Sets up the experimental and theoretical uncertainties on your observables
FL.make_measurement(N = 500) 

#Working from the forms given on pg. 28 of 2007.01296
def Two_HDM_WC(pars, tanb, cosba):
    v = pars['vev']

    sinba = np.sin(np.arccos(cosba))
    theta_w = np.arcsin((par['s2w'])**0.5)
    tan_tw = np.tan(theta_w)

    #Coupling modifiers
    K_u = sinba + cosba/tanb
    K_d = sinba - cosba*tanb
    K_v = sinba
    
    #K_d = abs(K_d)
    #K_u = abs(K_u)
    #K_v = abs(K_v)

    #Setting up masses and Yukawa couplings
    m_us = [par['m_u'], par['m_c'], running.get_mt(par, par['m_t'])] 
    m_ds = [par['m_d'], par['m_s'], par['m_b']]
    m_ls = [par['m_e'], par['m_mu'], par['m_tau']]
    Y_us = [m_u * np.sqrt(2) /v for m_u in m_us]
    Y_ds = [m_d * np.sqrt(2) /v for m_d in m_ds]
    Y_ls = [m_l * np.sqrt(2) /v for m_l in m_ls]


    #Calculating the couplings to use in flavio (hence kappa - 1)
    C_uH_pre = -1 * (K_u-1) / (v**2)
    C_uHs = [C_uH_pre * Y_u for Y_u in Y_us]

    C_dH_pre = -1 * (K_d-1) / (v**2)
    C_dHs = [C_dH_pre * Y_d for Y_d in Y_ds]
    C_lHs = [C_dH_pre * Y_l for Y_l in Y_ls]

    #Going off the form of the general SM Lagrangian and the usual 2HDM vector coupling modification
    g_p, g = 0.3584551, 0.6534878
    tan_W = tan_tw #g_p / g
    mW = par['m_W'] 

    C_W = (K_v-1)*(mW/v)**2 / (v**2)
    C_B = (tan_W**2) * C_W      #(K_v-1)*(mW*tan_W/v)**2 /(v**2)  
    C_WB = 4*tan_W * C_W        #(K_v-1)*4*tan_W* (mW/v)**2 /(v**2)  

    #Thankfully, it is straightforward to identify these WCs with those in Flavio

    #phi = C_H - Not used in SS calculations (flavio calls it phi also)

    #Not all are used to calculate signal strengths in flavio, eg. all the phi_11s are neglected  
    return C_uHs[1], C_uHs[2], C_dHs[1], C_dHs[2], C_lHs[1], C_lHs[2], C_W, C_B, C_WB

"""
u_22, u_33, d_22, d_33, e_22, e_33, C_W, C_B, C_WB = Two_HDM_WC(par, 1, 0.25)
print("\nFor tanb = 1, cos(b-a) = 0.25:\n")
print("u_22:",u_22,"\n")
print("u_33:",u_33,"\n")
print("d_22:",d_22,"\n")
print("d_33:",d_33,"\n")
print("e_22:",e_22,"\n")
print("e_33:",e_33,"\n")
print("C_W:",C_W,"\n")
print("C_B:",C_B,"\n")
print("C_WB:",C_WB,"\n")
exit()
"""

#The fitting function
def func(wcs):
    tanb, cos_b_a = wcs # state what the two parameters are going to be on the plot
    #cos_b_a = np.cos(b_a)

    par = flavio.default_parameters.get_central_all()
    scale = flavio.config['renormalization scale']['hdecays']
    wc = flavio.WilsonCoefficients()

    u_22, u_33, d_22, d_33, e_22, e_33, C_W, C_B, C_WB = Two_HDM_WC(par, 10**tanb, cos_b_a)
    
    wc.set_initial({"uphi_33": u_33, "uphi_22": u_22,
                    "dphi_33": d_33, "dphi_22": d_22,
                    "ephi_33": e_33, "ephi_22": e_22,
                    "phiW": C_W, "phiWB": C_WB, "phiB": C_B,  
                    },
                    scale = scale, eft = "SMEFT", basis = "Warsaw") 
    return FL.log_likelihood(par,wc)

#Doing the fit
cdat = fpl.likelihood_contour_data(func, -1, 2, -0.2, 0.2,
                n_sigma=(1,2),
                steps = 2500) 

#Plotting
tanb_max = 5
tanb_plot1, tanb_plot2 = np.arange(2, tanb_max+0.01, 0.01), np.arange(1, 2, 0.01)
cosba_plot = 2 / tanb_plot1
sin2b_plot = np.sin(2*np.arctan(10**tanb_plot2))
zero_line_cba, zero_line_tanb = [0,0], [-1,2]

plt.figure(figsize=(6,5))
fpl.contour(**cdat) 
#plt.plot(tanb_plot1, cosba_plot, label = "cos("r'$\beta - \alpha)$'+" = 2cot"r'$\beta$')
#plt.plot(tanb_plot2, sin2b_plot, linestyle = "dashed", color='red', label = "cos("r'$\beta - \alpha)$'+" = sin2"r'$\beta$')
#plt.legend(loc = "upper right")
plt.plot(zero_line_tanb, zero_line_cba, linestyle = "dashed", color='black')
#plt.title(r'All')
plt.xlabel(r"$\log_{10}[\tan\beta]$")
plt.ylabel(r'$\cos (\beta - \alpha )$')
#plt.show()
plt.savefig('SS_CEPC_SMvals.pdf', bbox_inches='tight')



def mHmin(contour):
    '''
        Finding the minimum and maximum values range in the contours
    '''
    x = contour['x']
    y = contour['y']
    z = contour['z']
    levels = contour['levels']

    xf, yf = np.where(z==np.min(z))
    xbf = x[xf[0],yf[0]]
    ybf = y[xf[0],yf[0]]

    minh_loc, mint_loc, maxt_loc = [],[],[]
    for i in levels:
        minh, mint, maxt = 0,0,0
        x_loc, y_loc = np.where(z<i)
        for j in range(len(x_loc)):
            k = (x_loc[j],y_loc[j])
            if y[k] > minh:
                minh = y[k]
            if x[k] < mint:
                mint = x[k]
            if x[k] > maxt:
                maxt = x[k]
        minh_loc.append(minh)
        mint_loc.append(mint)
        maxt_loc.append(maxt)

    return [xbf,ybf], minh_loc, mint_loc, maxt_loc

bf,minh,mint,maxt = mHmin(cdat)
#print("Best fit value is found for (tanb, cosba) =", bf)
#print("Print outs are lists for values at", sigmas, "sigmas")
#print("Max value of cosba is:", minh)
#print("Minimum value of tanb is:", mint)
#print("Maximum value of tanb is:", maxt)

