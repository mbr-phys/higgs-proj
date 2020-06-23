#!/bin/env python3

import flavio
from flavio.statistics.likelihood import FastLikelihood
import flavio.plots as fpl
import matplotlib.pyplot as plt

# List of observables for the fit, the second and third entries are the q^2 bins
my_obs = (
("<Rmue>(B+->Kll)", 1.1, 6.0),
("<Rmue>(B0->K*ll)", 0.045, 1.1),
("<Rmue>(B0->K*ll)", 1.1, 6.0)
)

# If you do multiple fits in one program, make sure the names are different!
FL = FastLikelihood(name = "likelihood test", observables = my_obs)

# Sets up the experimental and theoretical uncertainties on your observables
FL.make_measurement(N = 500, threads = 4)

# Takes two arguments, translates them to some Wilson coefficients, and gives
# you the likelihood at that point in parameter space.
def my_LL(wcs):
    ReC9mu, ImC9mu = wcs
    par = flavio.default_parameters.get_central_all()
    wc = flavio.WilsonCoefficients()
    wc.set_initial({"C9_bsmumu" : ReC9mu + 1j*ImC9mu,
                    "C10_bsmumu" : -(ReC9mu + 1j*ImC9mu)},
                   scale = 4.8, eft = "WET", basis = "flavio")
    return FL.log_likelihood(par, wc)

# Calculate the data to be plotted. Could save this for latter plotting using pickle
C9contour_data = fpl.likelihood_contour_data(my_LL, -8.5, 0, -5, 5,
                                             n_sigma = (1, 2), threads = 4)
   
plt.figure(figsize=(4,5))
fpl.contour(**C9contour_data, contour_args = {"linestyles" : ("solid", "dashed")})
plt.title(R"Complex $\Delta C_9^\mu = - \Delta C_{10}^\mu$")
plt.xlabel(R"$\Re C_9^{\mu} (= - C_{10}^\mu)$")
plt.ylabel(R"$\Im C_9^{\mu} (= - C_{10}^\mu)$")
plt.savefig("complex_c9_flavio.png", bbox_inches="tight")
