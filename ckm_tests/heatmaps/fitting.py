from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.special import gamma
from scipy.integrate import quad 

def chisq_simp(obs,the,sige,sigt,sigma):
    '''
        chisq val, all parameters lists of values, simple
    '''
    N = len(obs)
    chi = 0
    for i in range(N):
        sig = np.sqrt((sige[i]/sigma)**2 + (sigt[i]/sigma)**2)
        val = (obs[i]-the[i])/sig
        chi += val**2
    return chi

def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges

def chi_del(chi_min,chis,hs,ts,parm):
    '''
        computes delta chisq, for several CL giving by parm, using PDG statistics for delt chisq
    '''
    h_min,t_min = [],[]
    for i in range(len(hs)):
        if (chis[i]-chi_min) <= parm:
            h_min = np.append(h_min,hs[i])
            t_min = np.append(t_min,ts[i])

    points = np.vstack([t_min,h_min]).T
    edges = alpha_shape(points,alpha=0.03,only_outer=True)

    return h_min, t_min, [edges,points]

def chi_del_threed(chi_min,chis,hps,hos,tbs,parm):
    '''
        delta chisq fitting for threed fit
    '''
    hp_min,tb_min,ho_min = [],[],[]
    for i in range(len(hps)):
        if (chis[i]-chi_min) <= parm:
            hp_min = np.append(hp_min,hps[i])
            ho_min = np.append(ho_min,hos[i])
            tb_min = np.append(tb_min,tbs[i])

    return hp_min, ho_min, tb_min

def p_vals(chi_min,nu):
    '''
        testing fit validity (following Hughes and Hase ~ pg 104)
    '''
    def chi_dist(chi,nu):
        '''
            chisq probability distribution
        '''
        X = (chi**(nu/2 - 1) * np.exp(-chi/2))/(2**(nu/2) * gamma(nu/2))
        return X
    P, err = quad(chi_dist,chi_min,np.inf,args=(nu))
    return P, err












