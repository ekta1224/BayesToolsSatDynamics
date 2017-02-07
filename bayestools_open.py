# February 2017 
# These tools were create to perform the analysis in the
# following publication, which has been submitted to MNRAS and will be
# posted to the arXiv in the near future:

# Orbits of massive satellite galaxies - II. Bayesian Estimates of the
# Milky Way and Andromeda masses using high precision astrometry and
# cosmological simulations (E. Patel, G. Besla, K. Mandel), 2017

import math
import numpy as np
from scipy.stats import norm
import scipy

class BayesTools:
    def __init__(self, masses, data):
        """ 
        Inputs:                                                                                                                               
        masses: list of virial masses in 10^10 Msun                                                                                               
        data: vmax, sigma_vmax, pos, sigma_pos, vel, sigma_vel, j, sigma_j for a massive satellite (i.e. LMC or M33)                            

        All calculations are carried out in log space. Note that
        transforming results from log space to linear space is not
        equivalent to calculating posteriors and related quantities
        directly in linear space.

        """
        self.masses = np.log10(masses*1e10)
        self.data = data
        self.vmax, self.vmaxerr, self.pos, self.poserr, self.vel, self.velerr, self.j, self.jerr = self.data
        self.maxM = float(int(np.max(self.masses)+1))
        self.minM = math.floor(np.min(self.masses))
        assert(self.maxM-np.max(self.masses) < 1.)
        self.M = np.arange(self.minM, self.maxM, 0.001)
        print self.minM, self.maxM

    def ln_norm(self, y, mu, s):
        return (-(y-mu)**2./(2*s**2.)) - 0.5*np.log(2.*np.pi*s**2.)

    def vmax_lnweights(self, vmax1):
        vmax = self.vmax
        vmaxerr = self.vmaxerr
        return self.ln_norm(vmax1, vmax, vmaxerr)

    def pos_lnweights(self, pos1):
        pos = self.pos
        poserr = self.poserr
        return self.ln_norm(pos1, pos, poserr)

    def vel_lnweights(self, vel1):
        vel = self.vel
        velerr = self.velerr
        return self.ln_norm(vel1, vel, velerr)

    def j_lnweights(self, j1):
        j = self.j
        jerr = self.jerr
        return self.ln_norm(j1, j, jerr)

    def vmaxj_lnweights(self, vmax1, j1):
        ''' Momentum Likelihood: Eq. 10 '''
        j = self.j
        jerr = self.jerr
        vmax = self.vmax
        vmaxerr = self.vmaxerr
        return self.ln_norm(vmax1, vmax, vmaxerr) + self.ln_norm(j1, j, jerr)

    def vmaxposvel_lnweights(self, vmax1, pos1, vel1):
        ''' Instantaneous Likelihood: Eq. 8 '''
        pos = self.pos
        poserr = self.poserr
        vel = self.vel
        velerr = self.velerr
        vmax = self.vmax
        vmaxerr = self.vmaxerr
        return self.ln_norm(pos1, pos, poserr) + self.ln_norm(vel1, vel, velerr) + self.ln_norm(vmax1, vmax, vmaxerr)

    def weight_mean(self, weights):
        weights = np.exp(weights)
        masses = self.masses
        num = np.sum(masses*weights)
        den = np.sum(weights)
        return num/den

    def weight_std(self, weights):
        masses = self.masses
        bayesmean = self.weight_mean(weights)
        weights = np.exp(weights)
        diff = np.array([(m - bayesmean)**2. for m in masses])
        num = np.sum(diff * weights)
        den = np.sum(weights)
        return np.sqrt(num/den)

    def bayes_confint(self, probs, mean, conflevel):
        ''' 
            Note this function uses the posterior probabilties, not
            the importance sampling weights to compute credible
            intervals.

            Inputs:
            mean: the weighted mean of the posterior PDF
            conflevel: 0.68, 0.9, 0.95, etc.
        '''

        xs = self.M
        N = len(xs)
        k = 512
        ymin = np.min(xs)
        ymax = np.max(xs)
        deltay = (ymax - ymin)/(N-1.)
        probs = np.array(probs)
        probs = probs[~np.isnan(probs)]

        def y(i, N ):
            return ymin + ((i-1.)/(N-1.)*(ymax-ymin))

        yis = [y(ii, N) for ii in np.arange(1,N)]
        Py = probs
        bins = xs
        smallestM = np.min(np.abs(xs-mean))
        closestM = np.where(smallestM == np.abs(xs-mean))[0][0]
        Pmax = probs[closestM] 

        def pjj(j, Pmax):
            return (j-1.)/(k-1.)*Pmax

        Pj = [pjj(j,Pmax) for j in np.arange(1,k)]

        def Cj(pj):
            Cjj = 0
            for i in range(1,N-1):
                if Py[i] > pj:
                    Cjj += deltay*Py[i]
                else: 
                    Cjj += 0.
            return Cjj

        Cjs = [Cj(pj) for pj in Pj]
        C0 = conflevel
        Cjs = np.array(Cjs)

        for jj in range(len(Cjs)-1):
            if Cjs[jj] <= C0 and Cjs[jj+1] >= C0:
                j=jj; break
            elif Cjs[jj] >= C0 and Cjs[jj+1] <= C0:
                j=jj; break


        P0 = ((Pj[j+1] - Pj[j])/(Cjs[j+1] - Cjs[j])*(C0-Cjs[j])) + Pj[j]    

        thisi = []
        ps = []
        for i in range(len(Py)-1): 
            if Py[i] <= P0 and Py[i+1] >= P0: 
                thisi+=[i]
                ps.append(Py[i])
            elif Py[i] >= P0 and Py[i+1] <= P0: 
                thisi+=[i]
                ps.append(Py[i])

        y0s = [((yis[i+1]-yis[i])/(Py[i+1]-Py[i]))*(P0-Py[i])+yis[i] for i in thisi] 

        # if the posterior is bimodal; ensure that the correct limits of credible intervals are chosen
        if len(y0s) > 2: 
            mask1 = (xs <= y0s[1]) * (xs >= y0s[0])
            mask2 = (xs <= y0s[3]) * (xs >= y0s[2])
            mask3 = (xs <= y0s[2]) * (xs >= y0s[0])
            mask4 = (xs <= y0s[3]) * (xs >= y0s[1])
            mask5 = (xs <= y0s[3]) * (xs >= y0s[0])
            res1 = np.abs(scipy.integrate.simps(probs[mask1], xs[mask1])-conflevel)
            res2 = np.abs(scipy.integrate.simps(probs[mask2], xs[mask2])-conflevel)
            res3 = np.abs(scipy.integrate.simps(probs[mask3], xs[mask3])-conflevel)
            res4 = np.abs(scipy.integrate.simps(probs[mask4], xs[mask4])-conflevel)
            res5 = np.abs(scipy.integrate.simps(probs[mask5], xs[mask5])-conflevel)
            allres = [res1, res2, res3, res4, res5]
            minres = np.where(allres==np.min(allres))[0][0]
            print allres
            print minres
            print mean-y0s[0], y0s[3]-mean
            if minres == 0:
                assert(allres[0] < 5e-3)
                return mean-y0s[0], y0s[1]-mean
            if minres == 1:
                assert(allres[1] < 5e-3)
                return mean-y0s[2], y0s[3]-mean
            if minres == 2:
                assert(allres[2] < 5e-3)
                return mean-y0s[0], y0s[2]-mean
            if minres == 3:
                assert(allres[3] < 5e-3)
                return mean-y0s[1], y0s[3]-mean
            if minres == 4:
                assert(allres[4] < 5e-3)
                return mean-y0s[0], y0s[3]-mean
        else:
            mask = (xs <= y0s[1]) * (xs >= y0s[0])
            assert (np.abs(scipy.integrate.simps(probs[mask], xs[mask])-conflevel) < 5e-3)
            return mean-y0s[0], y0s[1]-mean

    def get_ess(self, weights):
        ''' Effective Sample Size: Eq. B2 '''
        weights = np.exp(weights)
        ess = np.sum(weights)**2./(np.sum(weights**2.))
        if ess < 50.:
            print 'CAUTION: The ESS is less than 50!'
        return ess

    def opt_bw(self, weights):
        ''' Optimal Bandwidth: Eq. B1 '''
        std = self.weight_std(weights)
        ess = self.get_ess(weights)
        return (4.*std**5./(3.*ess))**(1./5.)

    def kde_gauss(self, weights):
        ''' Kernel Density Estimation: See Appendix B '''
        masses = self.masses
        M = self.M
        optbw = self.opt_bw(weights)
        kdes = []
        for i in range(len(M)):
            kde = np.exp(weights) * norm.pdf(masses, M[i], optbw)/ np.sum(np.exp(weights))
            kdes.append(np.sum(kde))
        assert(np.abs(1.-scipy.integrate.simps(kdes, M)) < 5e-3)
        return kdes

