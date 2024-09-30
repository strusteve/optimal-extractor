import numpy as np
import scipy as scp
from mh_sampler import metropolis_hastings
import matplotlib.pyplot as plt


class nested_sampler(object):
    """ Calculates the bayesian evidence and samples the posterior through a 'Nested Sampling' algorithm (Skilling 2004).

    Parameters
    ----------

    lnprob : function
        Function which returns the natural logarithm of the posterior probability.
        Takes an array of generative model parameters as input.

    N : integer
        Number of samples drawn from the uniform distribution.

    log_evidence_tol: Float
        Sets the algorithm to stop when the natural logarithm of the ratio between the remaining evidence and the accumulated evidence
        is below this number.

    Returns
    -------

    posterior_samples: numpy.ndarray
        Array of equally-weighted posterior samples.
    """

    def __init__(self, N, logprob, log_evidence_tol):

        self.ext_logprob = logprob
        self.N = N
        self.user_log_evidence_tol = log_evidence_tol
        
    def prob_t(self, t):
        """ The probability distribution for the largest of N samples drawn uniformly from the interval [0, 1].

        Parameters
        ----------

        t : float or numpy.ndarray
            Value(s) at which to calculate the probability.

        Returns
        -------

        log_p : float
            Natural logarithm of the probability.
        """

        if t>0 and t<1:
            log_p = np.log(self.N) + ((self.N-1) * np.log(t))
        else:
            log_p = -1e99

        return log_p


    def sample_prob_t(self, samps, sigma):
        """ Sample the probability distribution for the largest of N samples drawn uniformly from the interval [0, 1], via MH algorithm.
        
        Parameters
        ----------

        samps : integer
            Number of samples drawn from probability distribution.
        
        sigma : float
            Width of gaussian proposal distributions.
        """

        mcmc = metropolis_hastings(self.prob_t)
        self.chain_t = mcmc.getchain(samps, [0.5], sigma).flatten()
        
    def metropolis_prior_sampling(self, sorted_prior_samples, lowest_loglikelihood, sigmas):
        """ Sampler for a 'likelihood-bounded' prior. Adapted from Feroz 2008.

        Parameters
        ----------

        N : integer
            Number of steps taken in the MCMC algorithm.

        sorted_prior_samples : numpy.ndarray
            Array of prior samples sorted by increasing likelihood.
        
        lowest_likelihood : float
            The lowest likelihood associated with the prior samples.
        
        sigmas : numpy.ndarray
            The width of gaussian proposal distributions for each parameter.

        Returns
        -------

        chain[-1] : numpy.ndarray
            A sample drawn from the likelihood bounded prior.

        accepted : integer
            Number of accepted steps in the metropolis-hastings algorithm.

        rejected : integer
            Number of rejected steps in the metropolis-hastings algorithm.
        """

        # Choose any other likelier (live) point
        likelier_point = sorted_prior_samples[1:][np.random.randint(0,self.N-1)]
        N_mh = 20
        accepted=0
        rejected=0
        chain = np.zeros((N_mh, len(likelier_point)))
        chain[0] = likelier_point

        # Begin chain
        for i in range(N_mh-1):

            current_point = chain[i]

            trial_point = np.zeros(len(current_point))
            for j in range(len(trial_point)):
                trial_point[j] = np.random.normal(current_point[j], sigmas[j])
            trial_loglikelihood = self.ext_logprob(trial_point)

            #Calculate prior ratio (NB uniform priors)
            prior_ratio = 1 #Included here for completeness

            #Acceptance ratio (NB symmetric proposal distributions), based on Feroz 2008 Eq 16
            if trial_loglikelihood > lowest_loglikelihood:
                alpha = min(1, prior_ratio)
            else:
                alpha = 0
                
            # Generate random number from 0-1
            mu = np.random.random_sample()

            #Accept or reject step (kept in logarithm due to underflow errors)
            if (mu < alpha):
                chain[i+1] = trial_point
                accepted += 1
            else:
                chain[i+1] = current_point
                rejected += 1

        return chain[-1], accepted, rejected

    def get_weighted_posterior(self, discarded_points, log_evidence_layers):
        """ Sample equally weighted posterior.

        Parameters
        ----------

        discarded_points : numpy.ndarray
            Array of the discarded points from the nested sampling algorithm.

        log_evidence_layers : numpy.ndarray
            Array of the natural logarithm of the accumulated evidence at each layer.
        
        """
        
        # Calculate posterior weights (Feroz 2008 Eq 9)
        posterior_weights = np.exp(log_evidence_layers - scp.special.logsumexp(log_evidence_layers))

        # Sample equally-weighted posterior
        idx = np.random.choice(len(discarded_points), size = len(discarded_points),  p = posterior_weights)
        posterior_samples = discarded_points[idx, :]

        return posterior_samples


    def run_sampler(self, prior_low, prior_high):
        """ Nested sampling routine.

        Parameters
        ----------

        prior_low : numpy.ndarray
            Array of lower bounds for each parameter's uniform prior distribution.

        prior_high : numpy.ndarray
            Array of upper bounds for each parameter's uniform prior distribution.
        
        """

        # Sample 'successive prior ratios' distribution
        chain_t = self.sample_prob_t(100000, 0.01)

        # Variables & Memory
        sigmas = np.ones(len(prior_high))
        loglike_layers = np.array(())
        prior_volumes = np.array(())
        log_evidence_layers = np.array(())
        discarded_points = np.array(())

        ################################################# 

        """ set the outer nest layer """

        # Draw N samples from the full prior
        prior_samples = np.zeros((self.N, len(prior_high)))
        for i in range(len(prior_high)):
            prior_samples.T[i] = np.random.uniform(prior_low[i], prior_high[i], self.N)

        # Evaluate the likelihood for each sample (try to get rid of for loop)
        loglikes = np.zeros(self.N)
        for j in range(self.N):
            loglike = self.ext_logprob(prior_samples[j])
            loglikes[j] = loglike

        # Sort samples in order of their likelihoods
        sorted_loglikes = loglikes[np.argsort(loglikes)]
        loglike_layers = np.append(loglike_layers, sorted_loglikes[0])
        sorted_prior_samples = prior_samples[np.argsort(loglikes)]
        discarded_points = np.append(discarded_points, sorted_prior_samples[0])

        # Set the initial prior volume
        prior_volumes = np.append(prior_volumes, 1)

        # Set the initial accumulated evidence
        log_evidence_layers = np.append(log_evidence_layers, -1e99)
        

        """ loop through nested layers """
        while True:

            # Replace the lowest-likelihood point with another from a likelihood bounded prior via metropolis algorithm
            new_point, accepted, rejected =self.metropolis_prior_sampling(sorted_prior_samples, sorted_loglikes[0], sigmas)
            new_loglike = self.ext_logprob(new_point)
            sorted_prior_samples[0] = new_point
            sorted_loglikes[0] = new_loglike

            # Record lowest likelihood and host point
            sorted_prior_samples = sorted_prior_samples[np.argsort(sorted_loglikes)]
            sorted_loglikes = sorted_loglikes[np.argsort(sorted_loglikes)]
            loglike_layers = np.append(loglike_layers, sorted_loglikes[0])
            discarded_points = np.vstack((discarded_points, sorted_prior_samples[0]))

            # Update metropolis stepsizes for bounded prior sampling, based on Feroz 2008 Eq 17
            if accepted > rejected:
                sigmas = sigmas * np.exp(1/accepted)
            elif accepted <= rejected:
                sigmas = sigmas * np.exp(-1/rejected)
            
            # Update the remaining prior volume for the new layer
            new_prior_volume = self.chain_t[np.random.randint(0, self.chain_t.shape[0])] * prior_volumes[-1]
            prior_volumes = np.append(prior_volumes, new_prior_volume)

            # Accumulate evidence
            weight = prior_volumes[-2] - prior_volumes[-1]
            log_evidence_contrib = sorted_loglikes[0] + np.log(weight)
            log_evidence_layers = np.append(log_evidence_layers, log_evidence_contrib)

            # Stopping criterion (Feroz 2008)
            log_max_contrib = sorted_loglikes[-1] + np.log(prior_volumes[-1])
            log_evidence = scp.special.logsumexp(log_evidence_layers)

            #print(f'log[total / accumulated] evidence: {int(log_max_contrib - log_evidence)}')

            if log_max_contrib - log_evidence < self.user_log_evidence_tol:
                posterior_samples = self.get_weighted_posterior(discarded_points, log_evidence_layers)
                return posterior_samples