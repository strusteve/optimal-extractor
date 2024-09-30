import numpy as np


class metropolis_hastings(object):
    """ Obtain Markov-Chain via a Metropolis Hastings algorithm.

    Parameters
    ----------

    logprob : function
        Function which returns the natural logarithm of the posterior probability. Takes an array of generative model parameters as input.
    """


    def __init__(self, logprob):
        
        self.ext_logprob = logprob


    # Metropolis-hastings MCMC algorithm - returns Markov chain
    def getchain(self, N, guess, stepsizes):
        """
        Run the sampler and return a chain of samples.

        Paramters
        ---------

        N : integer
            Number of algorithm iterations.

        guess : numpy.ndarray
            Starting point in parameter space for MH algorithm.

        stepsizes : numpy.ndarray
            Width of the gaussian proposal distributions. Higher values cause the sampler to traverse larger distances in parameter space. 
        """

        # Markov Chain
        self.chain = np.zeros((N, len(guess)))
        self.chain[0, :] = guess
        accepted = 0


        # Metropolis-Hastings Algorithm
        for i in range(N-1):

            #Get current parameters & subsequent likelihood
            current_params = self.chain[i, :]
            current_prob = self.ext_logprob(current_params)

            #Propose new parameters (step in parameter space) from gaussian proposal distributions & subsequent likelihood
            new_params = current_params + (stepsizes * np.random.randn(current_params.shape[0]))
            new_prob = self.ext_logprob(new_params)

            #Calculate acceptance ratio (NB uniform priors and symmetric proposal distributions)
            log_prob_ratio = new_prob - current_prob

            # Generate random number from 0-1
            mu = np.random.rand()

            #Accept or reject step (kept in logarithm due to underflow errors)
            if (np.log(mu) < min(np.log(1), log_prob_ratio)):
                self.chain[i+1] = new_params
                accepted += 1
            else:
                self.chain[i+1] = current_params

        # Print acceptance ratio for optimisation of stepsizes
        #print('Acceptance rate = ' + str((accepted/N)*100) + '%')

        return(self.chain)