import numpy as np

############################################################################################################################################
#
# Base class object for multi-population SIR disease modeling - Kevin Quinn 5/23
#
# The multi-population compartmental disease model acts over a group of subpopulations, where each of which is 
# an important location or group in the disease spreading process. These locations spread disease both within themselves 
# and towards other locations. For more details please refer to: https://www.math.uh.edu/~jmorgan/Math5341/Assignments/FinalExamPaper.pdf
#
#   INPUT:
#       populations -- A length n vector containing a population count for each location/group
#
#       time_steps -- a length n (monotone increasing) vector representing the time-steps over which the SIR process happens 
#
#       X0 -- A length (# of compartments) x n vector, This parameter describes the initial (at time step 0) number
#           of susceptible, infected, recovered, or other compartment allocated persons for each location/group. 
#           In a standard SIR model the first n entries of X0 be S0 (the initial number of susceptible persons for each location),
#           the next n are I0, and the last n are R0 (infected and recovered respectively)
#
#       Beta -- The (n x n) spread parameter matrix where each entry Beta[j,i] represents the represents the average number of unique
#           susceptible individuals from location i that an infected person from location j comes into contact with at every time-step
#
#       gamma -- A length n recovery parameter with each entry n[i] denoting the rate at which members of population i recover from
#                the disease 
#
#       type -- the type of compartmental model to be used (Currently only doing SIR models! more to be seen in the future) 
#
#   INITIALIZATION: 
#       set_initial_compartments() -- checks the X0 parameters and appropriately assigns them to compartments
#
#       set_beta() -- Ensures the gamma parameter is properly defined
#
#       set_gamma() -- checks the gamma parameter to ensures its properly defined
#
#
############################################################################################################################################

class disease_model:
    def __init__(self, populations, time_steps, X0, Beta, gamma, type = 'SIR'):
        self.populations = np.array(populations)
        self.time_steps = np.array(time_steps)
        self.n = len(self.populations) 
        self.T = len(self.time_steps)
        
        self.X0 = X0
        self.S0 = None
        self.I0 = None
        self.R0 = None
        self.set_initial_compartments(X0)
        
        self.S = np.zeros((self.T,self.n))
        self.S[0,:] = self.S0
        self.I = np.zeros((self.T,self.n))
        self.I[0,:] = self.I0
        self.R = np.zeros((self.T,self.n))
        self.R[0,:] = self.R0
        
        self.incidence = np.zeros((self.T - 1,self.n))
        
        self.Beta = None
        self.set_beta(Beta)
        
        self.gamma = None
        self.set_gamma(gamma)
        
    
    def set_initial_compartments(self, X0):
        # 3 x n vector in the SIR case
        if len(X0) == 3 * self.n:
            self.X0 = X0
            self.S0 = X0[:self.n]
            self.I0 = X0[self.n:2*self.n]
            self.R0 = X0[2*self.n:]
            
            if not np.array_equal(self.S0 + self.I0 + self.R0, self.populations):
                raise ValueError("S + I + R != N")
            
        else:
            raise ValueError("X0 parameter is unshapely")
        
        
    
    def set_beta(self, Beta):
        #Beta might be none if it's unknown (fitting)
        if Beta is None:
            return 
        
        # Or it could be a single real number
        elif isinstance(Beta, float):
            if self.n == 1:
                self.Beta = np.array([[Beta]])
            else:
                self.Beta = np.diag([Beta]*self.n)
            
        #Beta can be passed as a vector
        elif len(Beta.shape) == 1:
            # either as a length n^2 vector 
            if len(Beta) == self.n**2:
                self.Beta = Beta.reshape((self.n, self.n))
                
            # or as a length n(n+1)/2 vector (lower triangle of a symmetric matrix)
            elif len(Beta) == self.n * (self.n + 1) / 2:
                Bs = np.zeros((self.n, self.n))
                Bs[np.tril_indices(Bs.shape[0], k = 0)] = Beta
                self.Beta = Bs + Bs.T - np.diag(np.diag(Bs))
                
            else:
                raise ValueError("Beta parameter is unshapely")
            
        # More standard, beta is passed in as an n x n matrix
        else:
            if Beta.shape == (self.n, self.n):
                self.Beta = Beta
            else:
                raise ValueError("Beta parameter is unshapely")
            
        # Non-negative parameters
        if (self.Beta < 0).any():
            raise ValueError("Beta contains negative values")
        
        
            
            
    def set_gamma(self, gamma):
        # set gamma either as a length n vector or as a single value 
        # (so that all locations have the same value of gamma)
        if gamma is None:
            return
        elif isinstance(gamma, float):
            self.gamma = np.array([gamma])
        elif len(gamma) == self.n:
            self.gamma = gamma
        elif len(gamma) == 1:
            self.gamma = np.repeat(gamma,self.n)
        else:
            raise ValueError("Gamma parameter is unshapely")
        
        # Non-negative parameters
        if (self.gamma < 0).any() or (self.gamma > 1).any():
            raise ValueError("gamma contains negative values or values greater than 1")
        
        
        
    
    
        
        