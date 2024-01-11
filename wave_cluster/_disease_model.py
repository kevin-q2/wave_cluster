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
#       population -- an integer representing a population count
#
#       time_steps -- a length T (monotone increasing) vector representing the time-steps over which the SIR process happens 
#
#       X0 -- A 3 dimensional vector, This parameter describes the initial (at time step 0) number
#           of susceptible, infected, recovered, or other compartment allocated persons for each location/group. 
#           takes form [S0, I0, R0]
#
#       Beta -- floating point spread parameter which represents the represents the average number of unique
#           susceptible individuals that an infected person comes into contact with at every time-step
#
#       gamma -- floating point recovery parameter  denoting the rate at which members of the population recover from
#                the disease 
#
#
#   INITIALIZATION: 
#       set_initial_compartments() -- checks the X0 parameters and appropriately assigns them to compartments
#
#       set_beta() -- Ensures the beta parameter is properly defined
#
#       set_gamma() -- checks the gamma parameter to ensures its properly defined
#
#
############################################################################################################################################

class disease_model:
    def __init__(self, population, time_steps, X0, Beta, gamma):
        self.population = population
        self.time_steps = np.array(time_steps)
        self.T = len(self.time_steps)
        
        self.X0 = X0
        self.S0 = None
        self.I0 = None
        self.R0 = None
        self.set_initial_compartments(X0)
        
        self.S = np.zeros(self.T)
        self.S[0] = self.S0
        self.I = np.zeros(self.T)
        self.I[0] = self.I0
        self.R = np.zeros(self.T)
        self.R[0] = self.R0
        
        self.incidence = np.zeros(self.T - 1)
        
        self.Beta = None
        self.set_beta(Beta)
        
        self.gamma = None
        self.set_gamma(gamma)
        
    
    def set_initial_compartments(self, X0):
        # 3 x n vector in the SIR case
        if len(X0) == 3:
            self.X0 = X0
            self.S0 = X0[0]
            self.I0 = X0[1]
            self.R0 = X0[2]
            
            if self.S0 + self.I0 + self.R0 != self.population:
                raise ValueError("S + I + R != N")
            
        else:
            raise ValueError("X0 parameter is unshapely")
        
        
    
    def set_beta(self, Beta):
        #Beta might be none if it's unknown (fitting)
        if Beta is None:
            return 
        
        # Or it could be a single real number
        elif isinstance(Beta, float):
            if Beta < 0:
                raise ValueError("Beta contains negative values")
            else:
                self.Beta = Beta
        else: 
            raise ValueError("Beta must be a floating point value")
        
        
            
            
    def set_gamma(self, gamma):
        # set gamma either as a length n vector or as a single value 
        # (so that all locations have the same value of gamma)
        if gamma is None:
            return
        elif isinstance(gamma, float):
            if gamma < 0 or gamma > 1:
                raise ValueError("gamma contains negative values or values greater than 1")
            self.gamma = gamma
            
        else:
            raise ValueError("gamma must be a floating point value")
        
        
        
    
    
        
        