from ._disease_model import *
import numpy as np
from scipy.integrate import odeint

###################################################################################################################################
# An approximate continuous-time implementation of the SIR model - Kevin Quinn 5/23
#
#     INPUT:
#       This inherits from the base disease_model class. It initializes parameters using that 
#       base class (refer to that disease_model.py).
#
#     MODEL EVALUATION::
#       This class evaluates solutions to the SIR model on a (approximately) continuous time scale by integrating
#       the set of ordinary differential equations 
#
#       derivatives() -- given a state of the system Xt (akin to X0), 
#                       compute the derivatives of the ODE system at time t (used only internally)
#
#       evaluate() -- makes calls to derivatives() to compute the dataset to the initialized SIR system for all t in time_steps
#
#       calculate_incidence() -- translate prevalence data (original SIR model) to incidence data by Calculating S[t - 1] - S[t] for 
#                                every time step t
#
#########################################################################################################################################


class sir(disease_model):
    def __init__(self, population, time_steps, X0, Beta, gamma):
        super().__init__(population, time_steps, X0, Beta, gamma)
        

    def derivatives(self, X, t):
        # return derivative values at a particular state (X) of the system
        S = X[0]
        I = X[1]
        R = X[2]

        der = np.zeros(X.shape)
        der[0] = -(self.Beta * S * I)/self.population
        der[1] = (self.Beta * S * I)/self.population - self.gamma * I
        der[2] = self.gamma * I

        return der
    
    
    def evaluate(self, rtol = 1.49012e-8):
        # solve the ordinary differential equation
        ode_solve, info_dict = odeint(self.derivatives, self.X0, range(self.T), full_output = True, rtol = rtol)
        #print(info_dict)
        self.S = ode_solve[:, 0]
        self.I = ode_solve[:, 1]
        self.R = ode_solve[:, 2]
    
    
    def calculate_incidence(self):
        for t in range(1, self.T):
            self.incidence[t - 1] = self.S[t - 1] - self.S[t]