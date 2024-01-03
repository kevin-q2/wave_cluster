from ._disease_model import *
import numpy as np
from scipy.integrate import odeint

###################################################################################################################################
# An continuous-time implementation of the multi-location SIR model - Kevin Quinn 5/23
#
#     INPUT:
#       This inherits from the base disease_model class. It initializes parameters using that 
#       base class (refer to that disease_model.py).
#
#     MODEL EVALUATION::
#       This class evaluates solutions to the SIR model on a (approximately) continuous time scale by integrating
#       the set of ordinary differential equations 

#       derivatives() -- given a state of the system Xt (akin to X0), 
#                       compute the derivatives of the ODE system at time t (used only internally)
#
#       sensitivity_derivatives() - compute derivates with respect to each of the parameters 
#                      for more detail refer to http://www.aimspress.com/article/10.3934/mbe.2012.9.553
#
#       evaluate() -- makes calls to derivatives() to compute the dataset to the initialized SIR system for all t in time_steps
#
#       evaluate_with_sensitivity() - make calls to sensitvity_derivatives()
#
#       calculate_incidence() -- translate prevalence data (original SIR model) to incidence data by Calculating S[t - 1] - S[t] for 
#                                every time step t
#
#########################################################################################################################################


class sir(disease_model):
    def __init__(self, populations, time_steps, X0, Beta, gamma, alpha = 0, init_sensitivity = None):
        super().__init__(populations, time_steps, X0, Beta, gamma)
        self.init_sensitivity = init_sensitivity
        self.alpha = alpha
        

    def derivatives(self, X, t):
        # return derivative values at a particular state (X) of the system
        S = X[:self.n]
        I = X[self.n:2*self.n]
        R = X[2*self.n:]

        der = np.zeros(X.shape)
        for i in range(self.n):
            beta_loc = self.Beta[:,i]
            der[i] = -(S[i]/self.populations[i]) * np.dot(beta_loc, I) + self.alpha * R[i]
            der[self.n + i] = (S[i]/self.populations[i]) * np.dot(beta_loc, I) - self.gamma[i] * I[i]
            der[2*self.n + i] = self.gamma[i] * I[i] - self.alpha * R[i]

        return der
    
    
    # ONLY works for 1 location right now!
    def sensitivity_derivatives(self, X, t):
        # return derivative values at a particular state (X) of the system
        S = X[0]
        I = X[1]
        R = X[2]
        phi1 = X[3]
        phi2 = X[4]
        phi3 = X[5]
        phi4 = X[6]
        
        der = np.zeros(X.shape)
        N = self.populations[0]
        b = self.Beta[0,0]
        g = self.gamma[0]
        
        # SIR derivatives
        der[0] = -(S/N) * b * I
        der[1] = (S/N) * b * I - g*I
        der[2] = g*I
        
        # sensitivity derivatives 
        der[3] = -(b*I/N) * phi1 - (b*S/N) * phi3 - S*I/N 
        der[4] = -(b*I/N) * phi2 - (b*S/N) * phi4
        der[5] = (b*I/N) * phi1 + ((b*S/N) - g) * phi3 + S*I/N 
        der[6] = (b*I/N) * phi2 + ((b*S/N) - g) * phi4 - I 

        return der
    
    
    def evaluate(self, rtol = 1.49012e-8):
        # solve the ordinary differential equation
        ode_solve, info_dict = odeint(self.derivatives, self.X0, range(self.T), full_output = True, rtol = rtol)
        #print(info_dict)
        self.S = ode_solve[:, :self.n]
        self.I = ode_solve[:, self.n:2*self.n]
        self.R = ode_solve[:, 2*self.n:3*self.n]
        
        
    def evaluate_with_sensitivity(self, rtol = 1.49012e-8):
        # solve the ordinary differential equation
        if self.init_sensitivity is None:
            self.init_sensitivity = np.zeros(4)
        start = np.concatenate((self.X0, self.init_sensitivity))
        ode_solve, info_dict = odeint(self.sensitivity_derivatives, start, range(self.T), full_output = True, rtol = rtol)
        #print(info_dict)
        self.S = ode_solve[:,0]
        self.I = ode_solve[:,1]
        self.R = ode_solve[:,2]
        self.phi1 = ode_solve[:,3]
        self.phi2 = ode_solve[:,4]
        self.phi3 = ode_solve[:,5]
        self.phi4 = ode_solve[:,6]
    
    
    def calculate_incidence(self):
        for t in range(1, self.T):
            self.incidence[t - 1] = self.S[t - 1] - self.S[t]