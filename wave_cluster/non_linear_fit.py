from ode_model import *
import numpy as np
from scipy.optimize import least_squares

#######################################################################################################################################
#
# The following are a few simple functions designed to 
#  fit Beta/gamma parameters to the multi-location SIR model 
#  using Non-linear Least Squares Regression - Kevin Quinn 3/23
#
#     INPUT:
#       This inherits from the base disease_model class. It initializes parameters using that 
#       base class (refer to that disease_model.py)
#
#       gamma_est -- (optional) an estimate for the gamma parameter to be used (will only estimate beta)
#       is_incidence -- boolean indicating whether the data being fit corresponds to incidence or prevalence data
#       init_sensitivity -- initial values for the sensitvity derivatives (if using sensitvity method for computing jacobian)
#
#
#########################################################################################################################################


class non_linear_fit(ode_sir):
    def __init__(self, Data, populations, time_steps, X0, gamma_est = None, is_incidence = False, init_sensitivity = None, fit_X0 = True):
        super().__init__(populations, time_steps, X0, Beta = None, gamma = gamma_est, init_sensitivity = init_sensitivity)
        self.is_incidence = is_incidence
        self.process_data(Data)
        self.Data_est = np.zeros(Data.shape)
        self.fit_X0 = fit_X0
        
        
    # simple helper to process the data passed as input
    def process_data(self, d):
        # data may either be passed in as a vector:
        if len(d.shape) == 1:
            m = len(d)
            v = np.reshape(d, (m,1))
            self.Data = v
            
        # Or as a matrix
        elif len(d.shape) == 2:
            self.Data = d
            
        else:
            raise ValueError("Data is unshapely")
            
        
    def tri_to_sym(self, v):
        # takes an input vector v representing the lower triangular of a matrix and converts it into 
        # its corresponding symmetric matrix 
        d = int((-1 + np.sqrt(1 + 8*len(v)))/2)
        S = np.zeros((d,d))
        S[np.tril_indices(d, k = 0)] = v
        S = S + S.T - np.diag(np.diag(S))
        return S
    
        
    def sensitivity(self, theta, *args):
        # computes the parameter sensitivities (jacobian) using a system of ordinary differential equations
        # NOTE: this only works for one location right now -- meaning there are only two parameters (Beta, gamma)
        # please refer to https://www.aimspress.com/article/10.3934/mbe.2012.9.553 for more detail on 
        # SIR sensitvity analysis
        sens = args[1]
        if sens == False:
            raise ValueError("Can't compute sensitivities if jac == False")
        
        
        if self.is_incidence:
            jac = np.zeros((self.T - 1, 2))
            for t in range(1, self.T):
                jac[t - 1, 0] = self.phi1[t - 1] - self.phi1[t]
                jac[t - 1, 1] = self.phi2[t - 1] - self.phi2[t]
        else:
            jac = np.zeros((self.T, 2))
            jac[:,0] = self.phi3
            jac[:,1] = self.phi4
            
        return -1*jac
    
    
    def solver_call(self, sens = False):
        # A function to refer to when integrating and evaluating the model
        # Boolean value sens determines if integrals are evaluated with sensitivity derivates or not
        if sens:
            self.evaluate_with_sensitivity()
        else:
            self.evaluate()
        
        if self.is_incidence:
            self.calculate_incidence()
            return self.incidence
        else:
            return np.reshape(self.I, (len(self.I), 1))
   


    def beta_gamma_residual(self, theta, *args):
        # theta: [Beta, gamma] vector 
        # Computes residuals for the case when the unknown parameters are estimated as theta
        # NOTE alot of this was from a long time ago when I was computing Beta matrices and 
        # fitting data matrices. While the code is flexible enough to still work in my current settings 
        # I need to update a lot of this...
        
        # additional info to be supplied in *args are:
        penalty = args[0]
        sens = args[1]
        
        # Beta taken to be the first n(n+1)/2 entries of theta and gamma are the rest
        # Note that this means I'm using a symmetric Beta matrix, which defines an undirected graph
        #beta_i = theta[:int(self.n*(self.n+1)/2)]
        #gamma_i = theta[int(self.n*(self.n+1)/2):]
        
        # CHANGES!!
        beta_i = theta[0]
        gamma_i = theta[1]
        # solve with model
        self.set_beta(beta_i)
        self.set_gamma(gamma_i)
        
        if self.fit_X0:
            S0_i = theta[2]
            I0_i = theta[3]
            R0_i = self.populations[0] - S0_i - I0_i
            X0_i = np.array([S0_i, I0_i, R0_i])
            
            # CHANGES!
            try:
                self.set_initial_compartments(X0_i)
            except ValueError:
                return 100000
        
        
        I_i = self.solver_call(sens)
        # compute and return residuals
        res = self.Data - I_i
        return res.flatten()
    
   
    


    def non_linear_solver(self, bounds, theta0, penalty = 0, jacobian = '2-point'):
        # Solve for an estimated set of parameters!
        
        # Takes a bounds variable defining bounds on the set of parameters to solve within
        # bound is of form (lower, upper) where lower and upper are vectors with the same size as theta
        # theta: [Beta, gamma] vector 
        
        # Also requires an initial parameter theta0 to start the solver with

        # sens determines whether or not the model will be evalutated with sensitivities simultaneously
        sens = False
        
        # jacobian denotes the method for which to compute the jacobian with
        if jacobian == 'sensitivity':
            jacobian = self.sensitivity
            sens = True

        # using scipy's least squares solver! https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
        solver = least_squares(self.beta_gamma_residual, theta0, jac = jacobian, bounds = bounds, args = [penalty, sens], ftol = 1e-10)
            
        # recover estimates for Beta, gamma
        #self.set_beta(self.tri_to_sym(solver.x[:int(self.n*(self.n+1)/2)]))
        #self.set_gamma(solver.x[int(self.n*(self.n+1)/2):])
        self.set_beta(solver.x[0])
        self.set_gamma(solver.x[1])
        
        # CHANGES!
        if self.fit_X0:
            S0_hat = solver.x[2]
            I0_hat = solver.x[3]
            R0_hat = self.populations[0] - S0_hat - I0_hat
            X0_hat = np.array([S0_hat, I0_hat, R0_hat])
            self.set_initial_compartments(X0_hat)
        
            
        #Beta_hat = self.Beta
        #gamma_hat = self.gamma
        
       # if self.fit_X0:
       #     return Beta_hat, gamma_hat, X0_hat
       # else:
       #     return Beta_hat, gamma_hat, self.X0
        return self.Beta, self.gamma, self.X0
    
    
    def generator(self):
        # with parameters already estimated, perform integration to calculate full S,I,R estimates
        self.Data_est = self.solver_call(False)[:,0]