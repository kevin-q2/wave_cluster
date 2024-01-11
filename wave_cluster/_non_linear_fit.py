from ._sir import *
import numpy as np
from scipy.optimize import least_squares

#######################################################################################################################################
#
# The following are a few simple functions designed to 
#  fit parameters to an SIR model 
#  using Non-linear Least Squares Regression - Kevin Quinn 5/23
#
#     INPUT:
#       This inherits from the base disease_model class. It initializes most of its parameters using that 
#       base class (refer to that disease_model.py and sir.py)
#       
#       data -- T dimensional data vector to fit to
#       gamma_est -- (optional) an estimate for the gamma parameter to be used (will only estimate beta)
#       is_incidence -- boolean indicating whether the data being fit corresponds to incidence or prevalence data
#       fit_X0 -- boolean variable specifying if S0, I0 should be fit alongside the beta/gamma parameters
#
#
#   Methods:
#               Let theta be a k-dimensional vector of paramters to fit to ex) [beta, gamma, S0, I0]       
#
#       # using scipy's least squares solver! https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
#       non_linear_solver(bounds, theta0) -- Takes a bounds variable defining bounds on the set of parameters to solve within
#                                           bound is of form (lower, upper) where lower and upper are vectors of size k
#
#                                           Also takes a size k theta0 parameter of initial values for each parameter 
#
#
#########################################################################################################################################


class non_linear_fit(sir):
    def __init__(self, data, population, time_steps, X0, gamma_est = None, is_incidence = False, fit_X0 = True):
        super().__init__(population, time_steps, X0, Beta = None, gamma = gamma_est)
        
        self.data = data
        self.is_incidence = is_incidence
        self.data_est = np.zeros(data.shape)
        self.fit_X0 = fit_X0
    
    
    def solver_call(self):
        # A function to refer to when integrating and evaluating the model
        self.evaluate()
        
        if self.is_incidence:
            self.calculate_incidence()
            return self.incidence
        else:
            return self.I
   


    def residual(self, theta, *args):
        # Computes residuals for the case when the unknown parameters are estimated as theta
        
        # additional info to be supplied in *args are:
        penalty = args[0]
        
        beta_i = theta[0]
        gamma_i = theta[1]
        # solve with model
        self.set_beta(beta_i)
        self.set_gamma(gamma_i)
        
        if self.fit_X0:
            S0_i = theta[2]
            I0_i = theta[3]
            R0_i = self.population - S0_i - I0_i
            X0_i = np.array([S0_i, I0_i, R0_i])
            
            
            # NOTE: There are cases when the least squares solver will attempt to use a paramter set which is 
            # invalid (S + I + R != N). I don't have great fix for this other than to just 
            # severely penalize this case:
            try:
                self.set_initial_compartments(X0_i)
            except ValueError:
                return 1000000
        
        
        I_i = self.solver_call()
        # compute and return residuals
        res = self.data - I_i
        return res.flatten()
    
   
    


    def non_linear_solver(self, bounds, theta0, penalty = 0):
        # Solve for an estimated set of parameters 
        # Penalty parameter not being used currently!

        jacobian = '2-point'
        solver = least_squares(self.residual, theta0, jac = jacobian, bounds = bounds, args = [penalty], ftol = 1e-10)
            
        # recovery estimates for Beta, gamma
        self.set_beta(solver.x[0])
        self.set_gamma(solver.x[1])
        
        # recovery estimates for S0, I0, R0
        if self.fit_X0:
            S0_hat = solver.x[2]
            I0_hat = solver.x[3]
            R0_hat = self.population - S0_hat - I0_hat
            X0_hat = np.array([S0_hat, I0_hat, R0_hat])
            self.set_initial_compartments(X0_hat)
        
            
        return self.Beta, self.gamma, self.X0
    
    
    def generator(self):
        # with parameters already estimated, perform integration to calculate full S,I,R estimates
        self.data_est = self.solver_call()