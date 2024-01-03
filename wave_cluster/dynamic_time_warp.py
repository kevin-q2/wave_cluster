import numpy as np
import matplotlib.pyplot as plt


###################################################################################################
# Dynamic Time Warping Implementation 
# -Kevin Quinn 10/23
#
# Input:
#       x - input n dimensional vector 1
#       y - input m dimensional vector 2
#       distance - an input Function which should be a metric from which to compute distances 
#                   between single points. The function should take as input two real valued variables 
#                   and return a single real number distance.
#   
#       mult_penalty - penalties to enforce shape of alignment path 
#                      mult_penalty[0] penalizes steps taken in the direction D[i -1, j]
#                      mult_penalty[1] penalizes steps taken in the direction D[i, j - 1]
#                      mult_penalty[2] penalizes steps taken in the direction D[i -1, j - 1]
#
#       add_penalty - additive penalties to enforce shapes of alignment path
#                      add_penalty[0] penalizes steps taken in the direction D[i -1, j]
#                      add_penalty[1] penalizes steps taken in the direction D[i, j - 1]
#                      add_penalty[2] penalizes steps taken in the direction D[i -1, j - 1]
#
# Output:
#       cost - the total cost of aligment
#       alignment - the computed list of alignment indices  ex. [(0,0), (1,0), (2, 0), ..., (n - 1, m - 1)]
#       cost matrix - the matrix of costs computed at all possible alignment indices
#            
#
###################################################################################################
    


def dtw(x,y, distance, mult_penalty = [1,1,1], add_penalty = [0,0,0]):
    # Initialize variables
    n = len(x)
    m = len(y)
    DTW_cost = np.zeros((n,m))
    DTW_cost[:] = np.inf
    predecessor = {}
    
    # compute cost matrix and track predecessors for alignment
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                DTW_cost[i,j] = distance(x[i], y[j])
            elif i == 0:
                DTW_cost[i,j] = mult_penalty[1] * distance(x[i], y[j]) + DTW_cost[i,j-1] + add_penalty[1]
                predecessor[(i,j)] = (i,j-1)
            elif j == 0:
                DTW_cost[i,j] = mult_penalty[0] * distance(x[i], y[j]) + DTW_cost[i-1,j] + add_penalty[1]
                predecessor[(i,j)] = (i-1,j)
            else:
                # Note how the penalties affect the cost function differently
                lasts = [(i-1,j), (i,j-1), (i-1,j-1)]
                costs = [mult_penalty[0] * distance(x[i], y[j]) + DTW_cost[i-1,j] + add_penalty[0], 
                         mult_penalty[1] * distance(x[i], y[j]) + DTW_cost[i,j-1] + add_penalty[1],
                         mult_penalty[2] * distance(x[i], y[j]) + DTW_cost[i-1,j-1] + add_penalty[2]]
                
                mint = np.argmin(costs)
                # always prefer a diagonal move
                if costs[mint] == costs[2]:
                    mint = 2

                predecessor[(i,j)] = lasts[mint]
                DTW_cost[i,j] = costs[mint]
                
            
            
    # retrace steps to find optimal alignment
    alignment = [(n-1, m-1)]
    current = (n-1,m-1)
    while current != (0,0):
        parent = predecessor[current]
        alignment = [parent] + alignment
        current = parent
        
        
    return DTW_cost[(n-1,m-1)], alignment, DTW_cost
    
    
    

##################################################
# Helper function for plotting an alignment output 
# Takes as input an alignment list such as the one 
# computed by dynamic_time_warp and plots the traced 
# path taken across the n x m cost matrix 
##################################################
def permutation_plot(align):
    xs = [a[0] for a in align]
    ys = [a[1] for a in align]
    fig,ax = plt.subplots(1,1)
    ax.plot(xs,ys) 
    plt.show()




#####################################################
# Helper function for plotting the alignment as it 
# relates to the original vectors being aligned. 
# Takes and plots input vectors x, y and creates 
# connecting lines between each of their aligned indices 
# according to the alignment list, align. Offset controls 
# spacing between x and y in the plot and skips controls 
# the number of alignment indices skipped. 
#####################################################
def align_plot(x,y, align, offset, skips = 2):
    fig,ax = plt.subplots(1,1, figsize = (14,8))

    x_ = np.array(x)
    y_off = np.array(y) + offset

    ax.plot(x_)
    ax.plot(y_off)

    for a in align[::skips]:
        ax.plot([a[0],a[1]], [x_[a[0]], y_off[a[1]]], color = 'black', alpha = 0.4, linestyle = 'dashed', linewidth = 0.75)
    
    plt.show()