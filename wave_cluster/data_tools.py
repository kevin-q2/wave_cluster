import numpy as np

def noise(data, mean = 0, std = 1):
    fuzz = np.random.normal(mean, std, data.shape)
    noisy_D = data + fuzz
    noisy_D = noisy_D.clip(min=0)
    return noisy_D



def window(data, window_size):
    rows = data.shape[0]
    columns = data.shape[1]
    windowed_data = np.zeros((rows - window_size + 1, columns))
    for i in range(rows - window_size + 1):
        current_window = data[i:i+window_size,:]
        window_sum = np.sum(current_window,axis=0)
        windowed_data[i,:] = window_sum
        
    return windowed_data


# take a sliding windown average of x using #front (before) elements, current element, and #back (after) elements 
# the # of front elements should include the current index 
# (i.e. front = 7 gives average of the current day and the 6 days before it)
def window_average(X, front, back):
    f = X.shape[0]
    s = front + back + 1
    c = s
    if len(X.shape) > 1:
        sliders = np.zeros((X.shape[0] - s, X.shape[1]))
        while c < f:
            for col in range(X.shape[1]):
                slide = X[c - s: c, col]
                sliders[c - s, col] = np.mean(slide)
            c += 1
    else:
        sliders = np.zeros(X.shape[0] - s)
        while c < f:
            slide = X[c - s: c]
            sliders[c - s] = np.mean(slide)
            c += 1
        
    return np.array(sliders)



def elbow(curve, threshold):
    diffs = []
    for c in range(len(curve) - 1):
        if np.abs(curve[c + 1] - curve[c]) < threshold:
            return c
        
    return None
            
        