import numpy as np


##########################################################################################################################
# Random Program for splitting a data vector into segments
#
#   INPUT:
#       data_vector -- to split
#
#       segments -- the number of distinct segments to split the data into
#
#       segment_size -- the minimum length a segment can take
#
#  
#   ATTRIBUTES:
#       cuts -- a list of cut points used to segment the data
#
# ---Kevin Quinn 11/23
#############################################################################################################################


class random_segment:
    def __init__(self, data_vector, segments, segment_size = 1):
        self.segments = segments
        self.segment_size = segment_size
        self.T = len(data_vector)
        
        if self.T < self.segments * self.segment_size:
            raise ValueError("segments required will be longer than data")
        
        self.cuts = None
        
    def generate(self):
        start = 0
        cuts = [start]
        
        set_aside = self.segments * self.segment_size
        take_from = self.T - set_aside
        random_placements = np.random.choice(range(take_from), self.segments - 1, replace = True)
        random_placements = np.sort(random_placements)
        
        before = 0
        for k in range(self.segments - 1):
            p = random_placements[k]
            cut = cuts[-1] + self.segment_size + (p - before)
            cuts.append(cut)
            before = p
            
        cuts.append(self.T)
        return cuts
        
    
    
        
           
            
    