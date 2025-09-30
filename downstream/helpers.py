import numpy as np
from scipy import stats

def ci_bounds(data, confidence=0.95):
    """
    Compute the confidence interval bounds for a given data array.
    
    Parameters:
        data (array-like): Input data.
        confidence (float): Confidence level, default is 0.95.
    
    Returns:
        lower_bound (float): Lower bound of the confidence interval.
        upper_bound (float): Upper bound of the confidence interval.
    """
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    margin = sem * stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean - margin, mean + margin 
