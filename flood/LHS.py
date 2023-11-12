# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 11:01:13 2023

@author: ATANIM
"""

import numpy as np
from pyDOE import lhs

#%%
def latin_hypercube_sampling_nd(num_samples, dimension_ranges):
    num_dimensions = len(dimension_ranges)
    
    # Generate Latin hypercube samples
    lhs_samples = lhs(num_dimensions, samples=num_samples, criterion='maximin')

    # Scale the samples to the desired range for each dimension
    scaled_samples = np.zeros_like(lhs_samples)

    for i in range(num_dimensions):
        range_min, range_max = dimension_ranges[i]
        scaled_samples[:, i] = lhs_samples[:, i] * (range_max - range_min) + range_min

    return scaled_samples

# Example usage for 4-dimensional space with different ranges
num_samples = 10  # Number of samples
dimension_ranges = [(0, 1), (5, 10), (-1, 1), (2, 3)]  # Ranges for each dimension

samples_nd = latin_hypercube_sampling_nd(num_samples, dimension_ranges)
print(f"Latin Hypercube Samples ({len(dimension_ranges)}D):")
print(samples_nd)