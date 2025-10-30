# Record variance of a set of measurements and generate a plot

import numpy as np
import matplotlib.pyplot as plt

# numbers 
'''
148.1 (initial)
148.1
148.2
148.1
148.2
148.3
148.1
148.2
148.1
148.0
148.2
'''

measurements = [148.1, 148.1, 148.2, 148.1, 148.2, 148.3, 148.1, 148.2, 148.1, 148.0, 148.2]

# Calculate variance
variance = np.var(measurements)

# Generate plot
plt.hist(measurements, bins=4, edgecolor='black')
plt.title('Measurement Distribution Test 1')
plt.xlabel('Measurement')
plt.ylabel('Frequency')
plt.show()  

print(f'Variance of measurements: {variance}')


'''
second set of numbers:
66.4 (initial)
66.6
66.5
66.4
66.4
66.8
'''

measurements_2 = [66.4, 66.6, 66.5, 66.4, 66.4, 66.8]
# Calculate variance
variance_2 = np.var(measurements_2)
# Generate plot
plt.hist(measurements_2, bins=4, edgecolor='black')
plt.title('Measurement Distribution Test 2')
plt.xlabel('Measurement')
plt.ylabel('Frequency')
plt.show()

print(f'Variance of measurements: {variance_2}')