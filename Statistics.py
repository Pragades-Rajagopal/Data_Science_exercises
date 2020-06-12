#%%
import statistics as stat
import numpy as np
import math

a = np.arange(10, 20, 1)
print (a)

m = stat.mean(a)
print (m)

med = stat.median(a)
print (med)

mod = (1, 3, 4, 5, 7, 9, 2, 2)
print (stat.mode(mod))

var = stat.variance(a)
print (var)

sd = math.sqrt(var)
print (sd)



# %%
