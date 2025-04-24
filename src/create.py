import numpy as np
from scipy.stats import truncnorm

N = 4000
coords_xy = np.random.uniform(146, 594, (N,2))
rot_z = np.random.normal(0, 0.035, (N,1))

cams = np.hstack((np.zeros((coords_xy.shape[0], 2), dtype=int), rot_z, coords_xy, np.full((coords_xy.shape[0], 1), 352)))
locations = cams[0, 3:6]

np.save(r"\cams1000.npy", cams)
np.save(r"\locations4000.npy", np.array(locations))

print("Data Saved!")