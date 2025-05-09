import numpy as np
import matplotlib.pyplot as plt

my_dpi = 96
c_id = 0
# path = '0.005-z-800-test.npy'

path = "data/fragments/lego_pointnerf/0.005-z-800-1-test.npy"


zbuf = np.load(path)
n = zbuf[c_id, :, :, :]
h, w, channels = n.shape

plt.figure(figsize=(h/my_dpi, w/my_dpi), dpi=my_dpi)
plt.imshow(n)
plt.axis('off')
plt.margins(0,0)
plt.savefig(str(c_id)+".png", bbox_inches="tight", pad_inches=0.0)
plt.show()