import numpy as np
import matplotlib.pyplot as plt
from zernike import RZern


cart = RZern(6)
L, K = 34, 34
ddx = np.linspace(-1.0, 1.0, K)
ddy = np.linspace(-1.0, 1.0, L)
xv, yv = np.meshgrid(ddx, ddy)
cart.make_cart_grid(xv, yv)
print(cart.nk)

num_polynomials = 10
matrix = np.empty((L*K, num_polynomials))

c = np.zeros(cart.nk)
plt.figure(1)
for i in range(1, 10):
    plt.subplot(3, 3, i)
    c *= 0.0
    c[i] = 1.0
    Phi = cart.eval_grid(c, matrix=True)
    flattened_zernike = Phi.flatten()
    matrix[:, i] = flattened_zernike
    # plt.imshow(Phi, origin='lower', extent=(-1, 1, -1, 1))
    # plt.axis('off')



plt.show()