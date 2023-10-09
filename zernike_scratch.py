import numpy as np
import matplotlib.pyplot as plt

def zernike_radial(rho, n, m):
    radial_term = np.zeros_like(rho, dtype=float)
    for s in range((n - abs(m)) // 2 + 1):
        coef = (-1) ** s * np.math.factorial(n - s)
        coef /= (
            np.math.factorial(s)
            * np.math.factorial(int((n + abs(m)) / 2) - s)
            * np.math.factorial(int((n - abs(m)) / 2) - s)
        )
        radial_term += coef * rho ** (n - 2 * s)
    return radial_term

def zernike_normalization(n, m):
    # Calculate the normalization factor for Zernike polynomials
    norm_factor = np.sqrt((2 * (n + 1)) / (1 + (m == 0)))
    return norm_factor

def zernike_polynomial(size, n, m):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    rho = np.sqrt(X**2 + Y**2)
    
    # Create a circular mask with a slightly larger radius to include 34 elements
    mask = rho <= 1.06  # Adjust the radius as needed
    
    # Calculate the radial component of the Zernike polynomial
    radial_component = zernike_radial(rho, n, m)
    
    if m == 0:
        azimuthal_component = np.ones_like(rho)
    elif m > 0:
        azimuthal_component = np.cos(m * np.arctan2(Y, X))
    else:
        azimuthal_component = np.sin(-m * np.arctan2(Y, X))
    
    # Combine the radial and azimuthal components to get the Zernike polynomial
    zernike = radial_component * azimuthal_component
    
    # Apply the circular mask
    zernike[~mask] = 0.0

    # Normalize the Zernike polynomial
    norm_factor = zernike_normalization(n, m)
    zernike /= norm_factor
    
    return zernike

def generate_zernike_images(size, nm_indices):
    images = []
    for pair in nm_indices:
        n = pair[0]
        m = pair[1]
        zernike_poly = zernike_polynomial(size, n, m)
        images.append(zernike_poly)
    return images

size = 34
num_polynomials = 15  # Change this to the number of Zernike polynomials you want
nm_pairs = [(0,0),(1,1),(1,-1),(2,0),(2,2),(2,-2),(3,1),(3,-1),
            (4,0),(3,3),(3,-3),(4,2),(4,-2),(5,1),(5,-1),(6,0)]
rand_coeffs = np.random.rand(len(nm_pairs))
print(rand_coeffs)
zernike_images = generate_zernike_images(size, nm_pairs)

scaled_zernike_list = [image * scale for image, scale in zip(zernike_images, rand_coeffs)]

surface_random = np.sum(scaled_zernike_list, axis=0)


# Plot the Zernike polynomial images in a grid
fig, axes = plt.subplots(1, num_polynomials, figsize=(15, 3))
for i in range(num_polynomials):
    n = i
    m = -n if i % 2 == 0 else -(n-1)
    axes[i].imshow(zernike_images[i], extent=(-1, 1, -1, 1), cmap='viridis', origin='lower')
    axes[i].set_title(f'Zernike (n={n}, m={m})')
    axes[i].axis('off')
# plt.imshow(surface_random,origin='lower')
plt.show()