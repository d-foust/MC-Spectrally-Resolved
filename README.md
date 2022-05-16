# MC-Spectrally-Resolved
Monte Carlo simulations of spectrally resolved images in fluorescence microscopy for analysis with fluorescence fluctuation spectroscopy. Images are of a uniform distribution of fluorescent particles in two dimensions.

# 0. Load packages


```python
import numpy as np
from random import randrange
from numpy.random import *
import matplotlib.pyplot as plt
import ipympl
%matplotlib ipympl

from scipy.ndimage import gaussian_filter
from os.path import join
import pandas as pd
import seaborn as sns
```

# 1. Define some parameters


```python
# load detection spectra for mEGFP, mEYFP(Q69K), and mCherry2 determined experimentally
data_spectra = pd.read_csv('emission spectra.csv') 
```


```python
# normalize detection spectra to get probability densities
p_G = data_spectra['mEGFP'] / data_spectra['mEGFP'].sum()
p_Y = data_spectra['mEYFP'] / data_spectra['mEYFP'].sum()
p_R = data_spectra['mCherry2'] / data_spectra['mCherry2'].sum()
```


```python
# radius determines size of detection area
radius = 4
detection_area = (np.pi * radius**2) * 2
```


```python
image_size = 276 # size of image (x,y) in pixels
num_frames = 1 # frames to be simulated
num_density = 5 # average number of particles per detection area
brightness = 0.5 # photons per molecules per pixel
total_area = image_size**2 
num_detection_areas = total_area / detection_area
num_particles = int(np.round(num_density * num_detection_areas))
```

# 2. Simulate positions of particles


```python
image_particles = np.zeros([num_frames, image_size, image_size])
```


```python
for frame in range(num_frames):
        locs = randint(low=0, high=image_size-1, size=(num_particles,2)) # particles locations
        image_particles[frame] = np.histogram2d(locs[:,0], locs[:,1], 
                                                bins=[range(image_size+1), 
                                                      range(image_size+1)])[0] # generate particle map
```


```python
image_particles.shape
```




    (1, 276, 276)




```python
plt.figure()
plt.imshow(image_particles[0], interpolation='none', cmap='binary_r')
plt.tight_layout()
plt.axis('off')
plt.colorbar();
```

<img src="example%20images/image_particles.png" width="500">


# 3. Convolution with model PSF (i.e. Gaussian blur)


```python
image_gaussian = gaussian_filter(image_particles, sigma=(0,radius,radius))
```


```python
plt.figure()
plt.imshow(image_gaussian[0], interpolation='none', cmap='binary_r')
plt.tight_layout()
plt.axis('off')
plt.colorbar();
```


<img src="example%20images/image_gaussian.png" width="500">


# 4. Scale for molecular brightness


```python
image_scale = np.zeros([51,51])
image_scale[25,25] = 1
image_scale = gaussian_filter(image_scale, sigma=4)
scale_factor = image_scale.max()

image_brightness = (image_gaussian / scale_factor) * brightness
```


```python
plt.figure()
plt.imshow(image_brightness[0], interpolation='none', cmap='binary_r')
plt.tight_layout()
plt.axis('off')
plt.colorbar();
```


<img src="example%20images/image_colorized.png" width="500">


# 5. Poisson filter to capture stochastic nature of photon emission/detection


```python
image_poisson = poisson(lam=image_brightness)
```


```python
plt.figure()
plt.imshow(image_poisson[0], interpolation='none', cmap='binary_r')
plt.tight_layout()
plt.axis('off')
plt.colorbar();
```


<img src="example%20images/spectracomparison.png" width="500">


# 6. Split into individual channels


```python
p_G = data_spectra['mEGFP'] / data_spectra['mEGFP'].sum()
pvals = p_G
```


```python
image_colorized = np.zeros([image_poisson.shape[0],
                                len(pvals), 
                                image_poisson.shape[1], 
                                image_poisson.shape[2]])

for (t,x,y), val in np.ndenumerate(image_poisson):
        image_colorized[t,:,x,y] = multinomial(val, pvals=pvals)
```


```python
image_colorized.shape
```




    (1, 23, 276, 276)




```python
clim = (image_colorized[0].min(), np.percentile(image_colorized[0], 99))

fig, axs = plt.subplots(5,5, figsize=(9,9))

for i, ax in enumerate(axs.ravel()[:len(pvals)]):
    ax.imshow(image_colorized[0,i], interpolation='none', cmap='binary_r', clim=clim)
    ax.set_axis_off()
    ax.title.set_text(str(data_spectra['Channel'][i])+' nm')
plt.tight_layout(pad=0)
axs[4,3].set_axis_off(); axs[4,4].set_axis_off()
```


<img src="example%20images/spectracomparison.png" width="500">


## Compare spectra


```python
p_simulated = image_colorized[0].sum(axis=(1,2))
p_simulated /= p_simulated.sum() # normalization
```


```python
plt.figure()
plt.plot(pvals, 'r', label='Actual')
plt.plot(p_simulated, 'kx', label='Simulated')
plt.xticks(range(0,23,3), data_spectra['Channel'][0::3], fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('Intensity (AU)', fontsize=14)
plt.xlabel('Detection Channel (nm)', fontsize=14)
sns.despine()
plt.legend(fontsize=12)
plt.tight_layout();
```


<img src="example%20images/image_particles.png" width="500">


# Functions


```python
def sim_image(num_density=10, size=100, radius=4, brightness=0.1, 
              num_frames=1, pvals=[1]):
    """
    
    """
    detection_area = (np.pi * radius**2) * 2
    total_area = size**2
    num_detection_areas = total_area / detection_area
    num_particles = int(np.round(num_density * num_detection_areas))
    
    image_particles = np.zeros([num_frames, size, size])
    
    for frame in range(num_frames):
        locs = randint(low=0, high=size-1, size=(num_particles,2)) # particles locations
        image_particles[frame] = np.histogram2d(locs[:,0], locs[:,1], 
                                                bins=[range(size+1), range(size+1)])[0] # generate particle map
        
    image_gaussian = gaussian_filter(image_particles, sigma=(0,radius,radius))
    
    image_scale = np.zeros([51,51])
    image_scale[25,25] = 1
    image_scale = gaussian_filter(image_scale, sigma=4)
    scale_factor = image_scale.max()

    image_gaussian = (image_gaussian / scale_factor) * brightness
    
    image_poisson = poisson(lam=image_gaussian)
    
    image_simulated = multinomial_colorization(image_poisson, pvals=pvals)
    
    return image_simulated

def multinomial_colorization(image, pvals):
    
    image_colorized = np.zeros([image.shape[0],
                                len(pvals), 
                                image.shape[1], 
                                image.shape[2]])
     
    for (t,x,y), val in np.ndenumerate(image):
        image_colorized[t,:,x,y] = multinomial(val, pvals=pvals)
    
    return image_colorized
```
