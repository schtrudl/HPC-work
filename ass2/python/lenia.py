#
# Lenia cellular automaton
# 
# the code is simplifeid version of codes published at:
# - https://github.com/scienceetonnante/lenia/blob/main/GameOfLife_Lenia.ipynb
# - https://colab.research.google.com/github/OpenLenia/Lenia-Tutorial/blob/main/Tutorial_From_Conway_to_Lenia.ipynb 

# packages
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Gaussian function
def gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x-mu)/sigma)**2)

# ring convolution filter construction
def kernel_lenia(R, mu, sigma):
    y, x = np.ogrid[-R:R, -R:R]
    dist = np.sqrt((1+x)**2 + (1+y)**2) / R
    K = gauss(dist, mu, sigma)
    K[dist > 1] = 0
    K = K / np.sum(K)
    return K

# growth criteria
def growth_lenia(C, mu, sigma):
    return -1 + 2 * gauss(C, mu, sigma)        # Baseline -1, peak +1

# Lenia iteration t -> t+dt
def evolve_lenia(B, K, mu, sigma, dt):  
    # store result of convolution to a new matrix 
    C = sp.signal.convolve2d(B, K, mode='same', boundary='wrap') 
    # update the board 
    B = B + dt * growth_lenia(C, mu, sigma)
    # cell values should remain in interval [0, 1]
    B = np.clip(B, 0, 1)
    return B

# Orbium creature
orbium = np.array([[0,0,0,0,0,0,0.1,0.14,0.1,0,0,0.03,0.03,0,0,0.3,0,0,0,0], #
                   [0,0,0,0,0,0.08,0.24,0.3,0.3,0.18,0.14,0.15,0.16,0.15,0.09,0.2,0,0,0,0], #
                   [0,0,0,0,0,0.15,0.34,0.44,0.46,0.38,0.18,0.14,0.11,0.13,0.19,0.18,0.45,0,0,0], #
                   [0,0,0,0,0.06,0.13,0.39,0.5,0.5,0.37,0.06,0,0,0,0.02,0.16,0.68,0,0,0], #
                   [0,0,0,0.11,0.17,0.17,0.33,0.4,0.38,0.28,0.14,0,0,0,0,0,0.18,0.42,0,0], #
                   [0,0,0.09,0.18,0.13,0.06,0.08,0.26,0.32,0.32,0.27,0,0,0,0,0,0,0.82,0,0], #
                   [0.27,0,0.16,0.12,0,0,0,0.25,0.38,0.44,0.45,0.34,0,0,0,0,0,0.22,0.17,0], #
                   [0,0.07,0.2,0.02,0,0,0,0.31,0.48,0.57,0.6,0.57,0,0,0,0,0,0,0.49,0], #
                   [0,0.59,0.19,0,0,0,0,0.2,0.57,0.69,0.76,0.76,0.49,0,0,0,0,0,0.36,0], #
                   [0,0.58,0.19,0,0,0,0,0,0.67,0.83,0.9,0.92,0.87,0.12,0,0,0,0,0.22,0.07], #
                   [0,0,0.46,0,0,0,0,0,0.7,0.93,1,1,1,0.61,0,0,0,0,0.18,0.11], #
                   [0,0,0.82,0,0,0,0,0,0.47,1,1,0.98,1,0.96,0.27,0,0,0,0.19,0.1], #
                   [0,0,0.46,0,0,0,0,0,0.25,1,1,0.84,0.92,0.97,0.54,0.14,0.04,0.1,0.21,0.05], #
                   [0,0,0,0.4,0,0,0,0,0.09,0.8,1,0.82,0.8,0.85,0.63,0.31,0.18,0.19,0.2,0.01], #
                   [0,0,0,0.36,0.1,0,0,0,0.05,0.54,0.86,0.79,0.74,0.72,0.6,0.39,0.28,0.24,0.13,0], #
                   [0,0,0,0.01,0.3,0.07,0,0,0.08,0.36,0.64,0.7,0.64,0.6,0.51,0.39,0.29,0.19,0.04,0], # 
                   [0,0,0,0,0.1,0.24,0.14,0.1,0.15,0.29,0.45,0.53,0.52,0.46,0.4,0.31,0.21,0.08,0,0], #
                   [0,0,0,0,0,0.08,0.21,0.21,0.22,0.29,0.36,0.39,0.37,0.33,0.26,0.18,0.09,0,0,0], #
                   [0,0,0,0,0,0,0.03,0.13,0.19,0.22,0.24,0.24,0.23,0.18,0.13,0.05,0,0,0,0], #
                   [0,0,0,0,0,0,0,0,0.02,0.06,0.08,0.09,0.07,0.05,0.01,0,0,0,0,0]])

# simulation parameters
N = 128             # board size
steps = 100         # number of simulation steps
dt = 0.1            # sampling time
RKernel = 13        # kernel size
muKernel = 0.5      # kernel mean value
sigmaKernel = 0.15  # kernel standard deviation
muGrowth = 0.15     # growth mean value
sigmaGrowth = 0.015 # growth standard deviation

kernel = kernel_lenia(RKernel, muKernel, sigmaKernel)

# plot kernel and creature
plt.subplot(121)
plt.imshow(kernel, interpolation='none', cmap='inferno')
plt.title('Convolution filter')
plt.subplot(122)
plt.imshow(orbium, cmap='inferno',interpolation='bicubic',vmin=0,vmax=1)
plt.title('Orbium')

# plot growth function
a = np.arange(0, 1, 0.001)
plt.figure()
plt.plot(a, growth_lenia(a, muGrowth, sigmaGrowth))
plt.axhline(0, linestyle='--', color='red')
plt.title('Growth function')

# initial board with two creatures
# allocate empty board
board = np.zeros((N, N))
# put creatures to it
posY = 0
posX = N//3
board[posY:(posY + orbium.shape[0]), posX:(posX + orbium.shape[1])] = orbium 
posY = N//3
posX = 0
board[posY:(posY + orbium.shape[0]), posX:(posX + orbium.shape[1])] = orbium.T

# animation initialization
fig = plt.figure()
im = plt.imshow(board, cmap='inferno', interpolation='bicubic', vmin=0, vmax=1)
plt.axis('off')

# animation update step
def update(i):
    if((i+1)%(steps//10)==0):
        print('Step {}/{}'.format(i+1, steps))         
    global board
    board = evolve_lenia(board, kernel, muGrowth, sigmaGrowth, dt)
    im.set_array(board)
    return im,

# animation activation
ani = animation.FuncAnimation(fig, update, steps, interval=10, blit=True)
plt.show()
