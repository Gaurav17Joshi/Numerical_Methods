import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
Hyperparameters = {'W': 1, 'L': 5, 'N': 10**5, 'T': 10, 'D': 1, 'delt': float(1/50), 
                    'delx': float(2*(10**(-2))), 'dely': float(10**(-2)), 'strip': float(0.05)}

from diffusion import Unboxed_Diffusion, Boxed_Diffusion

DM = Unboxed_Diffusion(Hyperparameters)
BDM = Boxed_Diffusion(Hyperparameters)

DM.simulate()
BDM.simulate()

# Uncomment to plot the situation of particles at any given time {For Unboxed Diffusion}
# DM.plot_diff(time=20)

# Uncomment to plot the situation of particles at any given time {For Boxed Diffusion}
# BDM.plot_diff(time=10)

# Uncomment to plot the probability distribution of particles at any given time {For Unboxed Diffusion}
# DM.plot_prob_3d(time=2, convolve=False)

# Uncomment to plot the probability distribution of particles at any given time {For Unboxed Diffusion} {convolved}
# DM.plot_prob_3d(time=2, convolve=True)

# Uncomment to plot the probability distribution of particles at any given time {For Boxed Diffusion} {convolved}
# BDM.plot_prob_3d(time=2, convolve=True)

# Uncomment to plot the probability distribution of particles at any given time {For Boxed Diffusion} 
# BDM.plot_prob_3d(time=4, convolve=False)

# Uncomment to see the video of a single particle moving in the box {For Unboxed Diffusion}
DM.animated_path(5500, name = "plots/animated_path.gif")

# Uncomment to see the video of a single particle moving in the box {For Boxed Diffusion}
BDM.animated_path(5550, name = "plots/animated_boxed_path.gif")

# Plots the probability distribution of a point as a function of time {For Unboxed Diffusion}
xpoint = 3
ypoint = 0.5
print("Probability desity is: ",DM.get_prob_3d(xpoint = 0, ypoint = 0.5, time = 34, convolve = False)[0])

time = np.arange(0, 200, 1)
prob = np.zeros(200)
for i in range(200):
    prob[i] = BDM.get_prob_3d(xpoint = xpoint, ypoint = ypoint, time = i, convolve = True)[0]

plt.plot(time, prob)
plt.show()