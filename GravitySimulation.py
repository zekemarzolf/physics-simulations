import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math

class Mass:
    
    def __init__(self, m, r, x, y, xVel=0, yVel=0):
        self.mass = m
        self.radius = r
        self.x = x
        self.y = y
        self.xVel = xVel
        self.yVel = yVel
    
    def moveMass(self, forceSum):
        aX = forceSum.xAcc / self.mass
        aY = forceSum.yAcc / self.mass
        self.x += self.xVel * dt + 0.5* aX * (dt**2)
        self.y += self.yVel * dt + 0.5* aY * (dt**2)
        self.xVel += aX * dt
        self.yVel += aY * dt

class Force:
    
    def __init__(self, mass1, mass2):
        xdiff = mass2.x - mass1.x
        ydiff = mass2.y - mass1.y
        r = math.sqrt(xdiff ** 2 + ydiff ** 2)
        self.magnitude = (6.6743e-11 * mass1.mass * mass2.mass)/(r**2)
        ratio = self.magnitude/r
        self.xAcc = ratio * xdiff
        self.yAcc = ratio * ydiff

class ForceSum:
    
    def __init__(self):
        self.xAcc = 0
        self.yAcc = 0
    
    def addForce(self, force):
        self.xAcc += force.xAcc
        self.yAcc += force.yAcc

class Frame:
    
    def __init__(self, masses, frameNum):
        self.forceNum = len(masses) - 1
        self.order = frameNum
        self.massList = masses.copy()
        self.coordsList = []
        
        if (frameNum != 0):
            
            for p in self.massList:
                forces = self.massList.copy()
                forces.remove(p)
                self.sumOnP = ForceSum()
                
                for f in forces:
                    new_Force = Force(p, f)
                    self.sumOnP.addForce(new_Force)
                
                p.moveMass(self.sumOnP)
                
            for p in self.massList:
                self.coordsList.append((p.x, p.y))
        
        if (frameNum == 0):
            for p in self.massList:
                self.coordsList.append((p.x, p.y))

def frameList(masses, num_frames):
    frames = []
    current_masses = masses
    for i in range(num_frames):
        frame = Frame(current_masses, i)
        frames.append(frame)
        current_masses = frame.massList
    return frames

dt = 3600
a = Mass(5.972e24, 40, 0, 0, 0, 0)
b = Mass(7.34767e22, 10, 384400000, 0, 0, 1022)
masses = [a, b]

fig, ax = plt.subplots()
ax.set_xlim(-500000000, 500000000)
ax.set_ylim(-500000000, 500000000)

frames = frameList(masses, 1000)

artists = []
for frame in frames:
    coords = np.array(frame.coordsList)
    sizes = np.array([mass.radius for mass in frame.massList])
    scatter = ax.scatter(coords[:, 0], coords[:, 1], s=sizes, color='black')
    artists.append([scatter])


ani = animation.ArtistAnimation(
    fig,
    artists,
    interval = 42,
    blit = True
)

plt.show()

#Things to add next:
# - Merging Logic
# - Vectors with direction for mechanics calculations
# - Velocity Verlet Algorithm
# - THEN make a 3d version
