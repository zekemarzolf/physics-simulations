import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
from Constants import Gravity as g, FrameRates as FR

class Body:
    
    def __init__(self, m, r, position, velocity=[0, 0]):
        self.mass = m
        self.radius = r
        self.pos = np.array(position, dtype=float)
        self.vel = np.array(velocity, dtype=float)
    
    def moveBody(self, forceSum):
        acc = np.divide(forceSum.force, self.mass)
        self.pos += self.vel * dt + 0.5 * acc * (dt**2)
        self.vel += acc * dt

class Force:
    
    def __init__(self, body1, body2):
        d = np.subtract(body2.pos, body1.pos)
        dist2 = np.sqrt(np.sum(d**2)) + eps
        magnitude = (g.G * body1.mass * body2.mass)/(dist2**2)
        self.force = magnitude * d / dist2
        

class ForceSum:
    
    def __init__(self):
        self.force = [0, 0]
    
    def addForce(self, force):
        self.force = np.add(self.force, force.force)

class Frame:
    
    def __init__(self, bodies, frameNum):
        self.order = frameNum
        self.bodyList = []
        self.coordsList = []
        
        for body in bodies:
            new_body = Body(body.mass, body.radius, body.pos.copy(), body.vel.copy())
            self.bodyList.append(new_body)
            
        for b in self.bodyList:
            forceSum = ForceSum()
            forces = self.bodyList.copy()
            forces.remove(b)
            for f in forces:
                new_force = Force(b, f)
                forceSum.addForce(new_force)
            b.moveBody(forceSum)
            self.coordsList.append(b.pos)
                

def frameList(bodies, num_frames):
    frames = []
    current_bodies = bodies
    for i in range(num_frames):
        frame = Frame(current_bodies, i)
        frames.append(frame)
        current_bodies = frame.bodyList
    return frames

dt = 3600
eps = 1e3
a = Body(g.earthMass, 40, [0, 0])
b = Body(g.moonMass, 10, [g.moonDist, 0], [0, 0])
bodies = [a, b]

fig, ax = plt.subplots()
ax.set_xlim(-500000000, 500000000)
ax.set_ylim(-500000000, 500000000)

frames = frameList(bodies, 1000)

artists = []
for frame in frames:
    coords = np.array(frame.coordsList)
    sizes = np.array([body.radius for body in frame.bodyList])
    scatter = ax.scatter(coords[:, 0], coords[:, 1], s=sizes, color='black')
    artists.append([scatter])


ani = animation.ArtistAnimation(
    fig,
    artists,
    interval = FR.int24,
    blit = True
)

plt.show()

#Things to add next:
# - Velocity Verlet Algorithm
# - Merging Logic
# - THEN make a 3d version
#    - VPython?
# - GPU utilization
