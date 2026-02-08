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
        self.radii = []
        
        for body in bodies:
            new_body = Body(body.mass, body.radius, body.pos.copy(), body.vel.copy())
            self.bodyList.append(new_body)
            self.radii.append(new_body.radius)
                
        for i, j in collisionCheck(self.bodyList, self.radii):
            mergeBodies(self.bodyList[i], self.bodyList[j], self.bodyList)
            
        for b in self.bodyList:
            forceSum = ForceSum()
            forces = self.bodyList.copy()
            forces.remove(b)
            for f in forces:
                new_force = Force(b, f)
                forceSum.addForce(new_force)
            b.moveBody(forceSum)
            self.coordsList.append(b.pos)

def collisionCheck(bodies, radii):
    positions = np.array([b.pos for b in bodies])
    N = positions.shape[0]
    collisions = []
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(positions[j] - positions[i])
            if dist < radii[i] + radii[j]:
                collisions.append((i, j))
    return collisions
                
def mergeBodies(body1, body2, list):
    mergedMass = body1.mass + body2.mass
    mergedRadius = (body1.radius**3 + body2.radius**3)**(1/3)
    mergedVelocity = (body1.mass * body1.vel + body2.mass * body2.vel)/mergedMass
    mergedBody = Body(mergedMass, mergedRadius, body1.pos, mergedVelocity)
    list.remove(body1)
    list.remove(body2)
    list.append(mergedBody)

def frameList(bodies, num_frames):
    frames = []
    current_bodies = bodies
    for i in range(num_frames):
        frame = Frame(current_bodies, i)
        frames.append(frame)
        current_bodies = frame.bodyList
    return frames

dt = 86400
eps = 1e6
radScale = 1e-6
sun = Body(g.sunMass, g.sunRadius, [0, 0])
a = Body(g.earthMass, g.earthRadius, [g.earthSunDist, 0], [0, g.earthVel])
b = Body(g.moonMass, g.moonRadius * 10, [g.moonEarthDist + g.earthSunDist, 0], [0, 1022 + g.earthVel])
bodies = [a, b, sun]

fig, ax = plt.subplots()
ax.set_xlim(-2e11, 2e11)
ax.set_ylim(-2e11, 2e11)

frames = frameList(bodies, 1000)

artists = []
for frame in frames:
    coords = np.array(frame.coordsList)
    sizes = np.array([body.radius * radScale for body in frame.bodyList])
    scatter = ax.scatter(coords[:, 0], coords[:, 1], s=sizes, color='black')
    artists.append([scatter])


ani = animation.ArtistAnimation(
    fig,
    artists,
    interval = 100,
    blit = True
)

plt.show()

#Things to add next:
# - Velocity Verlet Algorithm
# - THEN make a 3d version
#    - VPython?
# - GPU utilization
