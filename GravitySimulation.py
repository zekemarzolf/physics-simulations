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
        self.merged = False
    
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
                
        for i in collisionCheck(self.bodyList, self.radii):
            mergeBodies(i, self.bodyList)
        self.bodyList = [b for b in self.bodyList if not b.merged]
            
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
        if any(bodies[i] is b for l in collisions for b in l):
            continue
        bodiesMerged = [bodies[i]]
        for j in range(i + 1, N):
            dist = np.linalg.norm(positions[j] - positions[i])
            if dist < radii[i] + radii[j]:
                bodiesMerged.append(bodies[j])
        if len(bodiesMerged) == 1:
            continue
        collisions.append(bodiesMerged)
    return collisions
                
def mergeBodies(m, l):
    mergedMass = 0
    for b in m:
        mergedMass += b.mass
    
    mergedRadius = 0
    for b in m:
        mergedRadius += b.radius**3
    mergedRadius = mergedRadius**(1/3)
    
    mergedVelocity = np.array([0, 0], dtype=float)
    for b in m:
        mergedVelocity += b.mass*b.vel
    mergedVelocity /= mergedMass
    
    mergedPos = np.array([0, 0], dtype=float)
    for b in m:
        mergedPos += b.pos*b.mass
    mergedPos /= mergedMass
    
    mergedBody = Body(mergedMass, mergedRadius, mergedPos, mergedVelocity)
    for b in m:
        b.merged = True
    l.append(mergedBody)

def frameList(bodies, num_frames):
    frames = []
    current_bodies = bodies
    for i in range(num_frames):
        frame = Frame(current_bodies, i)
        frames.append(frame)
        current_bodies = frame.bodyList
    return frames

dt = .05
eps = 20
sizeScale = .1
axlim1 = 1000
axlim2 = axlim1 * -1

#sun = Body(g.sunMass, g.sunRadius, [0, 0])
#earth = Body(g.earthMass, g.earthRadius, [g.earthSunDist, 0], [0, g.earthVel])
#moon = Body(g.moonMass, g.moonRadius * 10, [g.moonEarthDist + g.earthSunDist, 0], [0, 1022 + g.earthVel])
a = Body(1e17, 2, [-20, 0], [0, -5])
j = Body(1e17, 2, [20, 0], [0, 5])
b = Body(1e14, 5, [500, 0], [0, 100])
c = Body(1e14, 5, [-500, 0], [0, -100])
d = Body(1e14, 5, [0, 500], [-100, 0])
e = Body(1e14, 5, [0, -500], [100, 0])
f = Body(1e14, 5, [100, 0], [0, 200])
g1 = Body(1e14, 5, [-100, 0], [0, -200])
h = Body(1e14, 5, [0, 100], [-200, 0])
i = Body(1e14, 5, [0, -100], [200, 0])
bodies = [a, b, c, d, e, f, g1, h, i, j]

fig, ax = plt.subplots()
ax.set_xlim(axlim2, axlim1)
ax.set_ylim(axlim2, axlim1)

frames = frameList(bodies, 5000)

artists = []
for frame in frames:
    coords = np.array(frame.coordsList)
    sizes = np.array([(body.radius**2)*6.28*sizeScale for body in frame.bodyList])
    scatter = ax.scatter(coords[:, 0], coords[:, 1], s=sizes, color='black')
    artists.append([scatter])

ani = animation.ArtistAnimation(
    fig,
    artists,
    interval = 5,
    blit = True
)

plt.show()

#Things to add next:
# - Smarter Collision Detections
# - THEN make a 3d version
#    - VisPy
#    - PyVista
# - GPU utilization
