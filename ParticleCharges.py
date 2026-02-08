import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import math

class Particle:
    
    def __init__(self, charge, x, y):
        self.charge = charge
        self.x = x
        self.y = y
        self.xVel = 0
        self.yVel = 0
    
    def updateCoords(self, newX, newY):
        self.x = round(newX, 2)
        self.y = round(newY, 2)
    
    def accelerate(self, a):
        self.xVel += a.xAcc
        self.yVel += a.yAcc
    
    def printInfo(self):
        print(self.x, ", ", self.y, " ", " vel = ", round(self.xVel, 2), ", ", round(self.yVel, 2))
    

class Force:
    
    def __init__(self, part1, part2):
        xdiff = part2.x - part1.x
        ydiff = part2.y - part1.y
        
        #magnitude calculation
        d = math.sqrt(xdiff ** 2 + ydiff ** 2)
        self.magnitude = (9e9 * part1.charge * part2.charge)/(d**2 + .01)
        
        #separate into x and y
        ratio = self.magnitude/d
        self.xAcc = round(ratio * xdiff * 10, 4)
        self.yAcc = round(ratio * ydiff * 10, 4)
        
        #x direction
        if (self.magnitude >= 0):
            if (xdiff >= 0):
                self.xAcc *= -1
        else:
            #if (xdiff < 0):
            self.xAcc *= -1
        #y direction
        if (self.magnitude >= 0):
            if (ydiff >= 0):
                self.yAcc *= -1
        else:
            #if (ydiff < 0):
            self.yAcc *= -1

class AccelerationSum:
    
    def __init__(self):
        self.xAcc = 0
        self.yAcc = 0
    
    def addForce(self, force):
        self.xAcc += force.xAcc
        self.yAcc += force.yAcc

class Frame:
    
    def __init__(self, particles, frameNum):
        self.order = frameNum
        self.partList = particles.copy()
        self.coordsList = []
        
        #frames after 0
        if (frameNum != 0):
            
            for p in self.partList:
                forces = self.partList.copy()
                forces.remove(p)
                self.sumOnP = AccelerationSum()
                
                for f in forces:
                    new_Force = Force(p, f)
                    self.sumOnP.addForce(new_Force)
                
                moveParticle(p, self.sumOnP)
                
            for p in self.partList:
                self.coordsList.append((p.x, p.y))
        
        #initial frame
        if (frameNum == 0):
            for p in self.partList:
                self.coordsList.append((p.x, p.y))

def moveParticle(part, accSum):
    xDist = round(part.xVel + (accSum.xAcc/2), 4)
    yDist = round(part.yVel + (accSum.yAcc/2), 4)
    part.updateCoords(part.x + xDist, part.y + yDist)
    part.accelerate(accSum)

def frameList(particles, num_frames):
    frames = []
    current_particles = particles
    for i in range(num_frames):
        frame = Frame(current_particles, i)
        frames.append(frame)
        current_particles = frame.partList
    return frames

a = Particle(5e-6, 4, -7)
b = Particle(-4e-6, 7, 6)
c = Particle(3e-6, -6, 8)
d = Particle(-5e-6, -4, 20)
e = Particle(-5e-5, -20, -35)
f = Particle(4e-8, 0, 0)
particles = [a, b, c, d, e, f]

fig, ax = plt.subplots()
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)

frames = frameList(particles, 200)

artists = []
for frame in frames:
    coords = np.array(frame.coordsList)
    scatter = ax.scatter(coords[:, 0], coords[:, 1], s=100)
    artists.append([scatter])


ani = animation.ArtistAnimation(
    fig,
    artists,
    interval = 50,
    blit = True
)

plt.show()




