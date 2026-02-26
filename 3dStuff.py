import sys
import numpy as np
from vispy import app, scene
from vispy.scene import visuals

canvas = scene.SceneCanvas(keys='interactive', size=(800, 800), show=True)
view = canvas.central_widget.add_view()
global pos
pos = np.random.normal(size=(100, 3), scale=0.2)
scatter = visuals.Markers(pos=pos, face_color='w', size=10, edge_width=0)
view.add(scatter)
view.camera = 'turntable'

def update(ev):
    # Generate new, updated positions
    new_pos = np.random.normal(size=(100, 3), scale=0.2)
    # 5. Efficiently set new data
    scatter.set_data(new_pos)
    # Update global reference for next frame
    #global pos
    pos = new_pos

# 3. Timer
timer = app.Timer(interval=.1, connect=update, start=True)

if __name__ == '__main__':
    app.run()
