import numpy as np
import plotly.graph_objects as go
from numpy import *

def cylinder(r, h, a =0, nt=100, nv =50):
    """
    parametrize the cylinder of radius r, height h, base point a
    """
    theta = np.linspace(0, 2*np.pi, nt)
    v = np.linspace(a, a+h, nv )
    theta, v = np.meshgrid(theta, v)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = v
    return x, y, z

def boundary_circle(r, h, nt=100):
    """
    r - boundary circle radius
    h - height above xOy-plane where the circle is included
    returns the circle parameterization
    """
    theta = np.linspace(0, 2*np.pi, nt)
    x= r*np.cos(theta)
    y = r*np.sin(theta)
    z = h*np.ones(theta.shape)
    return x, y, z
r1 = 2
a1 = 0
h1 = 5


x1, y1, z1 = cylinder(r1, h1, a=a1)

colorscale = [[0, 'blue'],
             [1, 'blue']]

cyl1 = go.Surface(x=x1, y=y1, z=z1,
                 colorscale = colorscale,
                 showscale=False,
                 opacity=0.5)
xb_low, yb_low, zb_low = boundary_circle(r1, h=a1)
xb_up, yb_up, zb_up = boundary_circle(r1, h=a1+h1)

bcircles1 =go.Scatter3d(x = xb_low.tolist()+[None]+xb_up.tolist(),
                        y = yb_low.tolist()+[None]+yb_up.tolist(),
                        z = zb_low.tolist()+[None]+zb_up.tolist(),
                        mode ='lines',
                        line = dict(color='blue', width=2),
                        opacity =0.55, showlegend=False)


layout = go.Layout(scene_xaxis_visible=False, scene_yaxis_visible=False, scene_zaxis_visible=False)
fig =  go.Figure(data=[cyl1, bcircles1], layout=layout)

fig.update_layout(scene_camera_eye_z= 0.55)
fig.layout.scene.camera.projection.type = "orthographic" #commenting this line you get a fig with perspective proj

fig.show()