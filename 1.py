
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import PIL.Image
import time
import torch

image = PIL.Image.open("image.png")
image = image.resize((image.width // 4, image.height // 4), PIL.Image.Resampling.NEAREST)
image = image.convert("1")
data = 1*np.array(image)

device = "cpu"
data = torch.tensor(data, device=device)

def run(points, data, constraints, rate=5e-3):
    for time_slice in data:
        distances_cohort = torch.pdist(points[time_slice == 1])
        norms_cohort = torch.linalg.norm(points[time_slice == 1], dim=-1)
        distances_complement = torch.pdist(points[time_slice == 0])
        norms_complement = torch.linalg.norm(points[time_slice == 0], dim=-1)

        center_of_mass_1 = points[time_slice == 1].mean()
        norms_cohort_centers = torch.linalg.norm(points[time_slice == 1] - center_of_mass_1, dim=-1)

        loss = 0
        if len(distances_cohort) > 0:
            loss += ((0.1 - distances_cohort)**2).mean()
            loss += 0.5*((0.1 - norms_cohort)**2).mean()
            loss += 5*((0.1 - norms_cohort_centers)**2).mean()

        if len(norms_complement) > 0:
            loss += 50*((2 - distances_complement)**2).mean()
            loss += 5*((2 - norms_complement)**2).mean()

        if loss > 0:
            loss.backward()

    with torch.no_grad():
        points -= rate*points.grad
        points.grad.zero_()

    points_history.append(points.cpu().clone().detach().numpy())


points = torch.rand(data.shape[0], 2,
                    device=device,
                    requires_grad=True)

points = 2*np.array([[np.exp(1j*r).real, np.exp(1j*r).imag]
                     for r in 2*np.pi * np.arange(len(points)) / len(points)])

points = torch.tensor(points, dtype=torch.float, device=device, requires_grad=True)
points_history = [points.cpu().clone().detach().numpy()]

section_length = 5
step_size = 2

t1 = time.time()
for section_index in list(range(0, len(data.T) - section_length, step_size)):
    for _ in range(data.T[section_index:section_index+section_length].sum()*2):
        run(points, data=data.T[section_index:section_index+section_length], rate=5e-2, constraints=[])

print({"time": time.time() - t1})

print({"frame_count": len(points_history)})
h = np.array(points_history)
h = np.vstack(h)
(xmin, ymin) = h.min(axis=0)
(xmax, ymax) = h.max(axis=0)

# plt.style.use("dark_background")
fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

plotted_points = ax.plot(*points_history[0].T, "o", markersize=2)[0];

def update(index):
    plotted_points.set_data(*points_history[index].T)

    return [plotted_points]

k = 10
animation = FuncAnimation(fig,
                          update,
                          frames=range(0, len(points_history), k),
                          interval=1_000/60,
                          blit=True)

animation.save("test.mov", writer="ffmpeg", dpi=100)


import open3d as o3d

h = np.array(points_history)
batch, points, xy = h.shape
point_cloud = np.zeros((batch, points, 3))
point_cloud[:,:,0] = 0.5*np.arange(batch)[:,np.newaxis]
point_cloud[:,:,1:] = 1500*h
point_cloud = point_cloud.reshape(batch*points, 3)

all_lines = np.arange(0, batch*points, points) + np.arange(points)[:,np.newaxis]
all_lines = np.dstack([all_lines, all_lines+points])

all_lines = all_lines[:,:-1]
all_lines.reshape(-1, 2)

cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud))
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(point_cloud),
    lines=o3d.utility.Vector2iVector(all_lines.reshape(-1, 2)))

o3d.visualization.draw_geometries([cloud], zoom=0.6)