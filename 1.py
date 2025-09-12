
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

# data = torch.tensor([[0, 1]*70]*5 + [[1, 0]*70]*5)

def run(points, data, constraints, rate=5e-3):
    for time_slice in data:
        # active_points = points[time_slice == 1]
        active_points = points

        pairwise = active_points - active_points[:,torch.newaxis]
        pairwise += 1e-5
        distances_all = (pairwise**2).sum(axis=-1)**0.5

        distances_cohort = distances_all[time_slice == 1][:,time_slice == 1]
        upper_triangle = torch.ones_like(distances_cohort, dtype=bool).triu(1)
        distances_cohort_list = distances_cohort[upper_triangle]

        distances_opposing = distances_all[time_slice == 0][:,time_slice == 1]
        upper_triangle = torch.ones_like(distances_opposing, dtype=bool).triu(1)
        distances_opposing_list = distances_opposing[upper_triangle]

        upper_triangle = torch.ones_like(distances_all, dtype=bool).triu(1)
        distances_all_list = distances_all[upper_triangle]

        centers_of_mass_0 = points[time_slice == 0].mean()
        centers_of_mass_1 = points[time_slice == 1].mean()
        centers_of_mass_distance = ((centers_of_mass_0 - centers_of_mass_1)**2).sum()**0.5

        loss = 0
        if len(distances_cohort_list) > 0:
            loss += ((0.1 - distances_cohort_list)**2).mean()

        # radii_active = torch.linalg.norm(active_points, dim=-1)
        # if len(radii_active) > 0:
        #     loss += radii_active.mean()

        distances_inside = distances_opposing_list[distances_opposing_list < 2]
        if len(distances_inside) > 0:
            loss += ((2 - distances_inside)**2).mean()

        # if len(distances_all_list) > 0:
        #     loss += 0.25*((0.5 - distances_all)**2).mean()

        inactive_points = points[time_slice == 0]
        distances_inactive = torch.linalg.norm(inactive_points, dim=-1)
        distances_inside = distances_inactive[distances_inactive < 1.5]
        # distances_outside = distances_inactive[distances_inactive > 2.5]
        if len(distances_inside) > 0:
            loss += 0.25*((2 - distances_inside)**2).mean()

        distances_outside = distances_opposing_list[distances_opposing_list > 2]
        if len(distances_outside) > 0:
            loss += ((2 - distances_outside)**2).mean()

        # if not torch.isnan(centers_of_mass_distance):
        #     loss += (2 - centers_of_mass_distance)**2

        # for constraint in constraints:
        #     loss += constraint(distances_cohort_list,
        #                        distances_complement_list)

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

rest_locations = torch.tensor(points)

points = torch.tensor(points, requires_grad=True)
points_history = [points.cpu().clone().detach().numpy()]


section_length = 5
step_size = 5

count = 0
for section_index in list(range(0, len(data.T) // 5 - section_length, step_size)):
    for _ in range(data.T[section_index:section_index+section_length].sum()*5):
        count += 1

print({"count": count})

t1 = time.time()
for section_index in list(range(0, len(data.T) // 5 - section_length, step_size)):
    for _ in range(data.T[section_index:section_index+section_length].sum()*5):
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

k = 5
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
point_cloud[:,:,1:] = 500*h
point_cloud = point_cloud.reshape(batch*points, 3)
print(point_cloud.shape)

all_lines = np.arange(0, batch*points, points) + np.arange(points)[:,np.newaxis]
all_lines = np.dstack([all_lines, all_lines+points])

all_lines = all_lines[:,:-1]
all_lines.reshape(-1, 2)

cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud))
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(point_cloud),
    lines=o3d.utility.Vector2iVector(all_lines.reshape(-1, 2)))

o3d.visualization.draw_geometries([line_set, cloud], zoom=0.6)
# o3d.visualization.draw([line_set, cloud], line_width=1)