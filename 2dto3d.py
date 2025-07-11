from transformers import pipeline
from PIL import Image
import open3d as o3d
import numpy as np

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
image = Image.open("image copy.png")
depth = pipe(image)["depth"]

depth_array = np.array(depth)
height, width = depth_array.shape
x, y = np.meshgrid(np.arange(width), np.arange(height))
z = depth_array
points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
o3d.visualization.draw_geometries([pcd])