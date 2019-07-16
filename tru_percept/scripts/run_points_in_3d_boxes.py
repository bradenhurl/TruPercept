import tru_percept.points_in_3d_boxes as points_3d
import time

start = time.time()
points_3d.compute_points_in_3d_boxes()
end = time.time()
print("\nTime for points_in_3d_boxes: ", end - start)