import random
import math
from interior_grid import InteriorGrid
from shapely.geometry import Point

class Building:
    def __init__(self, name, door_node, door_coords, area):
        self.model = None
        self.name = name
        self.door_node = door_node
        self.door_coords = door_coords
        self.inventory = []
        self.is_on_fire = False
        self.area = area
        self.polygon_coords = None
        self.interior_grid = None
        self.local_coords = None
        self.local_bbox = None
        self.osm_polygon = None

    def init_interior(self):
        if self.interior_grid is not None:
            return
        if not self.polygon_coords:
            return
        coords = self.polygon_coords
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        rng = max(maxx - minx, maxy - miny)
        if rng == 0:
            return
        local_coords = [((x-minx)/rng*100, (y-miny)/rng*100) for x, y in coords]
        self.local_coords = local_coords
        lxs = [c[0] for c in local_coords]
        lys = [c[1] for c in local_coords]
        self.local_bbox = (min(lxs), min(lys), max(lxs), max(lys))
        self.interior_grid = InteriorGrid(polygon_coords_local=local_coords, building=self, n_floor=5, grid_spacing=6.0)

    def evacuate_step(self):
        if self.is_on_fire and len(self.inventory) > 0:
            if not self.interior_grid:
                num_to_evacuate = min(len(self.inventory), random.randint(1,3))
                for _ in range(num_to_evacuate):
                    idx = random.randint(0,len(self.inventory)-1)
                    blob = self.inventory.pop(idx)
                    blob.is_hidden = False
                    offset_x = random.uniform(-0.5, 0.5)
                    offset_y = random.uniform(-0.5, 0.5)
                    blob.x = self.door_coords[0] + offset_x
                    blob.y = self.door_coords[1] + offset_y
                    blob.start_x, blob.start_y = blob.x, blob.y
                    blob.end_x, blob.end_y = blob.x, blob.y
                    blob.become_panicked()

    def accept_student(self, student):
        if student not in self.inventory:
            self.inventory.append(student)
        self.init_interior()
        if self.interior_grid:
            self.interior_grid.add_student_from_outside(student)