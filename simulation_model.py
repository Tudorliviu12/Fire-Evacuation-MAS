import mesa
from mesa.time import RandomActivation
from mesa import Model
from agent_student import Student
import random
import math

from config import FIRE_GROWTH_MIN, FIRE_GROWTH_MAX, MAX_FIRE_RADIUS_SOFT_CAP


class CampusModel(Model):
    def __init__(self, G_all, G_drive, nodes_proj, doors, n_students):
        super().__init__()
        self.G_all = G_all
        self.G_drive = G_drive
        self.nodes_proj = nodes_proj
        self.building_doors = doors
        self.schedule = mesa.time.RandomActivation(self)
        self.n_students = n_students
        self.fire_started = False
        self.fire_center_x = 0
        self.fire_center_y = 0
        self.current_fire_radius: float = 0.0
        self.fire_blobs = []
        self.safe_nodes = []

        all_nodes_ids = list(self.nodes_proj.index)
        for i in range(n_students):
            start_node = random.choice(all_nodes_ids)
            delay = 0
            a = Student(i, self, start_node, delay=delay, indoors=False)
            self.schedule.add(a)

    def ignite_fire(self, x, y):
        if self.fire_started:
            return
        print(f"Fire started at {x}, {y}")
        self.fire_started = True
        self.fire_center_x = x
        self.fire_center_y = y
        self.current_fire_radius = 1.5


    def step(self):
        if self.fire_started:
            growth = random.uniform(FIRE_GROWTH_MIN, FIRE_GROWTH_MAX)
            if self.current_fire_radius > MAX_FIRE_RADIUS_SOFT_CAP:
                growth = growth * 0.3
            self.current_fire_radius += growth
            target_blobs = min(int(self.current_fire_radius * 8), 500)
            while len(self.fire_blobs) < target_blobs:
                ang = random.uniform(0, 2*math.pi)
                dst = random.uniform(0, self.current_fire_radius)
                self.fire_blobs.append({
                    'x': self.fire_center_x + math.cos(ang)*dst,
                    'y': self.fire_center_y + math.sin(ang)*dst,
                })
                if len(self.fire_blobs) > target_blobs:
                    self.fire_blobs = self.fire_blobs[:target_blobs]

        self.schedule.step()