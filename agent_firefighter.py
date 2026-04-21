from mesa import Agent
import networkx as nx
import osmnx as ox
import math
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from simulation_model import CampusModel

class Firefighter(Agent):
    def __init__(self, unique_id, model: 'CampusModel', start_x, start_y):
        super().__init__(unique_id, model)
        self.model: 'CampusModel' = model
        self.x, self.y = start_x, start_y
        self.start_x, self.start_y = self.x, self.y
        self.end_x, self.end_y = self.x, self.y

        self.base_speed = 5.0
        self.current_speed = self.base_speed
        self.path = []
        self.frames_current = 0
        self.frames_total = 1
        self.has_arrived = False
        self.is_active = True
        self.is_panicked = False
        self.calculate_route_to_fire()

    def calculate_route_to_fire(self):
        start_n = ox.distance.nearest_nodes(self.model.G_all, self.x, self.y)
        target_n = ox.distance.nearest_nodes(self.model.G_all, self.model.fire_center_x, self.model.fire_center_y)
        try:
            node_path = nx.shortest_path(self.model.G_all, start_n, target_n, weight='length')
            self.path = []
            for i in range(len(node_path) - 1):
                v = node_path[i+1]
                vx = self.model.G_all.nodes[v].get('x', 0)
                vy = self.model.G_all.nodes[v].get('y', 0)
                self.path.append((vx, vy))
        except Exception as e:
            self.path = []
            self.has_arrived = True


    def move(self):
        if self.has_arrived:
            return

        dist_to_fire = math.sqrt((self.x - self.model.fire_center_x)**2 + (self.y - self.model.fire_center_y)**2)
        if dist_to_fire <= self.model.current_fire_radius + 5.0:
            self.has_arrived = True
            self.path = []
            return
        if self.frames_current >= self.frames_total:
            if not self.path:
                self.has_arrived = True
                return
            next_point = self.path.pop(0)
            self.start_x, self.start_y = self.x, self.y
            self.end_x, self.end_y = next_point[0], next_point[1]
            dist = math.sqrt((self.end_x - self.start_x)**2 + (self.end_y - self.start_y)**2)
            self.frames_total = max(1, int(dist/max(0.1, self.current_speed)))
            self.frames_current = 0

        self.frames_current += 1
        fraction = self.frames_current / self.frames_total
        self.x = self.start_x + fraction * (self.end_x - self.start_x)
        self.y = self.start_y + fraction * (self.end_y - self.start_y)

    def step(self):
        self.move()
