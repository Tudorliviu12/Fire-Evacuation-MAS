from mesa import Agent
import networkx as nx
import osmnx as ox
import math
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from simulation_model import CampusModel

class Firetruck(Agent):
    def __init__(self, unique_id, model: 'CampusModel', start_node):
        super().__init__(unique_id, model)
        self.model: 'CampusModel' = model
        node_data = self.model.nodes_proj.loc[start_node]
        self.x, self.y = node_data.geometry.x, node_data.geometry.y
        self.start_x, self.start_y = self.x, self.y
        self.end_x, self.end_y = self.x, self.y
        self.base_speed = 3.5
        self.current_speed = self.base_speed
        self.path = []
        self.frames_current = 0
        self.frames_total = 1
        self.has_arrived = False
        self.calculate_route_to_fire()

    def calculate_route_to_fire(self):
        start_n = ox.distance.nearest_nodes(self.model.G_drive, self.x, self.y)
        target_n = ox.distance.nearest_nodes(self.model.G_drive, self.model.fire_center_x, self.model.fire_center_y)
        try:
            full_path = nx.shortest_path(self.model.G_drive, start_n, target_n, weight='length')
            self.path = full_path[1:] if len(full_path) > 1 else []
        except:
            self.path = []
            self.has_arrived = True

    def check_traffic(self):
        panicked_in_way = 0
        for agent in self.model.active_agents_cache:
            if not getattr(agent, 'is_hidden', False) and getattr(agent, 'is_panicked', False):
                dist = math.sqrt((self.x - agent.x)**2 + (self.y - agent.y)**2)
                if dist<20.0:
                    panicked_in_way += 1
        in_smoke = self.model.is_near_any_smoke(self.x, self.y)
        if in_smoke:
            self.current_speed = self.base_speed * 0.5
        elif panicked_in_way > 6:
            self.current_speed = self.base_speed * 0.7
        else:
            self.current_speed = self.base_speed

    def move(self):
        if self.has_arrived:
            return
        self.check_traffic()

        dist_to_fire = math.sqrt((self.x - self.model.fire_center_x)**2 + (self.y - self.model.fire_center_y)**2)
        if dist_to_fire <= self.model.current_fire_radius + 20.0:
            self.has_arrived = True
            self.path = []
            return

        if self.frames_current >= self.frames_total:
            if not self.path:
                self.has_arrived = True
                return
            next_node = self.path.pop(0)
            node_data = self.model.nodes_proj.loc[next_node]
            self.start_x, self.start_y = self.x, self.y
            self.end_x, self.end_y = node_data.geometry.x, node_data.geometry.y
            dist = math.sqrt((self.end_x - self.start_x)**2 + (self.end_y - self.start_y)**2)
            self.frames_total = max(1, int(dist/max(0.1, self.current_speed)))
            self.frames_current = 0

        self.frames_current += 1
        fraction = self.frames_current / self.frames_total
        self.x = self.start_x + fraction * (self.end_x - self.start_x)
        self.y = self.start_y + fraction * (self.end_y - self.start_y)

    def step(self):
        self.move()