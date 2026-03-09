from mesa import Agent
import networkx as nx
import osmnx as ox
import random
from config import CALM_SPEED_MIN, CALM_SPEED_MAX

class Student(Agent):
    def __init__(self, unique_id, model, start_node, delay=0, indoors=False, building_idx=None):
        super().__init__(unique_id, model)

        self.is_active = False
        self.start_delay = delay
        self.is_dead = False
        self.is_panicked = False
        self.color = 'blue'
        self.should_remove = False
        self.indoors = indoors
        self.building_idx = building_idx

        if self.indoors and building_idx is not None:
            door_coords = self.model.building_doors[building_idx]
            self.x, self.y = door_coords
        else:
            node_data = self.model.nodes_proj.loc[start_node]
            self.x, self.y = node_data.geometry.x, node_data.geometry.y

        self.start_x, self.start_y = self.x, self.y
        self.end_x, self.end_y = self.x, self.y
        self.base_speed = random.uniform(CALM_SPEED_MIN, CALM_SPEED_MAX)
        self.panic_speed = self.base_speed * 1.8
        self.current_speed = self.base_speed

        self.path = []
        self.frames_current = 0
        self.frames_total = 0


    def pick_random_destination(self):
        all_nodes = list(self.model.G_all.nodes())
        target_node = random.choice(all_nodes)
        try:
            current_node = ox.distance.nearest_nodes(self.model.G_all, self.x, self.y)
            full_path = nx.shortest_path(self.model.G_all, current_node, target_node, weight='length')
            self.path = full_path[1:] if len(full_path) > 1 else []
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            self.path = []


    def move(self):
        if not self.path and self.frames_current >= self.frames_total:
            self.pick_random_destination()
            if not self.path:
                return

        if self.frames_current >= self.frames_total:
            next_node = self.path.pop(0)
            node_data = self.model.nodes_proj.loc[next_node]

            self.start_x, self.start_y = self.x, self.y
            self.end_x, self.end_y = node_data.geometry.x, node_data.geometry.y

            dist = ((self.end_x - self.start_x) ** 2 + (self.end_y - self.start_y) ** 2) ** 0.5
            self.frames_total = max(1, int(dist/self.current_speed))
            self.frames_current = 0

        self.frames_current += 1

        fraction = self.frames_current / self.frames_total
        self.x = self.start_x + fraction * (self.end_x - self.start_x)
        self.y = self.start_y + fraction * (self.end_y - self.start_y)


    def step(self):
        if not self.is_active:
            if self.model.schedule.steps >= self.start_delay:
                self.is_active = True
            else:
                return

        if not self.is_dead:
            self.move()