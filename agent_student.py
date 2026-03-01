from mesa import Agent
import networkx as nx
import osmnx as ox
import random

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
        self.base_speed = random.uniform(1.1, 1.5)
        self.panic_speed = self.base_speed * 1.8
        self.current_speed = self.base_speed

        self.path = []
        self.frames_current = 0
        self.frames_total = 0

    def pick_random_destination(self):
        all_nodes = list(self.model.G.nodes())
        target_node = random.choice(all_nodes)
        try:
            current_node = ox.distance.nearest_nodes(self.model.G, self.x, self.y)
            full_path = nx.shortest_path(self.model.G, current_node, target_node, weight='length')
            self.path = full_path[1:] if len(full_path) > 1 else []
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            self.path = []