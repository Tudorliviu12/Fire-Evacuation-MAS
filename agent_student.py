from mesa import Agent
import networkx as nx
import osmnx as ox
import random
from config import CALM_SPEED_MIN, CALM_SPEED_MAX, PANIC_THRESHOLD_MAX, PANIC_THRESHOLD_MIN, DEATH_THRESHOLD_MAX, DEATH_THRESHOLD_MIN
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from simulation_model import CampusModel

class Student(Agent):
    def __init__(self, unique_id, model: 'CampusModel', start_node, delay=0, indoors=False, building_idx=None):
        super().__init__(unique_id, model)
        self.model: 'CampusModel' = model
        self.is_active = False
        self.start_delay = delay
        self.is_dead = False
        self.is_panicked = False
        self.color = 'blue'
        self.should_remove = False
        self.indoors = indoors
        self.building_idx = building_idx
        self.personal_panic_threshold = random.uniform(PANIC_THRESHOLD_MIN, PANIC_THRESHOLD_MAX)
        self.personal_death_threshold = random.uniform(DEATH_THRESHOLD_MIN, DEATH_THRESHOLD_MAX)

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

    def become_panicked(self):
        self.is_panicked = True
        self.color = 'red'
        self.current_speed = self.panic_speed
        if self.model.safe_nodes:
            target = random.choice(self.model.safe_nodes)
            try:
                current_node = ox.distance.nearest_nodes(self.model.G_all, self.x, self.y)
                full_path = nx.shortest_path(self.model.G_all, current_node, target, weight='length')
                self.path = full_path[1:]
                self.frames_current = self.frames_total
            except:
                self.path = []

    def check_survival(self):
        if not self.model.fire_started:
            return
        dist = ((self.x - self.model.fire_center_x)**2 + (self.y - self.model.fire_center_y)**2)**0.5

        sight_range = self.personal_panic_threshold + (self.model.current_fire_radius * 1.5)

        if dist<(self.model.current_fire_radius - self.personal_death_threshold):
            self.die()
            return

        if dist<sight_range:
            if not self.is_panicked:
                self.become_panicked()

    def die(self):
        self.is_dead = True
        self.color = 'black'
        self.current_speed = 0
        self.path = []


    def step(self):
        if not self.is_active:
            if self.model.schedule.steps >= self.start_delay:
                self.is_active = True
            else:
                return

        if self.is_dead:
            return

        self.check_survival()
        self.move()