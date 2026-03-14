from mesa import Agent
import networkx as nx
import osmnx as ox
import math
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
        self.panic_speed = self.base_speed * 2.0
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
            if self.is_panicked:
                self.should_remove = True
                return
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
        if self.is_dead or not self.model.fire_started:
            return
        dist = ((self.x - self.model.fire_center_x)**2 + (self.y - self.model.fire_center_y)**2)**0.5

        sight_range = self.personal_panic_threshold + (self.model.current_fire_radius * 3.0)

        if dist<(self.model.current_fire_radius - self.personal_death_threshold):
            self.die()
            return

        if not self.is_panicked:
            for smoke in self.model.smoke_blobs[::5]:
                d_smoke = ((self.x - smoke['x'])**2 + (self.y - smoke['y'])**2)**0.5
                if d_smoke < 8.0:
                    self.become_panicked()
                    return

        if dist<sight_range:
            if not self.is_panicked:
                self.become_panicked()

    def check_surroundings(self):
        if self.is_dead or not self.model.fire_started or self.is_panicked:
            return
        panicked_nearby = 0
        for agent in self.model.schedule.agents:
            if isinstance(agent, Student) and agent!=self and agent.is_panicked:
                d = ((self.x - agent.x)**2 + (self.y - agent.y)**2)**0.5
                if d<12.0:
                    panicked_nearby += 1

        if panicked_nearby >= 2:
            if random.random() < 0.1:
                self.become_panicked()


    def plan_next_move(self):
        if len(self.path) == 0:
            current_node = ox.distance.nearest_nodes(self.model.G_all, self.x, self.y)
            neighbors = list(self.model.G_all.neighbors(current_node))
            cand_nodes = []

            for n in neighbors:
                nx_x = self.model.nodes_proj.loc[n].geometry.x
                nx_y = self.model.nodes_proj.loc[n].geometry.y
                d_fire_n = math.sqrt((nx_x-self.model.fire_center_x)**2 + (nx_y-self.model.fire_center_y)**2)
                cand_nodes.append((n, d_fire_n))
            if cand_nodes:
                cand_nodes.sort(key=lambda x: x[1], reverse=True)
                next_node = cand_nodes[0][0]
                self.path = [next_node]

    def die(self):
        self.is_dead = True
        self.color = 'black'
        self.current_speed = 0
        self.path = []


    def step(self):
        if not self.is_active or self.is_dead:
            if not self.is_active:
                if self.model.schedule.steps >= self.start_delay:
                    self.is_active = True
                else: return
            if self.is_dead:
                return

        self.check_survival()

        if self.is_dead:
            return

        if self.model.schedule.steps % 20 == 0:
            self.check_surroundings()

        if self.is_panicked:
            curr_node = ox.distance.nearest_nodes(self.model.G_all, self.x, self.y)
            if curr_node in self.model.safe_nodes:
                self.should_remove = True
                return
            self.plan_next_move()

        self.move()