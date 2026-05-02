from mesa import Agent
import networkx as nx
import osmnx as ox
import math
import random
from faker import Faker
from pathfinder import DStarLite
from building import Building
from config import TRUCK_DELAY_MAX, TRUCK_DELAY_MIN, GO_TO_DESTINATION_PROB, STUDENT_CHANCE, CALM_SPEED_MIN, CALM_SPEED_MAX, PANIC_THRESHOLD_MAX, PANIC_THRESHOLD_MIN, DEATH_THRESHOLD_MAX, DEATH_THRESHOLD_MIN
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from simulation_model import CampusModel

fake = Faker('ro_RO')

class Student(Agent):
    def __init__(self, unique_id, model: 'CampusModel', start_node, delay=0, indoors=False, building_idx=None):
        super().__init__(unique_id, model)
        self.full_name = fake.name()
        self.model: 'CampusModel' = model
        self.is_active = False
        self.start_delay = delay
        self.is_dead = False
        self.is_aware = False
        self.is_panicked = False
        self.color = 'blue'
        self.should_remove = False
        self.indoors = indoors
        self.building_idx = building_idx
        self.personal_panic_threshold = random.uniform(PANIC_THRESHOLD_MIN, PANIC_THRESHOLD_MAX)
        self.personal_death_threshold = random.uniform(DEATH_THRESHOLD_MIN, DEATH_THRESHOLD_MAX)
        self.target_name = ""
        self.target_node = None
        self.path = []
        self.edge_waypoints = []
        self.is_hidden = False
        self.waiting_timer = 0
        self.reaction_time_ticks = 0
        self.is_frozen = False
        self.frozen_timer = 0
        self.is_calling_112 = False
        if indoors and building_idx is not None and building_idx < 22:
            self.is_resident = True
            self.home_dorm_idx = building_idx
            self.home_dorm = f"T{building_idx + 1}"
        else:
            if random.random() < STUDENT_CHANCE:
                self.is_resident = True
                self.home_dorm_idx = random.randint(0, 21)
                self.home_dorm = f"T{self.home_dorm_idx + 1}"
            else:
                self.is_resident = False
                self.home_dorm_idx = None
                self.home_dorm = "None"

        if self.indoors and building_idx is not None:
            building_agent = self.model.buildings[building_idx]
            self.x, self.y = building_agent.door_coords
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
        self.current_building = None

        self.dstar = None
        self.choose_new_mission()

    def recalculate_path(self, retries=1):
        try:
            if getattr(self.model, 'fire_started', False) and self.target_node is not None:
                n_data = self.model.nodes_proj[self.target_node]
                d_target = math.sqrt((n_data.geometry.x - self.model.fire_center_x)**2 + (n_data.geometry.y - self.model.fire_center_y)**2)
                if d_target <= self.model.current_fire_radius + 15.0:
                    self.pick_safe_destination()
                    return

            curr_node = ox.distance.nearest_nodes(self.model.G_working, self.x, self.y)
            if not self.model.G_working.has_node(curr_node) or not self.model.G_working.has_node(self.target_node):
                raise nx.NodeNotFound("Node not found")
            self.dstar = DStarLite(self.model.G_working, curr_node, self.target_node)
            self.dstar.compute_shortest_path()
            full_path = self.dstar.get_path()
            if not full_path:
                raise nx.NetworkXNoPath("No Path")
            self.path = full_path[1:] if len(full_path) > 1 else []
            self.frames_current = self.frames_total

        except Exception:
            try:
                curr_node = ox.distance.nearest_nodes(self.model.G_all, self.x, self.y)
                if self.model.G_all.has_node(curr_node) and self.model.G_all.has_node(self.target_node):
                    full_path = nx.shortest_path(self.model.G_all, curr_node, self.target_node, weight='length')
                    if full_path:
                        self.path = full_path[1:] if len(full_path) > 1 else []
                        self.edge_waypoints = []
                        self.frames_current = self.frames_total
                        return
            except Exception:
                pass
            self.path = []
            self.edge_waypoints = []

    def pick_safe_destination(self):
        safe_options = []
        safe_weights = []
        for i, node in enumerate(self.model.hotspot_nodes):
            node_data = self.model.nodes_proj.loc[node]
            nx_coord, ny_coord = node_data.geometry.x, node_data.geometry.y
            d_fire = math.sqrt((nx_coord - self.model.fire_center_x)**2 + (ny_coord - self.model.fire_center_y)**2)
            if d_fire > self.model.current_fire_radius + 40.0:
                safe_options.append(i)
                safe_weights.append(self.model.hotspot_weights[i])

        if safe_options:
            choice_idx = random.choices(safe_options, weights=safe_weights, k=1)[0]
            self.target_name = self.model.hotspot_names[choice_idx]
            self.target_node = self.model.hotspot_nodes[choice_idx]
        else:
            self.target_name = "Evacuation Point"
            self.target_node = self.model.hotspot_nodes[0]

        self.edge_waypoints = []
        self.recalculate_path(retries=0)

    def notify_edge_burned(self, u, v):
        if self.dstar is None or self.is_dead or not self.is_active or not self.path:
            return

        path_affected = False
        curr_node = ox.distance.nearest_nodes(self.model.G_working, self.x, self.y)
        full_check_path = [curr_node] + self.path

        for i in range(len(full_check_path) - 1):
            if (full_check_path[i] == u and full_check_path[i + 1] == v) or (full_check_path[i] == v and full_check_path[i + 1] == u):
                path_affected = True
                break

        if path_affected:
            self.dstar.graph = self.model.G_working
            self.dstar.notify_edge_changed(u,v)
            new_path = self.dstar.get_path()
            if new_path:
                self.path = new_path[1:] if len(new_path) > 1 else []
            else:
                self.recalculate_path(retries=1)

    def pick_random_destination(self):
        all_nodes = list(self.model.G_working.nodes())
        target_node = random.choice(all_nodes)
        try:
            current_node = ox.distance.nearest_nodes(self.model.G_working, self.x, self.y)
            full_path = nx.shortest_path(self.model.G_working, current_node, target_node, weight='length')
            self.path = full_path[1:] if len(full_path) > 1 else []
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            self.path = []


    def move(self):
        if self.is_frozen:
            if self.frozen_timer > 0:
                self.frozen_timer -= 1
                return
            else:
                self.is_frozen = False
                self.is_calling_112 = False
                self.choose_new_mission()
                return

        if getattr(self, 'reaction_time_ticks', 0) > 0:
            self.reaction_time_ticks -= 1
            return

        if self.is_hidden:
            self.waiting_timer -= 1
            if self.waiting_timer <= 0:
                building_agent = self.current_building
                if building_agent:
                    if "Mall" in building_agent.name or "T" in building_agent.name: threshold = 40
                    elif "Facultate" in building_agent.name or "Cantina in " in building_agent.name: threshold = 30
                    else: threshold = 3
                    if len(building_agent.inventory) > threshold or self.is_aware:
                        building_agent.inventory.remove(self)
                        self.current_building = None
                        self.is_hidden = False
                        self.x, self.y = building_agent.door_coords
                        self.start_x, self.start_y = self.x, self.y
                        self.choose_new_mission()
                    else:
                        self.waiting_timer = random.randint(250,500)
                else:
                    self.is_hidden = False
                    self.choose_new_mission()
            return

        if not self.path and self.frames_current >= self.frames_total:
            if self.is_aware:
                self.choose_new_mission()
                return

            self.is_hidden = True
            target_building = next((b for b in self.model.buildings if b.door_node == self.target_node), None)
            if target_building:
                self.current_building = target_building
                if self not in target_building.inventory:
                    target_building.inventory.append(self)
                if "Mall" in target_building.name or "T" in target_building.name:
                    self.waiting_timer = random.randint(500,3000)
                elif "Facultate" in target_building.name:
                    self.waiting_timer = random.randint(400,2000)
                else:
                    self.waiting_timer = random.randint(150,500)
            else:
                self.waiting_timer = random.randint(100,300)
                if "Going To" in self.target_name:
                    self.should_remove = True
            return

        if self.frames_current >= self.frames_total:
            if hasattr(self, 'edge_waypoints') and self.edge_waypoints:
                next_pt = self.edge_waypoints.pop(0)
                self.start_x, self.start_y = self.x, self.y
                self.end_x, self.end_y = next_pt
                dist = ((self.end_x - self.start_x)**2 + (self.end_y - self.start_y)**2)**0.5
                self.frames_total = max(1, int(dist / self.current_speed))
                self.frames_current = 0
            elif self.path:
                next_node = self.path.pop(0)
                if self.dstar:
                    if next_node != self.dstar.start:
                        self.dstar.k_m += self.dstar.heuristic(self.dstar.start, next_node)
                        self.dstar.start = next_node
                curr_node = ox.distance.nearest_nodes(self.model.G_all, self.x, self.y)
                self.edge_waypoints = []
                try:
                    edge_data = self.model.G_all.get_edge_data(curr_node, next_node)
                    if edge_data is None:
                        edge_data = self.model.G_all.get_edge_data(next_node, curr_node)
                    if edge_data is not None:
                        key = list(edge_data.keys())[0]
                        data = edge_data[key]
                        if 'geometry' in data:
                            coords = list(data['geometry'].coords)
                            dist_first = math.sqrt((self.x - coords[0][0])**2 + (self.y - coords[0][1])**2)
                            dist_last = math.sqrt((self.x - coords[-1][0])**2 + (self.y - coords[-1][1])**2)
                            if dist_last < dist_first:
                                coords.reverse()
                            self.edge_waypoints = coords[1:]
                except Exception:
                    pass

                if not self.edge_waypoints:
                    node_data = self.model.nodes_proj.loc[next_node]
                    self.start_x, self.start_y = self.x, self.y
                    self.end_x, self.end_y = node_data.geometry.x, node_data.geometry.y
                else:
                    next_pt = self.edge_waypoints.pop(0)
                    self.start_x, self.start_y = self.x, self.y
                    self.end_x, self.end_y = next_pt

                dist = ((self.end_x - self.start_x)**2 + (self.end_y - self.start_y)**2)**0.5
                self.frames_total = max(1, int(dist / self.current_speed))
                self.frames_current = 0

        self.frames_current += 1
        fraction = self.frames_current / self.frames_total

        if self.model.fire_started and (self.is_panicked or self.is_aware):
            next_x = self.start_x + fraction * (self.end_x - self.start_x)
            next_y = self.start_y + fraction * (self.end_y - self.start_y)
            next_dist = math.sqrt((next_x - self.model.fire_center_x)**2 + (next_y - self.model.fire_center_y)**2)
            if next_dist <= self.model.current_fire_radius + 2.5:
                self.path = []
                self.frames_current = self.frames_total
                self.recalculate_path(retries=1)
                return

        self.x = self.start_x + fraction * (self.end_x - self.start_x)
        self.y = self.start_y + fraction * (self.end_y - self.start_y)

    def become_panicked(self):
        self.is_aware = True
        self.color = 'red'

        if self.model.fire_started and not getattr(self.model, 'alarm_triggered', False):
            if random.random() < 0.4:
                self.model.alarm_triggered = True
                self.model.hero_name = self.full_name
                self.model.truck_timer = random.randint(TRUCK_DELAY_MIN,TRUCK_DELAY_MAX)
                self.reaction_time_ticks = 50
                self.is_calling_112 = True
                self.is_frozen = True
                self.frozen_timer = 70

        if not self.is_frozen:
            if len(self.path) == 0:
                self.choose_new_mission()
            else:
                self.recalculate_path(retries=1)

    def check_survival(self):
        if self.is_dead or not self.model.fire_started:
            return

        if self.is_hidden:
            return

        dist = ((self.x - self.model.fire_center_x)**2 + (self.y - self.model.fire_center_y)**2)**0.5
        if dist<(self.model.current_fire_radius - self.personal_death_threshold):
            self.die()
            return

        panic_radius = self.model.current_fire_radius + self.personal_panic_threshold
        if dist < panic_radius:
            self.is_panicked = True
            self.current_speed = self.panic_speed
        else:
            self.is_panicked = False
            self.current_speed = self.base_speed

        if not self.is_aware:
            sight_range = self.personal_panic_threshold + self.model.current_fire_radius * 3.0
            if dist < sight_range:
                self.become_panicked()
                return
            for smoke in self.model.smoke_blobs[::5]:
                d_smoke = math.sqrt((self.x - smoke['x'])**2 + (self.y - smoke['y'])**2)
                if d_smoke < 8.0:
                    self.become_panicked()
                    return

        if not self.is_panicked:
            for smoke in self.model.smoke_blobs[::5]:
                d_smoke = ((self.x - smoke['x'])**2 + (self.y - smoke['y'])**2)**0.5
                if d_smoke < 8.0:
                    self.become_panicked()
                    return

    def check_surroundings(self):
        if self.is_dead or not self.model.fire_started or self.is_panicked:
            return
        panicked_nearby = 0
        for agent in self.model.active_agents_cache:
            if agent is not self and agent.is_panicked:
                d = ((self.x - agent.x)**2 + (self.y - agent.y)**2)**0.5
                if d<25.0:
                    panicked_nearby += 1
                    if panicked_nearby >= 2:
                        break

        if panicked_nearby >= 2:
            if random.random() < 0.1:
                self.become_panicked()

    def plan_next_move(self):
        if self.is_dead: return
        if self.is_frozen: return

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

    def choose_new_mission(self):
        if self.is_aware and getattr(self.model, 'fire_started', False):
            if self.target_node is not None:
                nd = self.model.nodes_proj.loc[self.target_node]
                d_fire = math.sqrt((nd.geometry.x - self.model.fire_center_x)**2 + (nd.geometry.y - self.model.fire_center_y)**2)
                if d_fire > self.model.current_fire_radius + 15.0:
                    self.recalculate_path()
                    return
            self.pick_safe_destination()
            return
        if random.random() < GO_TO_DESTINATION_PROB:
            choice_idx = random.choices(
                range(len(self.model.hotspot_names)),
                weights=self.model.hotspot_weights,
                k=1,
            )[0]
            self.target_name = self.model.hotspot_names[choice_idx]
            self.target_node = self.model.hotspot_nodes[choice_idx]

        else:
            if self.home_dorm_idx is not None:
                self.target_name = f"Cămin T{self.home_dorm_idx+1}"
                self.target_node = self.model.dorm_nodes[self.home_dorm_idx]
            else:
                exit_options = [i for i, name in enumerate(self.model.hotspot_names) if "Going To" in name]
                if exit_options:
                    idx = random.choice(exit_options)
                    self.target_name = self.model.hotspot_names[idx]
                    self.target_node = self.model.hotspot_nodes[idx]
                else:
                    self.target_name = "Iulius Mall"
                    self.target_node = self.model.hotspot_nodes[0]

        self.recalculate_path()


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

        self.check_survival()

        if self.is_dead:
            return

        if self.model.schedule.steps % 20 == 0:
            self.check_surroundings()

        self.move()