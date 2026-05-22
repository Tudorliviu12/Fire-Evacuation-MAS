import math
import random
import networkx as nx
import numpy as np
from scipy.spatial import KDTree
from shapely import Polygon, Point, LineString
from interior_agent import InteriorAgent

class InteriorGrid:

    def __init__(self, polygon_coords_local, building, n_floor=5, grid_spacing=5.0):
        self.polygon = Polygon(polygon_coords_local)
        temp_safe = self.polygon.buffer(-1.0)
        self.safe_polygon = temp_safe if not temp_safe.is_empty else self.polygon
        self.building = building
        self.n_floor = n_floor
        self.grid_spacing = grid_spacing

        self.graph = nx.Graph()
        self.grid_nodes = self.generate_grid()
        self.floors = {i: [] for i in range(n_floor)}

        self.fire_floors = set()
        self.fire_centers = {}
        self.fire_blobs = []
        self.fire_spread_timer = 0

        self.mass_evacuation = False

        self.init_agents()
        self.stair_nodes = self.find_stair_nodes()
        self.stair_cooldown = {s: 0 for s in self.stair_nodes}
        self.stair_occupancy = {s: 0 for s in self.stair_nodes}
        self.stair_queues = {s: [] for s in self.stair_nodes}
        self.fire_alarm_active = False

        if self.grid_nodes:
            self.grid_center = (sum(n[0] for n in self.grid_nodes) / len(self.grid_nodes), sum(n[1] for n in self.grid_nodes) / len(self.grid_nodes),)
        else:
            self.grid_center = (0.0, 0.0)

        self.build_kdtree()
        self.path_cache: dict = {}
        self.graph_version: int = 0
        self.path_cache_version: int = -1
        self.active_nodes_cache = None
        self.active_nodes_dirty = True
        self.burned_nodes: set = set()
        self.evac_paths = {}
        self.update_evac_paths()

    def build_kdtree(self):
        if self.grid_nodes:
            coords = np.array(self.grid_nodes, dtype=np.float64)
            self.kdtree = KDTree(coords)
            self.kdtree_nodes = self.grid_nodes
        else:
            self.kdtree = None
            self.kdtree_nodes = []

    def get_nearest_node(self, x, y):
        if self.kdtree is None:
            return min(self.grid_nodes, key=lambda n: (x - n[0])**2 + (y - n[1])**2)
        _, idx = self.kdtree.query([x, y])
        return self.kdtree_nodes[idx]

    def get_active_nodes(self):
        if self.active_nodes_dirty or self.active_nodes_cache is None:
            active = [
                n for n in self.grid_nodes
                if self.graph.has_node(n) and self.graph.degree(n) > 0
            ]
            self.active_nodes_cache = active if active else self.grid_nodes
            self.active_nodes_dirty = False
        return self.active_nodes_cache

    def get_nearest_active_node(self, x, y):
        active = self.get_active_nodes()
        if active is self.grid_nodes and self.kdtree is not None:
            return self.get_nearest_node(x, y)
        return min(active, key=lambda n: (x - n[0])**2 + (y - n[1])**2)

    def get_cached_path(self, source, target):
        if self.path_cache_version != self.graph_version:
            self.path_cache.clear()
            self.path_cache_version = self.graph_version
        key = (source, target)
        cached = self.path_cache.get(key)
        if cached is not None:
            return cached
        try:
            path = nx.shortest_path(self.graph, source, target, weight='weight')
            self.path_cache[key] = path
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            self.path_cache[key] = None
            return None

    def set_fire_on_floor(self, floor, x=None, y=None):
        self.fire_floors.add(floor)
        self.building.model.fire_center_x = self.building.door_coords[0]
        self.building.model.fire_center_y = self.building.door_coords[1]
        self.building.model.fire_started = True
        if floor not in self.fire_centers:
            if x is not None and y is not None:
                cx, cy = x, y
            elif self.stair_nodes:
                st = random.choice(self.stair_nodes)
                cx, cy = st[0], st[1]
            elif self.grid_nodes:
                n = random.choice(self.grid_nodes)
                cx, cy = n[0], n[1]
            else:
                cx, cy = 50.0, 50.0
            self.fire_centers[floor] = {'x': cx, 'y': cy, 'radius': 1.5}

    def generate_grid(self):
        minx, miny, maxx, maxy = self.polygon.bounds
        nodes = []
        x = minx + self.grid_spacing
        while x < maxx:
            y = miny + self.grid_spacing
            while y < maxy:
                if self.safe_polygon.contains(Point(x, y)):
                    nodes.append((x, y))
                y += self.grid_spacing
            x += self.grid_spacing

        for node in nodes:
            self.graph.add_node(node)
        threshold = self.grid_spacing * 1.5
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                n1, n2 = nodes[i], nodes[j]
                dist = math.sqrt((n1[0]-n2[0])**2 + (n1[1]-n2[1])**2)
                if dist <= threshold:
                    line = LineString([n1, n2])
                    if self.polygon.contains(line):
                        self.graph.add_edge(n1, n2, weight=dist)
        return nodes

    def find_stair_nodes(self):
        if not self.grid_nodes:
            return []
        left  = min(self.grid_nodes, key=lambda n: n[0])
        right = max(self.grid_nodes, key=lambda n: n[0])
        return [left, right]

    def init_agents(self):
        inv = self.building.inventory
        if not inv or not self.grid_nodes:
            return
        for i, student in enumerate(inv):
            floor = getattr(student, 'target_floor', random.randint(0, self.n_floor - 1))
            floor = min(floor, self.n_floor - 1)
            node  = random.choice(self.grid_nodes)
            x = node[0] + random.uniform(-1.5, 1.5)
            y = node[1] + random.uniform(-1.5, 1.5)
            if not self.safe_polygon.contains(Point(x, y)):
                x = node[0] + random.uniform(-0.3, 0.3)
                y = node[1] + random.uniform(-0.3, 0.3)
            agent = InteriorAgent(agent_id=i, x=x, y=y, floor=floor, map_student=student)
            self.floors[floor].append(agent)

    def move_agent_to_floor(self, agent, new_floor):
        if agent in self.floors.get(agent.floor, []):
            self.floors[agent.floor].remove(agent)
        agent.floor = new_floor
        if new_floor in self.floors:
            self.floors[new_floor].append(agent)

    def exit_building(self, agent):
        for floor_list in self.floors.values():
            if agent in floor_list:
                floor_list.remove(agent)
                break

        for q in self.stair_queues.values():
            if agent in q:
                q.remove(agent)

        if agent.map_student is None:
            return

        if agent.map_student in self.building.inventory:
            self.building.inventory.remove(agent.map_student)

        stud = agent.map_student
        stud.is_hidden = False
        stud.indoors = False
        stud.current_building = None
        offset_x = random.uniform(-1.5, 1.5)
        offset_y = random.uniform(-1.5, 1.5)
        stud.x = self.building.door_coords[0] + offset_x
        stud.y = self.building.door_coords[1] + offset_y
        stud.start_x, stud.start_y = stud.x, stud.y
        stud.end_x, stud.end_y = stud.x, stud.y
        stud.path = []
        stud.edge_waypoints = []
        stud.frames_current = 0
        stud.frames_total = 1

        if agent.is_aware_of_fire or self.building.is_on_fire:
            stud.color = 'red'
            if not stud.is_panicked:
                stud.become_panicked()
            stud.target_node = None
            stud.pick_safe_destination()
            attempts = 0
            while stud.target_node == self.building.door_node and attempts < 5:
                stud.target_node = None
                stud.pick_safe_destination()
                attempts += 1
            if hasattr(stud, 'init_path_to_target'):
                stud.init_path_to_target()
        else:
            stud.choose_new_mission()
            if hasattr(stud, 'init_path_to_target'):
                stud.init_path_to_target()

        agent.map_student = None

    def add_student_from_outside(self, student):
        if not self.grid_nodes:
            return
        floor = 0
        target = getattr(student, 'target_floor', random.randint(0, self.n_floor - 1))
        best_stair = self.stair_nodes[0] if self.stair_nodes else random.choice(self.grid_nodes)
        agent = InteriorAgent(agent_id=student.unique_id, x=best_stair[0], y=best_stair[1], floor=floor, map_student=student)
        agent.target_floor = target
        self.floors[floor].append(agent)

    def update_evac_paths(self):
        self.evac_paths = {}
        for stair in self.stair_nodes:
            if stair in self.graph:
                try:
                    paths = nx.single_source_shortest_path(self.graph, stair, weight='weight')
                    for node, path in paths.items():
                        rev_path = list(reversed(path))
                        if node not in self.evac_paths or len(rev_path) < len(self.evac_paths[node]):
                            self.evac_paths[node] = rev_path
                except Exception:
                    pass


    def step(self):
        if getattr(self, 'fire_extinguished_indoors', False):
            return

        if hasattr(self, 'interior_water_particles'):
            self.interior_water_particles = [p for p in self.interior_water_particles if p['life'] > 0]
            for p in self.interior_water_particles:
                p['x'] += p['vx']
                p['y'] += p['vy']
                p['life'] -= 1

        for floor, fc in self.fire_centers.items():
            growth_speed = 0.01 + (fc['radius'] / 60.0) * 0.08
            fc['radius'] = min(fc['radius'] + growth_speed, 60.0)

            if not getattr(self, 'fire_alarm_active', False) and any(fc['radius'] >= 5.0 for fc in self.fire_centers.values()):
                self.fire_alarm_active = True
                campus_model = getattr(self.building, 'model', None)
                if campus_model and not getattr(campus_model, 'alarm_triggered', False):
                    campus_model.alarm_triggered = True
                    campus_model.truck_timer = random.randint(80,150)
                    campus_model.hero_name = f"Fire Alarm in {self.building.name}"

                for floor_agents in self.floors.values():
                    for agent in floor_agents:
                        if not agent.is_aware_of_fire:
                            r = random.random()
                            if r < 0.3:
                                agent.alarm_response_timer = random.randint(0, 20)
                            elif r < 0.7:
                                agent.alarm_response_timer = random.randint(40, 100)
                            else:
                                agent.alarm_response_timer = random.randint(120, 250)

            if hasattr(self, 'graph'):
                cx, cy, rad = fc['x'], fc['y'], fc['radius'] * 0.8
                r_sq = rad*rad
                nodes_to_check = [
                    n for n in self.grid_nodes
                    if n not in self.burned_nodes
                    and abs(n[0] - cx) <= rad and abs(n[1] - cy) <= rad
                ]
                nodes_to_burn = [
                    n for n in nodes_to_check
                    if (n[0]-cx)**2 + (n[1]-cy)**2 < r_sq
                ]
                if nodes_to_burn:
                    for n in nodes_to_burn:
                        if self.graph.has_node(n):
                            edges = list(self.graph.edges(n))
                            if edges:
                                self.graph.remove_edges_from(edges)
                        self.burned_nodes.add(n)
                    self.graph_version += 1
                    self.active_nodes_dirty = True
                    self.update_evac_paths()

            target_blobs = min(int(fc['radius'] * 5), 150)
            blobs_floor  = [b for b in self.fire_blobs if b['floor'] == floor]

            if len(blobs_floor) < target_blobs:
                needed = target_blobs - len(blobs_floor)
                for _ in range(needed*2):
                    if len(blobs_floor) >= target_blobs: break
                    ang = random.uniform(0, 2 * math.pi)
                    dst = random.uniform(0, fc['radius'])
                    bx  = fc['x'] + math.cos(ang) * dst
                    by  = fc['y'] + math.sin(ang) * dst
                    if self.polygon.contains(Point(bx, by)):
                        blob = {'x': bx, 'y': by, 'floor': floor}
                        self.fire_blobs.append(blob)
                        blobs_floor.append(blob)
            if len(blobs_floor) > target_blobs:
                other = [b for b in self.fire_blobs if b['floor'] != floor]
                self.fire_blobs = other + random.sample(blobs_floor, target_blobs)

        for s in self.stair_cooldown:
            if self.stair_cooldown[s] > 0:
                self.stair_cooldown[s] -= 1

        for floor_agents in self.floors.values():
            for agent in list(floor_agents):
                agent.step(
                    self.polygon, self.safe_polygon,
                    self.grid_nodes, self.stair_nodes, self
                )

    def get_agents_on_floor(self, floor):
        return self.floors.get(floor, [])

    def get_stair_nodes(self):
        return self.stair_nodes