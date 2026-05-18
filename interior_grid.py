import math
import random
import networkx as nx
from shapely import Polygon, Point, LineString


class InteriorAgent:
    def __init__(self, agent_id, x, y, floor, map_student=None):
        self.agent_id = agent_id
        self.x = x
        self.y = y
        self.floor = floor
        self.map_student = map_student
        if random.random() < 0.15:
            self.rest_timer = 0
        else:
            self.rest_timer = random.randint(20,300)

        self.path = []
        self.target_floor = getattr(map_student, 'target_floor', floor)
        self.is_exiting = False
        self.stair_timer = 0
        self.in_transit = False

    def move(self, speed):
        if not self.path:
            return True
        next_node = self.path[0]
        dx = next_node[0] - self.x
        dy = next_node[1] - self.y
        dist = math.sqrt(dx*dx + dy*dy)

        if dist <= speed:
            self.x = next_node[0]
            self.y = next_node[1]
            self.path.pop(0)
            if not self.path:
                return True
        else:
            self.x += (dx/dist)*speed
            self.y += (dy/dist)*speed
        return False

    def step(self, polygon, safe_polygon, grid_nodes, stair_nodes, building_grid):
        if getattr(building_grid.building, 'is_on_fire', False):
            is_panicked_already = self.map_student and self.map_student.is_panicked
            if self.floor in building_grid.fire_floors:
                if not is_panicked_already and self.map_student:
                    self.map_student.become_panicked()
                    self.is_exiting = True
                    self.target_floor = 0
                    self.rest_timer = 0
                    self.path = []
            elif random.random() < 0.005:
                if not is_panicked_already and self.map_student:
                    self.map_student.become_panicked()
                    self.is_exiting = True
                    self.target_floor = 0
                    self.rest_timer = 0
                    self.path = []

        if self.stair_timer > 0:
            self.stair_timer -= 1
            if self.stair_timer <= 0:
                self.in_transit = False
                building_grid.move_agent_to_floor(self, self.target_floor)
                if self.is_exiting and self.target_floor == 0:
                    building_grid.exit_building(self)
                else:
                    self.rest_timer = 0
                    self.path = []
                    if grid_nodes:
                        valid_nodes = [n for n in grid_nodes if (n[0]-self.x)**2 + (n[1]-self.y)**2 > 10.0]
                        if valid_nodes:
                            target = random.choice(valid_nodes)
                            closest_current = min(grid_nodes, key=lambda n: (self.x - n[0])**2 + (self.y - n[1])**2)
                            try:
                                self.path = nx.shortest_path(building_grid.graph, closest_current, target, weight='weight')
                            except nx.NetworkXNoPath:
                                self.path = [target]
            return

        if self.floor != self.target_floor or self.is_exiting:
            if not stair_nodes:
                return
            best_stair = min(stair_nodes, key=lambda s: (self.x - s[0])**2 + (self.y - s[1])**2)
            dist_to_stair = math.sqrt((best_stair[0] - self.x)**2 + (best_stair[1] - self.y)**2)

            if dist_to_stair < 2.0:
                self.in_transit = True
                self.path = []
                floor_difference = abs(self.floor - self.target_floor)
                if floor_difference == 0 and self.is_exiting:
                    self.stair_timer = 20
                else:
                    multiplier = 8 if (self.map_student and self.map_student.is_panicked) else 25
                    self.stair_timer = max(5, floor_difference * multiplier)
            else:
                if not self.path or self.path[-1] != best_stair:
                    closest_current = min(grid_nodes, key=lambda n: (self.x - n[0])**2 + (self.y - n[1])**2)
                    try:
                        self.path = nx.shortest_path(building_grid.graph, closest_current, best_stair, weight='weight')
                    except nx.NetworkXNoPath:
                        self.path = [best_stair]
                move_speed = 1.0 if (self.map_student and self.map_student.is_panicked) else 0.3
                self.move(move_speed)
            return

        if self.rest_timer > 0:
            self.rest_timer -= 1
            if random.random() < 0.0005:
                self.is_exiting = True
                self.target_floor = 0
                self.path = []
                if self.map_student:
                    self.map_student.choose_new_mission()
            elif random.random() < 0.001:
                self.target_floor = random.randint(0, building_grid.n_floor-1)
                self.path = []
            return

        if not self.path:
            if grid_nodes:
                target_node = random.choice(grid_nodes)
                closest_current = min(grid_nodes, key=lambda n: (self.x - n[0])**2 + (self.y - n[1])**2)
                try:
                    self.path = nx.shortest_path(building_grid.graph, closest_current, target_node, weight='weight')
                except nx.NetworkXNoPath:
                    self.path = [target_node]
            return

        arrived = self.move(0.2)
        if arrived:
            self.rest_timer = random.randint(90, 400)


class InteriorGrid:
    def __init__(self, polygon_coords_local, building, n_floor=5, grid_spacing=5.0):
        self.polygon = Polygon(polygon_coords_local)
        temp_safe = self.polygon.buffer(-1.0)
        if temp_safe.is_empty:
            self.safe_polygon = self.polygon
        else:
            self.safe_polygon = temp_safe
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

        self.init_agents()
        self.stair_nodes = self.find_stair_nodes()

    def set_fire_on_floor(self, floor, x=None, y=None):
        self.fire_floors.add(floor)

        if floor not in self.fire_centers:
            if x is not None and y is not None:
                cx, cy = x, y
            else:
                if self.stair_nodes:
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
                pt = Point(x, y)
                if self.safe_polygon.contains(pt):
                    nodes.append((x, y))
                y += self.grid_spacing
            x += self.grid_spacing

        for node in nodes:
            self.graph.add_node(node)
        threshold = self.grid_spacing * 1.5
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                n1 = nodes[i]
                n2 = nodes[j]
                dist = math.sqrt((n1[0]-n2[0])**2 + (n1[1]-n2[1])**2)
                if dist <= threshold:
                    line = LineString([n1, n2])
                    if self.polygon.contains(line):
                        self.graph.add_edge(n1, n2, weight=dist)
        return nodes

    def find_stair_nodes(self):
        if not self.grid_nodes:
            return []
        left = min(self.grid_nodes, key=lambda n: n[0])
        right = max(self.grid_nodes, key=lambda n: n[0])
        return [left, right]

    def init_agents(self):
        inv = self.building.inventory
        if not inv or not self.grid_nodes:
            return
        for i, student in enumerate(inv):
            floor = getattr(student, 'target_floor', random.randint(0, self.n_floor - 1))
            if floor >= self.n_floor:
                floor = self.n_floor - 1
            node = random.choice(self.grid_nodes)
            x = node[0] + random.uniform(-1.5, 1.5)
            y = node[1] + random.uniform(-1.5, 1.5)
            if not self.safe_polygon.contains(Point(x, y)):
                x = node[0] + random.uniform(-0.3, 0.3)
                y = node[1] + random.uniform(-0.3, 0.3)
            agent = InteriorAgent(agent_id=i, x=x, y=y, floor=floor, map_student=student)
            self.floors[floor].append(agent)

    def move_agent_to_floor(self, agent, new_floor):
        if agent in self.floors[agent.floor]:
            self.floors[agent.floor].remove(agent)
        agent.floor = new_floor
        if new_floor in self.floors:
            self.floors[new_floor].append(agent)

    def exit_building(self, agent):
        if agent in self.floors[agent.floor]:
            self.floors[agent.floor].remove(agent)
        if agent.map_student in self.building.inventory:
            self.building.inventory.remove(agent.map_student)

        stud = agent.map_student
        stud.is_hidden = False
        stud.x, stud.y = self.building.door_coords
        stud.start_x, stud.start_y = stud.x, stud.y
        stud.end_x, stud.end_y = stud.x, stud.y
        if self.building.is_on_fire and not stud.is_panicked:
            stud.become_panicked()

    def add_student_from_outside(self, student):
        if not self.grid_nodes:
            return
        floor = 0
        target = getattr(student, 'target_floor', random.randint(0, self.n_floor - 1))
        best_stair = self.stair_nodes[0] if self.stair_nodes else random.choice(self.grid_nodes)
        x, y = best_stair[0], best_stair[1]
        agent = InteriorAgent(agent_id=student.unique_id, x=x, y=y, floor=floor, map_student=student)
        agent.target_floor = target
        self.floors[floor].append(agent)

    def step(self):
        for floor, fc in getattr(self, 'fire_centers', {}).items():
            growth_speed = 0.01 + (fc['radius'] / 60.0) * 0.08
            fc['radius'] += growth_speed
            fc['radius'] = min(fc['radius'], 60.0)

            target_blobs = min(int(fc['radius'] * 8), 500)
            blobs_floor = [b for b in self.fire_blobs if b['floor'] == floor]

            attempts = 0
            while len(blobs_floor) < target_blobs and attempts < target_blobs * 5:
                attempts += 1
                ang = random.uniform(0, 2 * math.pi)
                dst = random.uniform(0, fc['radius'])
                bx = fc['x'] + math.cos(ang) * dst
                by = fc['y'] + math.sin(ang) * dst
                if self.polygon.contains(Point(bx, by)):
                    new_blob = {'x': bx, 'y': by, 'floor': floor}
                    self.fire_blobs.append(new_blob)
                    blobs_floor.append(new_blob)

            if len(blobs_floor) > target_blobs:
                other_floors = [b for b in self.fire_blobs if b['floor'] != floor]
                self.fire_blobs = other_floors + random.sample(blobs_floor, target_blobs)

        if self.fire_floors:
            self.fire_spread_timer += 1
            if self.fire_spread_timer > 600:
                self.fire_spread_timer = 0
                floors_to_add = []
                for f in list(self.fire_floors):
                    if f in self.fire_centers and self.fire_centers[f]['radius'] > 12.0:
                        if random.random() < 0.3:
                            if f < self.n_floor - 1 and (f+1) not in self.floors:
                                floors_to_add.append(f+1)
                            if f > 0 and  (f - 1) not in self.fire_floors:
                                floors_to_add.append(f-1)
                for new_f in floors_to_add:
                    self.set_fire_on_floor(new_f)

        for floor_agents in self.floors.values():
            for agent in list(floor_agents):
                agent.step(self.polygon, self.safe_polygon, self.grid_nodes, self.stair_nodes, self)

    def get_agents_on_floor(self, floor):
        return self.floors.get(floor, [])

    def get_stair_nodes(self):
        return self.stair_nodes