import random
import math

from networkx.algorithms.assortativity.pairs import node_degree_xy
from shapely.geometry import Polygon, Point

class InteriorAgent:
    def __init__(self, agent_id, x, y, floor, map_student=None):
        self.agent_id = agent_id
        self.x = x
        self.y = y
        self.floor = floor
        self.map_student = map_student
        self.vx = random.uniform(-0.3, 0.3)
        self.vy = random.uniform(-0.3, 0.3)
        self.rest_timer = random.randint(30, 120)
        self.is_moving = False
        self.move_timer = 0

        self.target_floor = getattr(map_student, 'target_floor', floor)
        self.is_exiting = False
        self.stair_timer = 0
        self.in_transit = False

    def step(self, polygon, grid_nodes, stair_nodes, building_grid):
        if self.stair_timer > 0:
            self.stair_timer -= 1
            if self.stair_timer <= 0:
                self.in_transit = False
                building_grid.move_agent_to_floor(self, self.target_floor)
                if self.is_exiting and self.target_floor == 0:
                    building_grid.exit_building(self)
                else:
                    self.is_moving = False
                    self.rest_timer = 0
                    if grid_nodes:
                        valid_nodes = [n for n in grid_nodes if (n[0]-self.x)**2 + (n[1]-self.y)**2 > 10.0]
                        if valid_nodes:
                            target = random.choice(valid_nodes)
                            dx = target[0] - self.x
                            dy = target[1] - self.y
                            dist = math.sqrt(dx*dx + dy*dy)
                            speed = random.uniform(0.8, 1.2)
                            self.vx = (dx/dist)*speed
                            self.vy = (dy/dist)*speed
                            self.move_timer = max(1, int(dist/speed))
                            self.is_moving = True
            return

        if self.floor != self.target_floor or self.is_exiting:
            if not stair_nodes:
                return
            best_stair = min(stair_nodes, key=lambda s: (self.x - s[0])**2 + (self.y - s[1])**2)
            dx = best_stair[0] - self.x
            dy = best_stair[1] - self.y
            dist = math.sqrt(dx*dx + dy*dy)

            if dist < 2.0:
                self.in_transit = True
                floor_difference = abs(self.floor - self.target_floor)
                if floor_difference == 0 and self.is_exiting:
                    self.stair_timer = 20
                else:
                    self.stair_timer = max(10, floor_difference * 20)
            else:
                speed = 1.0
                nx = self.x + (dx/dist)*speed
                ny = self.y + (dy/dist)*speed
                if polygon.contains(Point(nx, ny)):
                    self.x = nx
                    self.y = ny
                else:
                    closest_node = min(grid_nodes, key=lambda n: (self.x - n[0])**2 + (self.y - n[1])**2)
                    self.x += random.uniform(-0.5, 0.5)
                    self.y += random.uniform(-0.5, 0.5)
            return

        if self.rest_timer > 0:
            self.rest_timer -= 1
            if random.random() < 0.0005:
                self.is_exiting = True
                self.target_floor = 0
            elif random.random() < 0.001:
                self.target_floor = random.randint(0, building_grid.n_floor-1)
            return

        if not self.is_moving:
            if grid_nodes:
                target = random.choice(grid_nodes)
                dx = target[0] - self.x
                dy = target[1] - self.y
                dist = math.sqrt(dx*dx + dy*dy)
                if dist > 0.1:
                    speed = random.uniform(0.4, 1.2)
                    self.vx = (dx/dist)*speed
                    self.vy = (dy/dist)*speed
                    self.move_timer = max(1, int(dist/speed))
                    self.is_moving = True
            return

        nx = self.x + self.vx
        ny = self.y + self.vy

        if polygon.contains(Point(nx, ny)):
            self.x = nx
            self.y = ny
        else:
            self.is_moving = False
            self.move_timer = 0
            self.rest_timer = random.randint(10,40)

        self.move_timer -= 1
        if self.move_timer <= 0:
            self.is_moving = False
            self.rest_timer = random.randint(20,80)


class InteriorGrid:
    def __init__(self, polygon_coords_local, building, n_floor=5, grid_spacing=5.0):
        self.polygon = Polygon(polygon_coords_local)
        self.building = building
        self.n_floor = n_floor
        self.grid_spacing = grid_spacing
        self.grid_nodes = self.generate_grid()
        self.floors = {i: [] for i in range(n_floor)}
        self.init_agents()
        self.stair_nodes = self.find_stair_nodes()

    def generate_grid(self):
        minx, miny, maxx, maxy = self.polygon.bounds
        nodes = []
        x = minx + self.grid_spacing
        while x < maxx:
            y = miny + self.grid_spacing
            while y < maxy:
                pt = Point(x,y)
                if self.polygon.contains(pt):
                    nodes.append((x, y))
                y += self.grid_spacing
            x += self.grid_spacing
        return nodes

    def find_stair_nodes(self):
        if not self.grid_nodes:
            return []
        left = min(self.grid_nodes, key=lambda n: n[0])
        right = max(self.grid_nodes, key=lambda n: n[0])
        return [left, right]

    def init_agents(self):
        inv = self.building.inventory
        if not inv or not self.grid_nodes: return

        for i, student in enumerate(inv):
            floor = getattr(student, 'target_floor', random.randint(0, self.n_floor - 1))
            if floor >= self.n_floor: floor = self.n_floor - 1
            node = random.choice(self.grid_nodes)
            x = node[0] + random.uniform(-1.5, 1.5)
            y = node[1] + random.uniform(-1.5, 1.5)
            if not self.polygon.contains(Point(x,y)):
                x, y = node
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
        stud.choose_new_mission()

    def add_student_from_outside(self, student):
        if not self.grid_nodes: return
        floor = 0
        target = getattr(student, 'target_floor', random.randint(0, self.n_floor - 1))
        best_stair = self.stair_nodes[0] if self.stair_nodes else random.choice(self.grid_nodes)
        x, y = best_stair[0], best_stair[1]
        agent = InteriorAgent(agent_id = student.unique_id, x=x, y=y, floor=floor, map_student=student)
        agent.target_floor = target
        self.floors[floor].append(agent)

    def step(self):
        for floor_agents in self.floors.values():
            for agent in list(floor_agents):
                agent.step(self.polygon, self.grid_nodes, self.stair_nodes, self)

    def get_agents_on_floor(self, floor):
        return self.floors.get(floor, [])

    def get_stair_nodes(self):
        return self.stair_nodes