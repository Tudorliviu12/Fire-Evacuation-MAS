from mesa import Agent
import networkx as nx
import osmnx as ox
import math
import random
from typing import TYPE_CHECKING
from agent_firefighter import Firefighter
if TYPE_CHECKING:
    from simulation_model import CampusModel

class Firetruck(Agent):
    def __init__(self, unique_id, model: 'CampusModel', start_node):
        super().__init__(unique_id, model)
        self.model: 'CampusModel' = model
        node_data = self.model.nodes_proj.loc[start_node]
        self.x, self.y = node_data.geometry.x, node_data.geometry.y
        self.home_x, self.home_y = self.x, self.y
        self.start_x, self.start_y = self.x, self.y
        self.end_x, self.end_y = self.x, self.y
        self.base_speed = 4.0
        self.current_speed = self.base_speed
        self.path = []
        self.return_path = []
        self.frames_current = 0
        self.frames_total = 1
        self.has_arrived = False
        self.is_returning = False
        self.firefighters = []
        self.boarded_count = 0
        self.troops_deployed = False
        self.assigned_angle = self.assign_angle()
        self.staging_x, self.staging_y = self.fire_stay_position()
        self.calculate_route_to_fire()

    def build_safe_Gdrive(self, base_graph):
        fire_radius = max(self.model.current_fire_radius + 5.0, 10.0)
        safe = base_graph.copy()
        to_remove = [
            n for n in safe.nodes()
            if math.sqrt((safe.nodes[n].get('x', 0) - self.model.fire_center_x)**2 + (safe.nodes[n].get('y', 0) - self.model.fire_center_y)**2) < fire_radius
        ]
        safe.remove_nodes_from(to_remove)
        return safe

    def nearest_node_in_graph(self, graph, x, y):
        try:
            n = ox.distance.nearest_nodes(self.model.G_all, x, y)
            if n in graph:
                return n
        except Exception:
            pass
        best_n, best_d = None, float('inf')
        for n in graph.nodes():
            try:
                row = self.model.nodes_proj.loc[n]
                nx_x, ny_y = row.geometry.x, row.geometry.y
            except KeyError:
                nx_x = graph.nodes[n].get('x', None)
                ny_y = graph.nodes[n].get('y', None)
                if nx_x is None:
                    continue
            d = (nx_x - x) ** 2 + (ny_y - y) ** 2
            if d < best_d:
                best_d = d
                best_n = n
        return best_n

    def calculate_route_to_fire(self):
        safe_graph = self.build_safe_Gdrive(self.model.G_drive)
        if len(safe_graph.nodes) == 0:
            self.path = []
            self.has_arrived = True
            return

        start_n = ox.distance.nearest_nodes(self.model.G_drive, self.x, self.y)
        if start_n not in safe_graph:
            start_n = self.nearest_node_in_graph(safe_graph, self.x, self.y)
        if start_n is None:
            self.path = []
            self.has_arrived = True
            return

        target_n = self.nearest_node_in_graph(safe_graph, self.staging_x, self.staging_y)
        if target_n is None:
            self.path = []
            self.has_arrived = True
            return

        if target_n in self.model.nodes_proj.index:
            self.staging_x = self.model.nodes_proj.loc[target_n].geometry.x
            self.staging_y = self.model.nodes_proj.loc[target_n].geometry.y

        try:
            node_path = nx.shortest_path(safe_graph, start_n, target_n, weight='length')
            self.path = self.build_waypoint(safe_graph, node_path)
            return
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

        try:
            safe_all = self.build_safe_Gdrive(self.model.G_all)
            start_all = ox.distance.nearest_nodes(self.model.G_all, self.x, self.y)
            if start_all not in safe_all:
                start_all = self.nearest_node_in_graph(safe_all, self.x, self.y)
            target_all = self.nearest_node_in_graph(safe_all, self.staging_x, self.staging_y)

            if start_all and target_all and nx.has_path(safe_all, start_all, target_all):
                node_path = nx.shortest_path(safe_all, start_all, target_all, weight='length')
                self.path = self.build_waypoint(safe_all, node_path)
                if target_all in self.model.nodes_proj.index:
                    self.staging_x = self.model.nodes_proj.loc[target_all].geometry.x
                    self.staging_y = self.model.nodes_proj.loc[target_all].geometry.y
                return
        except Exception:
            pass
        self.path = []
        self.has_arrived = True

    def build_waypoint(self, graph, node_path):
        waypoints = []
        for i in range(len(node_path) - 1):
            u = node_path[i]
            v = node_path[i + 1]
            ux = graph.nodes[u].get('x', 0)
            uy = graph.nodes[u].get('y', 0)
            vx = graph.nodes[v].get('x', 0)
            vy = graph.nodes[v].get('y', 0)

            try:
                edge_data = graph.get_edge_data(u, v)
                if edge_data is None:
                    waypoints.append((vx, vy))
                    continue
                key = list(edge_data.keys())[0]
                data = edge_data[key]

                if 'geometry' in data:
                    coords = list(data['geometry'].coords)
                    dist_first = math.sqrt((ux - coords[0][0])**2 + (uy-coords[0][1])**2)
                    dist_last = math.sqrt((ux - coords[-1][0])**2 + (uy-coords[-1][1])**2)
                    if dist_last < dist_first:
                        coords.reverse()
                    waypoints.extend(coords[1:])
                else:
                    waypoints.append((vx, vy))

            except Exception as e:
                waypoints.append((vx, vy))
        return waypoints

    def calculate_route_to_home(self):
        start_n = ox.distance.nearest_nodes(self.model.G_drive, self.x, self.y)
        target_n = ox.distance.nearest_nodes(self.model.G_drive, self.home_x, self.home_y)
        try:
            node_path = nx.shortest_path(self.model.G_drive, start_n, target_n, weight='length')
            self.return_path = self.build_waypoint(self.model.G_drive, node_path)
        except:
            try:
                start_n = ox.distance.nearest_nodes(self.model.G_all, self.x, self.y)
                target_n = ox.distance.nearest_nodes(self.model.G_all, self.home_x, self.home_y)
                node_path = nx.shortest_path(self.model.G_all, start_n, target_n, weight='length')
                self.return_path = self.build_waypoint(self.model.G_all, node_path)
            except:
                self.return_path = []

    def firefighter_boarded(self, ff):
        self.boarded_count += 1
        if self.boarded_count >= len(self.firefighters):
            self.calculate_route_to_home()
            self.path = self.return_path
            self.frames_current = self.frames_total
            self.is_returning = True

    def spawn_firefighters(self):
        num_firefighters = random.randint(3,4)
        arc_spread = 2.0
        start_angle = self.assigned_angle - (arc_spread / 2)
        step_angle = arc_spread / max(1, (num_firefighters - 1))

        for i in range(num_firefighters):
            angle = start_angle + i * step_angle
            ff = Firefighter(
                f"FF_{self.unique_id}_{i}",
                self.model,
                self.x, self.y,
                angle_offset=angle,
                truck=self
            )
            self.firefighters.append(ff)
            self.model.schedule.add(ff)

    def find_best_position(self, truck_angle):
        occupied = self.model.occupied_fire_angles
        if not occupied:
            return truck_angle
        best_angle = truck_angle
        best_min_dist = -1

        for candidate_degree in range(0, 360, 45):
            candidate = math.radians(candidate_degree)
            min_dist = min(abs(math.atan2(math.sin(candidate - occ), math.cos(candidate - occ))) for occ in occupied)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_angle = candidate
        return best_angle

    def assign_angle(self):
        if not hasattr(self.model, 'occupied_fire_angles'):
            self.model.occupied_fire_angles = []
        for angle_degree in range(0, 360, 45):
            angle_rad = math.radians(angle_degree)
            too_close = any(
                abs(math.atan2(math.sin(angle_rad - occ), math.cos(angle_rad - occ))) < math.radians(30)
                for occ in self.model.occupied_fire_angles
            )
            if not too_close:
                self.model.occupied_fire_angles.append(angle_rad)
                return angle_rad
        return self.find_best_position(random.uniform(0, 2*math.pi))

    def fire_stay_position(self):
        standoff = 20.0
        sx = self.model.fire_center_x + math.cos(self.assigned_angle)*standoff
        sy = self.model.fire_center_y + math.sin(self.assigned_angle)*standoff
        return sx, sy

    def check_traffic(self):
        panicked_in_way = 0
        for agent in self.model.active_agents_cache:
            if not getattr(agent, 'is_hidden', False) and getattr(agent, 'is_panicked', False):
                dx = self.x - agent.x
                dy = self.y - agent.y
                if dx*dx + dy*dy < 400.0:
                    panicked_in_way += 1
                    if panicked_in_way > 6:
                        break
        in_smoke = self.model.is_near_any_smoke(self.x, self.y)
        if in_smoke:
            self.current_speed = self.base_speed * 0.5
        elif panicked_in_way > 6:
            self.current_speed = self.base_speed * 0.7
        else:
            self.current_speed = self.base_speed

    def move(self):
        if self.is_returning and not self.path:
            dist_home = math.sqrt((self.x - self.home_x)**2 + (self.y - self.home_y)**2)
            if dist_home < 5.0 or not self.return_path:
                self.model.schedule.remove(self)
                return

        if self.has_arrived and not self.is_returning:
            return

        self.check_traffic()

        if not self.is_returning:
            dist_to_fire = math.sqrt((self.x - self.model.fire_center_x)**2 + (self.y - self.model.fire_center_y)**2)
            safe_stop_dist = self.model.current_fire_radius + 5.0
            dist_to_staging = math.sqrt((self.x-self.staging_x)**2 + (self.y-self.staging_y)**2)

            too_close_to_truck = False
            for other in self.model.schedule.agents:
                if type(other).__name__ == ('Firetruck') and other is not self:
                    d = math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
                    if d < 15.0:
                        too_close_to_truck = True
                        break

            if dist_to_fire <= safe_stop_dist or dist_to_staging <= 5.0 or too_close_to_truck or (self.frames_current >= self.frames_total and not self.path):
                if not self.troops_deployed:
                    self.troops_deployed = True
                    self.has_arrived = True
                    self.path = []
                    self.spawn_firefighters()
                return

        if self.frames_current >= self.frames_total:
            if not self.path:
                if self.is_returning:
                    self.model.schedule.remove(self)
                else:
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