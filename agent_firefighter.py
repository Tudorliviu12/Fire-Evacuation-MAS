from mesa import Agent
import networkx as nx
import osmnx as ox
import math
import random
from faker import Faker
from shapely.geometry import Point
from typing import TYPE_CHECKING
from config import FIREFIGHTER_STANDOFF, MIN_FIREFIGHTER_DIST, WATER_SPREAD_ANGLE, WATER_PARTICLES_PER_FRAME, WATER_PARTICLE_SPEED, WATER_EXTINGUISH_POWER, FIREFIGHTER_RETREAT_DIST, FIREFIGHTER_ADVANCE_DIST
from pathfinder import DStarLite
if TYPE_CHECKING:
    from simulation_model import CampusModel

fake = Faker('ro_RO')

class Firefighter(Agent):
    def __init__(self, unique_id, model: 'CampusModel', start_x, start_y, angle_offset=0.0, truck=None):
        super().__init__(unique_id, model)
        self.full_name = f"Firefighter {fake.last_name()}"
        self.model: 'CampusModel' = model
        self.x = start_x + random.uniform(-1.0, 1.0)
        self.y = start_y + random.uniform(-1.0, 1.0)
        self.angle_offset = angle_offset
        self.truck = truck
        self.shoot_cooldown = 0
        self.shoot_interval = random.randint(5,10)
        self.burst_count = 0
        self.burst_max = random.randint(3,6)
        self.burst_paused = 0
        self.base_speed = 1.0
        self.current_speed = self.base_speed
        self.water_particles = []
        self.is_active = True
        self.is_panicked = False
        self.is_dead = False
        self.is_returning = False
        self.stuck_timer = 0
        self.stuck_last_x = self.x
        self.stuck_last_y = self.y
        self.escape_node = None

        self.target_x: float = self.x
        self.target_y: float = self.y

        self.dstar = None
        self.dstar_goal = None
        self.path = []
        self.edge_waypoints = []
        self.frames_current = 0
        self.frames_total = 1
        self.start_x, self.start_y = self.x, self.y
        self.end_x, self.end_y = self.x, self.y
        self.has_arrived = False
        self.is_hidden = False
        self.tunnel_dx = 0.0
        self.tunnel_dy = 0.0
        self.tunnel_cooldown = 0
        self.pos_history = []
        self.spawn_cooldown = 30

        self.compute_standoff_post()
        self.init_path_to_post()

    def compute_standoff_post(self):
        ideal_x = self.model.fire_center_x + math.cos(self.angle_offset) * (self.model.current_fire_radius + FIREFIGHTER_STANDOFF)
        ideal_y = self.model.fire_center_y + math.sin(self.angle_offset) * (self.model.current_fire_radius + FIREFIGHTER_STANDOFF)

        if getattr(self.model, 'buildings_shape', None) is None:
            self.target_x, self.target_y = ideal_x, ideal_y
            return

        ideal_pt = Point(ideal_x, ideal_y)

        if not self.model.buildings_shape.contains(ideal_pt) and self.model.buildings_shape.distance(ideal_pt) > 2.0:
            self.target_x, self.target_y = ideal_x, ideal_y
            return

        for step in range(1,25):
            for sign in [1, -1]:
                test_angle = self.angle_offset + sign * (step* 0.26)
                tx = self.model.fire_center_x + math.cos(test_angle) * (self.model.current_fire_radius + FIREFIGHTER_STANDOFF)
                ty = self.model.fire_center_y + math.sin(test_angle) * (self.model.current_fire_radius + FIREFIGHTER_STANDOFF)

                pt = Point(tx, ty)
                if not self.model.buildings_shape.contains(pt) and self.model.buildings_shape.distance(pt) > 2.0:
                    self.angle_offset = test_angle
                    self.target_x = tx
                    self.target_y = ty
                    return
        self.target_x, self.target_y = ideal_x, ideal_y

    def init_path_to_post(self):
        try:
            start_n = ox.distance.nearest_nodes(self.model.G_all, self.x, self.y)
            goal_n = ox.distance.nearest_nodes(self.model.G_all, self.target_x, self.target_y)
        except Exception:
            return

        if start_n == goal_n:
            return

        try:
            self.dstar = DStarLite(self.model.G_all, start_n, goal_n)
            self.dstar_goal = goal_n
            self.dstar.compute_shortest_path()
            full_path = self.dstar.get_path()
            if full_path and len(full_path) > 1:
                self.path = full_path[1:]
                return
            self.path = nx.shortest_path(self.model.G_all, start_n, goal_n, weight='length')[1:]
        except Exception:
            try:
                start_n = ox.distance.nearest_nodes(self.model.G_all, self.x, self.y)
                goal_n = ox.distance.nearest_nodes(self.model.G_all, self.target_x, self.target_y)
                self.path = nx.shortest_path(self.model.G_all, start_n, goal_n, weight='length')[1:]
            except Exception:
                self.path = []

    def is_too_close(self, x, y):
        for agent in self.model.active_agents_cache:
            if agent is self:
                continue
            if agent is self.truck:
                continue
            if type(agent).__name__ in ('Firefighter', 'Firetruck'):
                if (x-agent.x)**2 + (y-agent.y)**2 <= MIN_FIREFIGHTER_DIST**2:
                    return True
        return False

    def move(self):
        if (not self.model.fire_started or self.is_returning) and self.truck:
            if not self.is_returning:
                self.pos_history = []
                self.path = []
                self.edge_waypoints = []
                self.frames_current = 0
                self.frames_total = 1
                self.start_x, self.start_y = self.x, self.y
                self.end_x, self.end_y = self.x, self.y
                self.is_returning = True
            self.return_to_truck()
            return

        dist_to_fire = math.sqrt((self.x - self.model.fire_center_x) ** 2 + (self.y - self.model.fire_center_y) ** 2)

        if dist_to_fire < self.model.current_fire_radius + 3.0:
            dx = self.x - self.model.fire_center_x
            dy = self.y - self.model.fire_center_y
            norm = max(0.1, math.sqrt(dx * dx + dy * dy))
            self.x += (dx / norm) * self.current_speed * 2
            self.y += (dy / norm) * self.current_speed * 2
            return

        if self.has_arrived:
            dist_to_fire = math.sqrt((self.x - self.model.fire_center_x)**2 + (self.y - self.model.fire_center_y)**2)
            ideal_dist = self.model.current_fire_radius + FIREFIGHTER_STANDOFF
            if dist_to_fire > ideal_dist + 15.0:
                self.has_arrived = False
                self.pos_history = []
            else:
                self.shoot_water()
                return

        self.compute_standoff_post()
        dist_to_post = math.sqrt((self.target_x - self.x) ** 2 + (self.target_y - self.y) ** 2)

        if dist_to_post <= 3.0:
            self.has_arrived = True
            return

        next_x, next_y = self.model.move_avoid_buildings(
            self.x, self.y, self.target_x, self.target_y,
            self.current_speed, self.model.buildings_shape
        )
        if self.spawn_cooldown > 0:
            self.spawn_cooldown -= 1
            self.x, self.y = next_x, next_y
        else:
            if not self.is_too_close(next_x, next_y):
                self.x, self.y = next_x, next_y

        if not self.is_too_close(next_x, next_y):
            self.x, self.y = next_x, next_y

        if not hasattr(self, 'pos_history'):
            self.pos_history = []
        self.pos_history.append((self.x, self.y))
        if len(self.pos_history) > 25:
            self.pos_history.pop(0)

        if len(self.pos_history) >= 25 and not self.is_returning:
            old_x, old_y = self.pos_history[0]
            total_moved = math.sqrt((self.x - old_x)**2 + (self.y - old_y)**2)
            if total_moved < 1.0:
                self.pos_history = []
                self.teleport_around_fire()


    def teleport_around_fire(self):
        buildings_shape = getattr(self.model, 'buildings_shape', None)
        for attempt in range(36):
            angle = self.angle_offset + (attempt * math.pi / 18)
            r = self.model.current_fire_radius + FIREFIGHTER_STANDOFF + random.uniform(0, 5)
            tx = self.model.fire_center_x + math.cos(angle) * r
            ty = self.model.fire_center_y + math.sin(angle) * r
            if buildings_shape is None or not buildings_shape.contains(Point(tx, ty)):
                self.x = tx
                self.y = ty
                self.has_arrived = True
                self.is_hidden = False
                return
        self.x = self.model.fire_center_x + math.cos(self.angle_offset) * (self.model.current_fire_radius + FIREFIGHTER_STANDOFF + 10)
        self.y = self.model.fire_center_y + math.sin(self.angle_offset) * (self.model.current_fire_radius + FIREFIGHTER_STANDOFF + 10)
        self.has_arrived = True
        self.is_hidden = False

    def return_to_truck(self):
        if self.truck is None:
            return
        tx, ty = self.truck.x, self.truck.y
        dx = tx - self.x
        dy = ty - self.y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist <= 2.0:
            self.truck.firefighter_boarded(self)
            self.is_active = False
            self.model.schedule.remove(self)
            return

        if not self.path:
            try:
                start_n = ox.distance.nearest_nodes(self.model.G_all, self.x, self.y)
                end_n = ox.distance.nearest_nodes(self.model.G_all, tx, ty)
                self.path = nx.shortest_path(self.model.G_all, start_n, end_n, weight='length')[1:]
            except Exception:
                self.path = []

        if self.path:
            next_node = self.path[0]
            node_data = self.model.nodes_proj.loc[next_node]
            nx_, ny_ = node_data.geometry.x, node_data.geometry.y
            dist_to_node = math.sqrt((nx_ - self.x) ** 2 + (ny_ - self.y) ** 2)
            if dist_to_node < 1.5:
                self.path.pop(0)
            else:
                speed = self.base_speed * 1.5
                self.x, self.y = self.model.move_avoid_buildings(
                    self.x, self.y, nx_, ny_, speed, self.model.buildings_shape
                )
        else:
            speed = self.base_speed * 1.5
            self.x, self.y = self.model.move_avoid_buildings(
                self.x, self.y, tx, ty, speed, self.model.buildings_shape
            )

    def shoot_water(self):
        if self.burst_paused > 0:
            self.burst_paused -= 1
            return
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
            return

        dx = self.model.fire_center_x - self.x
        dy = self.model.fire_center_y - self.y
        dist_to_fire = math.sqrt(dx * dx + dy * dy)
        base_angle = math.atan2(dy, dx)
        exact_life = int(dist_to_fire / WATER_PARTICLE_SPEED) + random.randint(0,2)

        for _ in range(WATER_PARTICLES_PER_FRAME):
            angle = base_angle + random.uniform(-WATER_SPREAD_ANGLE, WATER_SPREAD_ANGLE)
            self.model.water_particles.append({
                'x': self.x,
                'y': self.y,
                'vx': math.cos(angle)*WATER_PARTICLE_SPEED,
                'vy': math.sin(angle)*WATER_PARTICLE_SPEED,
                'life': exact_life
            })

        self.burst_count += 1
        if self.burst_count >= self.burst_max:
            self.burst_count = 0
            self.burst_paused = random.randint(18,35)
        else:
            self.shoot_cooldown = self.shoot_interval

    def step(self):
        self.move()