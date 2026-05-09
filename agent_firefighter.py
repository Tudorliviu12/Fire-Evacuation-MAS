from mesa import Agent
import networkx as nx
import osmnx as ox
import math
import random
from typing import TYPE_CHECKING
from config import FIREFIGHTER_STANDOFF, MIN_FIREFIGHTER_DIST, WATER_SPREAD_ANGLE, WATER_PARTICLES_PER_FRAME, WATER_PARTICLE_SPEED, WATER_EXTINGUISH_POWER, FIREFIGHTER_RETREAT_DIST, FIREFIGHTER_ADVANCE_DIST
if TYPE_CHECKING:
    from simulation_model import CampusModel

class Firefighter(Agent):
    def __init__(self, unique_id, model: 'CampusModel', start_x, start_y, angle_offset=0.0, truck=None):
        super().__init__(unique_id, model)
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
        self.is_walking_path = True
        self.is_returning = False

        self.target_x: float = self.x
        self.target_y: float = self.y

        self.compute_standoff_post()

    def compute_standoff_post(self):
        self.target_x = self.model.fire_center_x + math.cos(self.angle_offset) * (self.model.current_fire_radius + FIREFIGHTER_STANDOFF)
        self.target_y = self.model.fire_center_y + math.sin(self.angle_offset) * (self.model.current_fire_radius + FIREFIGHTER_STANDOFF)

    def is_too_close(self, x, y):
        for agent in self.model.active_agents_cache:
            if agent is self:
                continue
            if type(agent).__name__ in ('Firefighter', 'Firetruck'):
                if (x-agent.x)**2 + (y-agent.y)**2 <= MIN_FIREFIGHTER_DIST**2:
                    return True
        return False

    def move(self):
        if (not self.model.fire_started or self.is_returning) and self.truck:
            self.is_returning = True
            tx, ty = self.truck.x, self.truck.y
            dx = tx - self.x
            dy = ty - self.y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist <= 2.0:
                self.truck.firefighter_boarded(self)
                self.is_active = False
                if self in self.model.schedule.agents:
                    self.model.schedule.remove(self)
            else:
                speed = self.base_speed * 1.5
                self.x += (dx/dist) * speed
                self.y += (dy/dist) * speed
            return

        self.target_x = self.model.fire_center_x + math.cos(self.angle_offset) * (self.model.current_fire_radius + FIREFIGHTER_STANDOFF)
        self.target_y = self.model.fire_center_y + math.sin(self.angle_offset) * (self.model.current_fire_radius + FIREFIGHTER_STANDOFF)
        dist_to_post = math.sqrt((self.target_x-self.x)**2 + (self.target_y-self.y)**2)

        if dist_to_post<=1.5:
            self.shoot_water()
            return

        dir_x = (self.target_x - self.x)/max(0.1, dist_to_post)
        dir_y = (self.target_y - self.y) / max(0.1, dist_to_post)
        next_x = self.x + dir_x * self.current_speed
        next_y = self.y + dir_y * self.current_speed
        next_dist_to_fire = math.sqrt((next_x-self.model.fire_center_x)**2 + (next_y-self.model.fire_center_y)**2)
        if next_dist_to_fire < self.model.current_fire_radius + 2.5:
            next_x, next_y = self.x, self.y

        if self.is_too_close(next_x, next_y):
            next_x += random.uniform(-0.5, 0.5)
            next_y += random.uniform(-0.5, 0.5)

        self.x, self.y = next_x, next_y

    def return_to_truck(self):
        if self.truck is None:
            return
        tx, ty = self.truck.x, self.truck.y
        dx = tx - self.x
        dy = ty - self.y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist <= 2.0:
            self.truck.firefighter_boarded(self)
            self.is_active = False
            self.model.schedule.remove(self)
        else:
            speed = self.base_speed * 1.5
            self.x += (dx/dist)*speed
            self.y += (dy/dist)*speed

    def shoot_water(self):
        if self.burst_paused > 0:
            self.burst_paused -= 1
            return
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
            return

        dx = self.model.fire_center_x - self.x
        dy = self.model.fire_center_y - self.y
        base_angle = math.atan2(dy, dx)

        for _ in range(WATER_PARTICLES_PER_FRAME):
            angle = base_angle + random.uniform(-WATER_SPREAD_ANGLE, WATER_SPREAD_ANGLE)
            self.model.water_particles.append({
                'x': self.x,
                'y': self.y,
                'vx': math.cos(angle)*WATER_PARTICLE_SPEED,
                'vy': math.sin(angle)*WATER_PARTICLE_SPEED,
                'life': 60,
            })

        self.burst_count += 1
        if self.burst_count >= self.burst_max:
            self.burst_count = 0
            self.burst_paused = random.randint(18,35)
        else:
            self.shoot_cooldown = self.shoot_interval

    def step(self):
        self.move()
