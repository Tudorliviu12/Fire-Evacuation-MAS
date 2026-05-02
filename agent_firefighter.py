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
    def __init__(self, unique_id, model: 'CampusModel', start_x, start_y, angle_offset=0.0):
        super().__init__(unique_id, model)
        self.model: 'CampusModel' = model
        self.x = start_x + math.cos(angle_offset) * 3.0
        self.y = start_y + math.sin(angle_offset) * 3.0
        self.angle_offset = angle_offset
        self.shoot_cooldown = 0
        self.shoot_interval = random.randint(8,14)
        self.burst_count = 0
        self.burst_max = random.randint(3,6)
        self.burst_paused = 0
        self.base_speed = 0.6
        self.current_speed = self.base_speed
        self.water_particles = []
        self.is_active = True
        self.is_panicked = False
        self.is_dead = False

        self.target_x: float = self.x
        self.target_y: float = self.y

        self.compute_standoff_post()


    def compute_standoff_post(self):
        if not self.model.fire_started:
            self.target_x, self.target_y = self.x, self.y
            return
        standoff_dist = self.model.current_fire_radius + FIREFIGHTER_STANDOFF
        self.target_x = self.model.fire_center_x + math.cos(self.angle_offset) * standoff_dist
        self.target_y = self.model.fire_center_y + math.sin(self.angle_offset) * standoff_dist

    def is_too_close(self, x, y):
        for agent in self.model.schedule.agents:
            if agent is self:
                continue
            agent_type = type(agent).__name__
            if agent_type in ('Firefighter', 'Firetruck'):
                d = math.sqrt((x-agent.x)**2 + (y-agent.y)**2)
                if d <= MIN_FIREFIGHTER_DIST:
                    return True
        return False

    def move(self):
        if not self.model.fire_started:
            return
        self.compute_standoff_post()
        dist_to_post = math.sqrt((self.target_x - self.x)**2 + (self.target_y - self.y)**2)

        if dist_to_post <= 1.0:
            self.shoot_water()
            return

        dir_x = (self.target_x - self.x) / dist_to_post
        dir_y = (self.target_y - self.y) / dist_to_post
        next_x = self.x + dir_x * self.current_speed
        next_y = self.y + dir_y * self.current_speed

        if self.is_too_close(next_x, next_y):
            next_x += random.uniform(-0.5, 0.5)
            next_y += random.uniform(-0.5, 0.5)
        self.x = next_x
        self.y = next_y

    def shoot_water(self):
        if not hasattr(self.model, 'water_particles'):
            self.model.water_particles = []

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
