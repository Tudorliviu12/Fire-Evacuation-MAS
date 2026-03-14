import mesa
from mesa.time import RandomActivation
from mesa import Model
from agent_student import Student
import random
import math

from config import MAX_SMOKE, WIND_ANGLE, FIRE_GROWTH_MIN, FIRE_GROWTH_MAX, MAX_FIRE_RADIUS_SOFT_CAP, SMOKE_SPEED, \
    SMOKE_GROWTH, SMOKE_LIFESPAN


class CampusModel(Model):
    def __init__(self, G_all, G_drive, nodes_proj, doors, n_students):
        super().__init__()
        self.G_all = G_all
        self.G_drive = G_drive
        self.nodes_proj = nodes_proj
        self.building_doors = doors
        self.schedule = mesa.time.RandomActivation(self)
        self.n_students = n_students
        self.fire_started = False
        self.fire_center_x = 0
        self.fire_center_y = 0
        self.current_fire_radius: float = 0.0
        self.fire_blobs = []
        self.safe_nodes = []
        self.smoke_blobs = []
        self.wind_angle = WIND_ANGLE

        all_nodes_ids = list(self.nodes_proj.index)
        for i in range(n_students):
            start_node = random.choice(all_nodes_ids)
            delay = 0
            a = Student(i, self, start_node, delay=delay, indoors=False)
            self.schedule.add(a)

    def ignite_fire(self, x, y):
        if self.fire_started:
            return
        print(f"Fire started at {x}, {y}")
        self.fire_started = True
        self.fire_center_x = x
        self.fire_center_y = y
        self.current_fire_radius = 1.5


    def step(self):
        to_remove = [a for a in self.schedule.agents if getattr(a, 'should_remove', False)]
        for agent in to_remove:
            self.schedule.remove(agent)

        if self.fire_started:
            growth = random.uniform(FIRE_GROWTH_MIN, FIRE_GROWTH_MAX)
            if self.current_fire_radius > MAX_FIRE_RADIUS_SOFT_CAP:
                growth = growth * 0.3
            self.current_fire_radius += growth
            target_blobs = min(int(self.current_fire_radius * 8), 500)

            while len(self.fire_blobs) < target_blobs:
                ang = random.uniform(0, 2*math.pi)
                dst = random.uniform(0, self.current_fire_radius)
                self.fire_blobs.append({
                    'x': self.fire_center_x + math.cos(ang)*dst,
                    'y': self.fire_center_y + math.sin(ang)*dst,
                })
                if len(self.fire_blobs) > target_blobs:
                    self.fire_blobs = self.fire_blobs[:target_blobs]

            smoke_chance = 0.1 if self.current_fire_radius < 5 else 0.4

            if len(self.smoke_blobs) < MAX_SMOKE and random.random() < smoke_chance:
                ang_spawn = random.uniform(0, 2*math.pi)
                dst_spawn = random.uniform(0, self.current_fire_radius*0.3)
                self.smoke_blobs.append({
                    'x': self.fire_center_x + math.cos(ang_spawn)*dst_spawn,
                    'y': self.fire_center_y + math.sin(ang_spawn)*dst_spawn,
                    'size': 5, 'age': 0,
                    'angle': self.wind_angle + random.uniform(-15, 15),
                })
            for i in range(len(self.smoke_blobs) -1, -1, -1):
                smoke = self.smoke_blobs[i]
                growth_factor = SMOKE_GROWTH * (self.current_fire_radius / 15.0)
                smoke['size'] += max(0.1, growth_factor)
                rad_b = math.radians(smoke['angle'])
                smoke['x'] += math.cos(rad_b) * SMOKE_SPEED
                smoke['y'] += math.sin(rad_b) * SMOKE_SPEED
                smoke['age'] += 1
                if smoke['age'] > SMOKE_LIFESPAN:
                    self.smoke_blobs.pop(i)


        self.schedule.step()


    def is_near_any_smoke(self, x, y):
        if not self.fire_started:
            return False
        for blob in self.smoke_blobs[::5]:
            d = math.sqrt((x-blob['x'])**2 + (y - blob['y'])**2)
            if d<blob['size']:
                return True
        return False
