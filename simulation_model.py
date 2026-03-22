import mesa
from mesa.time import RandomActivation
from mesa import Model
import osmnx as ox
from shapely import LineString

from agent_student import Student
import geopandas as gpd
from shapely.geometry import Point
import random
import math

from config import RAW_LOCATIONS, MAX_SMOKE, WIND_ANGLE, FIRE_GROWTH_MIN, FIRE_GROWTH_MAX, MAX_FIRE_RADIUS_SOFT_CAP, SMOKE_SPEED, \
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
        self.burned_edges = set()
        self.G_working = self.G_all.copy()

        self.hotspot_names = list(RAW_LOCATIONS.keys())
        self.hotspot_nodes = []
        self.hotspot_weights = []

        for name in self.hotspot_names:
            lat_raw, lon_raw, weight = RAW_LOCATIONS[name]
            p = gpd.GeoSeries([Point(lon_raw, lat_raw)], crs="EPSG:4326")
            p_proj = p.to_crs("EPSG:3857").iloc[0]

            node = ox.distance.nearest_nodes(self.G_all, p_proj.x, p_proj.y)
            self.hotspot_nodes.append(node)
            self.hotspot_weights.append(weight)

        self.dorm_nodes = []
        for coords in self.building_doors.values():
            try:
                node = ox.distance.nearest_nodes(self.G_all, coords[0], coords[1])
                self.dorm_nodes.append(node)
            except Exception as e:
                print(f"Error in building door {coords} - {e}\n")

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

    def block_fire_edges(self):
        if not self.fire_started or self.schedule.steps % 5 != 0:
            return

        fire_pt = Point(self.fire_center_x, self.fire_center_y)

        for u,v,k,data in self.G_all.edges(keys=True, data=True):
            if(u,v,k) in self.burned_edges:
                continue

            if 'geometry' in data:
                edge_geom = data['geometry']
            else:
                nx_u, ny_u = self.G_all.nodes[u]['x'], self.G_all.nodes[u]['y']
                nx_v, ny_v = self.G_all.nodes[v]['x'], self.G_all.nodes[v]['y']
                edge_geom = LineString([(nx_u, ny_u), (nx_v, ny_v)])

            dist = fire_pt.distance(edge_geom)
            if dist < self.current_fire_radius + 5.0:
                self.burned_edges.add((u,v,k))
                if self.G_working.has_edge(u,v,k):
                    self.G_working.remove_edge(u,v,key=k)

            nx_u = self.G_all.nodes[u]['x']
            ny_u = self.G_all.nodes[u]['y']
            nx_v = self.G_all.nodes[v]['x']
            ny_v = self.G_all.nodes[v]['y']
            mid_x = (nx_u + nx_v) / 2
            mid_y = (ny_u + ny_v) / 2
            dist = math.sqrt((mid_x - self.fire_center_x)**2 + (mid_y - self.fire_center_y)**2)
            if dist < self.current_fire_radius:
                self.burned_edges.add((u,v,k))
                if self.G_working.has_edge(u,v,k):
                    self.G_working.remove_edge(u,v,key=k)


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

        self.block_fire_edges()
        self.schedule.step()


    def is_near_any_smoke(self, x, y):
        if not self.fire_started:
            return False
        for blob in self.smoke_blobs[::5]:
            d = math.sqrt((x-blob['x'])**2 + (y - blob['y'])**2)
            if d<blob['size']:
                return True
        return False
