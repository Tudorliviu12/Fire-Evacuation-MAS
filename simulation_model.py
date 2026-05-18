import mesa
from mesa.time import RandomActivation
from mesa import Model
import osmnx as ox
from shapely import LineString
from agent_student import Student
from agent_firetruck import Firetruck
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union
import random
import math
from building import Building
from config import RAW_LOCATIONS, MAX_SMOKE, WIND_ANGLE, FIRE_GROWTH_MIN, FIRE_GROWTH_MAX, MAX_FIRE_RADIUS_SOFT_CAP, SMOKE_SPEED, SMOKE_GROWTH, SMOKE_LIFESPAN, WATER_EXTINGUISH_POWER


class CampusModel(Model):
    def __init__(self, G_all, G_drive, nodes_proj, buildings_gdf, doors, n_students):
        super().__init__()
        self.G_all = G_all
        self.G_drive = G_drive
        self.nodes_proj = nodes_proj
        self.buildings_gdf = buildings_gdf
        self.building_doors = doors
        self.schedule = mesa.time.RandomActivation(self)
        self.n_students = n_students
        self.fire_started = False
        self.fire_ever_started = False
        self.fire_center_x = 0
        self.fire_center_y = 0
        self.current_fire_radius: float = 0.0
        self.fire_blobs = []
        self.safe_nodes = []
        self.smoke_blobs = []
        self.water_particles = []
        self.wind_angle = WIND_ANGLE
        self.burned_edges = set()
        self.alarm_triggered = False
        self.truck_dispatched = False
        self.truck_timer = 0
        self.hero_name = ""
        self.fire_strength = 1
        self.trucks_max_needed = 1
        self.truck_count = 0
        self.next_truck_timer = 0
        self.ticks_fire_growing = 0
        self.ticks_since_last_truck = 0
        self.fire_growth_rate = FIRE_GROWTH_MIN
        self.G_working = self.G_all.copy()
        self.occupied_fire_angles = []
        self.active_agents_cache = []
        self.pending_alerts = []
        self.hotspot_names = list(RAW_LOCATIONS.keys())
        self.hotspot_nodes = []
        self.hotspot_weights = []
        self.buildings_shape = unary_union(buildings_gdf.geometry) if not buildings_gdf.empty else None

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

        self.buildings = []
        self.buildings_weights = []

        for idx, row in self.buildings_gdf.iterrows():
            door_coords = self.building_doors[idx]
            door_node = ox.distance.nearest_nodes(self.G_all, door_coords[0], door_coords[1])
            area = row.geometry.area

            name = row.get('nume_Camin', f"Camin_T{idx+1}")
            b = Building(name=name, door_node=door_node, door_coords=door_coords, area=area)
            b.osm_polygon = row.geometry
            self.buildings.append(b)
            self.buildings_weights.append(area)

        for i, name in enumerate(self.hotspot_names):
            node = self.hotspot_nodes[i]
            node_data = self.nodes_proj.loc[node]
            door_coords = (node_data.geometry.x, node_data.geometry.y)

            area_weight = self.hotspot_weights[i] * 400
            b = Building(name=name, door_node=node, door_coords=door_coords, area=area_weight)
            self.buildings.append(b)
            self.buildings_weights.append(area_weight)

        for i in range(n_students):
            if random.random() < 0.1:
                start_node = random.choice(all_nodes_ids)
                a = Student(i, self, start_node, delay=0, indoors=False)
                self.schedule.add(a)
            else:
                if random.random() < 0.6:
                    dorm_indices = list(range(22))
                    dorm_weights = self.buildings_weights[:22]
                    chosen_idx = random.choices(dorm_indices, weights=dorm_weights, k=1)[0]
                else:
                    chosen_idx = random.choices(range(len(self.buildings)), weights=self.buildings_weights, k=1)[0]

                chosen_building = self.buildings[chosen_idx]
                a = Student(i, self, start_node=None, delay=0, indoors=True, building_idx=chosen_idx)
                a.is_hidden = True
                a.waiting_timer = random.randint(300,5000)
                a.current_building = chosen_building
                chosen_building.inventory.append(a)
                self.schedule.add(a)

    def snap_to_nearest_hotspot(self, x, y):
        best_dist = float('inf')
        best_building = None
        for building in self.buildings:
            if building.name not in RAW_LOCATIONS:
                continue
            _, _, weight = RAW_LOCATIONS[building.name]
            snap_radius = 10 + weight * 0.5
            dist = math.sqrt((building.door_coords[0] - x)**2 + (building.door_coords[1] - y)**2)
            if dist < snap_radius and dist < best_dist:
                best_dist = dist
                best_building = building

        if best_building is not None:
            best_building.is_on_fire = True
            return (x,y)
        return (x,y)

    def ignite_fire(self, x, y):
        if self.fire_ever_started:
            return
        print(f"Fire started at {x}, {y}")
        self.fire_ever_started = True
        self.fire_strength = random.randint(1,4)
        self.trucks_max_needed = min(3, self.fire_strength + 1)
        self.truck_count = 0
        self.next_truck_timer = 0
        self.ticks_fire_growing = 0
        self.ticks_since_last_truck = 0
        self.fire_growth_rate = FIRE_GROWTH_MIN + (FIRE_GROWTH_MAX - FIRE_GROWTH_MIN) * (self.fire_strength - 1) / 4.0
        self.fire_started = True
        self.fire_center_x = x
        self.fire_center_y = y
        self.current_fire_radius = 1.5

    def block_fire_edges(self):
        if not self.fire_started or self.schedule.steps % 5 != 0:
            return

        fire_pt = Point(self.fire_center_x, self.fire_center_y)
        check_radius = self.current_fire_radius + 10.0

        for u,v,k,data in self.G_all.edges(keys=True, data=True):
            if(u,v,k) in self.burned_edges:
                continue

            nx_u = self.G_all.nodes[u]['x']
            ny_u = self.G_all.nodes[u]['y']
            nx_v = self.G_all.nodes[v]['x']
            ny_v = self.G_all.nodes[v]['y']
            mid_x = (nx_u + nx_v) / 2
            mid_y = (ny_u + ny_v) / 2
            dx = mid_x - self.fire_center_x
            dy = mid_y - self.fire_center_y

            if dx*dx + dy*dy > (check_radius*check_radius):
                continue

            if 'geometry' in data:
                edge_geom = data['geometry']
            else:
                nx_u, ny_u = self.G_all.nodes[u]['x'], self.G_all.nodes[u]['y']
                nx_v, ny_v = self.G_all.nodes[v]['x'], self.G_all.nodes[v]['y']
                edge_geom = LineString([(nx_u, ny_u), (nx_v, ny_v)])

            dist = fire_pt.distance(edge_geom)
            if dist < self.current_fire_radius + 5.0:
                if (u,v,k) not in self.burned_edges:
                    self.burned_edges.add((u,v,k))
                    if self.G_working.has_edge(u,v,key=k):
                        self.G_working.remove_edge(u,v,key=k)
                        self.notify_agents_edge_burned(u,v)

    def check_buildings_fire(self):
        if not self.fire_started:
            return
        fire_pt = Point(self.fire_center_x, self.fire_center_y)
        for building in self.buildings:
            if not building.is_on_fire:
                if building.osm_polygon is not None:
                    if building.osm_polygon.contains(fire_pt):
                        building.is_on_fire = True
                    else:
                        dist = fire_pt.distance(building.osm_polygon)
                        if dist<(self.current_fire_radius + 5.0):
                            building.is_on_fire = True
                else:
                    dist = math.sqrt((building.door_coords[0] - self.fire_center_x)**2 + (building.door_coords[1] - self.fire_center_y)**2)
                    if dist < self.current_fire_radius + 30.0:
                        building.is_on_fire = True
            building.evacuate_step()

    def notify_agents_edge_burned(self, u, v):
        for agent in self.active_agents_cache:
            if type(agent).__name__ == 'Student':
                if not getattr(agent, 'is_dead', False) and not getattr(agent, 'is_hidden', False):
                    agent.notify_edge_burned(u, v)

    def step(self):
        to_remove = [a for a in self.schedule.agents if getattr(a, 'should_remove', False)]
        self.active_agents_cache = [
            a for a in self.schedule.agents
            if getattr(a, 'is_active', False) and not getattr(a, 'is_dead', False)
        ]
        for agent in to_remove:
            self.schedule.remove(agent)
            if not self.fire_started:
                chosen_idx = random.choices(range(len(self.buildings)), weights=self.buildings_weights, k=1)[0]
                chosen_building = self.buildings[chosen_idx]
                new_id = self.schedule.steps * 10000 + random.randint(0,9999)
                new_agent = Student(new_id, self, start_node=None, delay=0, indoors=True, building_idx = chosen_idx)
                new_agent.is_hidden = True
                new_agent.waiting_timer = random.randint(500,3000)
                new_agent.current_building = chosen_building
                chosen_building.inventory.append(new_agent)
                self.schedule.add(new_agent)

        if self.fire_started:
            if self.alarm_triggered:
                if not self.truck_dispatched:
                    self.truck_timer -= 1
                    if self.truck_timer <= 0:
                        self.truck_dispatched = True
                        self.truck_count = 1
                        self.ticks_since_last_truck = 0
                        entry_node = self.safe_nodes[0]
                        entry_x = self.nodes_proj.loc[entry_node].geometry.x
                        entry_y = self.nodes_proj.loc[entry_node].geometry.y
                        auto_entry = ox.distance.nearest_nodes(self.G_drive, entry_x, entry_y)
                        truck = Firetruck("TRUCK_1", self, auto_entry)
                        self.schedule.add(truck)

                elif self.truck_count < self.trucks_max_needed:
                    trucks_arrived = [
                        a for a in self.schedule.agents
                        if type(a).__name__ == 'Firetruck' and getattr(a, 'has_arrived', False)
                    ]
                    if not trucks_arrived:
                        self.ticks_since_last_truck = 0
                    else:
                        self.ticks_since_last_truck += 1
                        if self.ticks_fire_growing > 50 and self.ticks_since_last_truck > 100:
                            self.truck_count += 1
                            self.ticks_since_last_truck = 0
                            self.ticks_fire_growing = 0
                            entry_node = self.safe_nodes[0]
                            entry_x = self.nodes_proj.loc[entry_node].geometry.x
                            entry_y = self.nodes_proj.loc[entry_node].geometry.y
                            auto_entry = ox.distance.nearest_nodes(self.G_drive, entry_x, entry_y)
                            truck = Firetruck(f"TRUCK_{self.truck_count}", self, auto_entry)
                            self.schedule.add(truck)

            growth = self.fire_growth_rate + random.uniform(-0.01, 0.01)
            if self.current_fire_radius > MAX_FIRE_RADIUS_SOFT_CAP:
                growth = growth * 0.3

            total_damage = 0.0
            remaining = []
            hit_positions = []
            if hasattr(self, 'water_particles'):
                for p in self.water_particles:
                    p['x'] += p['vx']
                    p['y'] += p['vy']
                    p['life'] -= 1
                    dist_to_fire = math.sqrt((p['x'] - self.fire_center_x)**2 + (p['y'] - self.fire_center_y)**2)

                    if self.current_fire_radius < 5.0:
                        hit_radius = max(2.0, self.current_fire_radius * 1.5)
                        small_fire_boost = max(1.0, 5.0/max(0.5, self.current_fire_radius))
                    else:
                        hit_radius = max(1.5, self.current_fire_radius * 0.8)
                        small_fire_boost = 1.0

                    hit = dist_to_fire <= hit_radius
                    dead = p['life'] <= 0
                    if hit:
                        damage = (WATER_EXTINGUISH_POWER * small_fire_boost) / max(1.0, self.fire_strength * 0.3)
                        total_damage += damage
                        hit_positions.append((p['x'], p['y']))
                    elif not dead:
                        remaining.append(p)
                self.water_particles = remaining

            change = growth - total_damage
            if change > 0:
                self.ticks_fire_growing += 1
            else:
                self.ticks_fire_growing = max(0, self.ticks_fire_growing - 1)

            self.current_fire_radius += change
            self.current_fire_radius = max(0.0, self.current_fire_radius)

            if hit_positions:
                extinguish_radius = 3.0
                blobs_to_keep = []
                for blob in self.fire_blobs:
                    extinct = False
                    for bx, by in hit_positions:
                        if math.sqrt((blob['x'] - bx)**2 + (blob['y'] - by)**2) <= extinguish_radius:
                            extinct = True
                            break
                    if not extinct:
                        blobs_to_keep.append(blob)
                self.fire_blobs = blobs_to_keep

            target_blobs = min(int(self.current_fire_radius * 8), 500)
            attempts = 0
            while len(self.fire_blobs) < target_blobs and attempts < target_blobs * 5:
                attempts += 1
                ang = random.uniform(0, 2*math.pi)
                dst = random.uniform(0, self.current_fire_radius)
                bx = self.fire_center_x + math.cos(ang)*dst
                by = self.fire_center_y + math.sin(ang)*dst
                pt = Point(bx, by)
                if not any(b.osm_polygon is not None and b.osm_polygon.contains(pt) for b in self.buildings):
                    self.fire_blobs.append({'x': bx, 'y': by})
            if len(self.fire_blobs) > target_blobs:
                self.fire_blobs = random.sample(self.fire_blobs, target_blobs)

            if self.current_fire_radius <= 0.3:
                self.current_fire_radius = 0.0
                self.fire_started = False
                self.fire_blobs = []
                self.water_particles = []
                self.smoke_blobs = []
                for b in self.buildings:
                    b.is_on_fire = False

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

            for i in range(len(self.smoke_blobs) - 1, -1, -1):
                smoke = self.smoke_blobs[i]
                growth_factor = min(SMOKE_GROWTH *(self.current_fire_radius/15.0), SMOKE_GROWTH*3)
                smoke['size'] += max(0.1, growth_factor)
                rad_b = math.radians(smoke['angle'])
                smoke['x'] += math.cos(rad_b)*SMOKE_SPEED
                smoke['y'] += math.sin(rad_b)*SMOKE_SPEED
                smoke['age'] += 1
                if smoke['age'] > SMOKE_LIFESPAN:
                    self.smoke_blobs.pop(i)

        self.block_fire_edges()
        if self.fire_started and self.schedule.steps % 10 == 0:
            self.check_buildings_fire()
        if self.pending_alerts:
            still_pending = []
            for (tick, student, sender_name) in self.pending_alerts:
                if self.schedule.steps >= tick:
                    if not student.is_dead and not student.is_aware:
                        student.informed_by = sender_name
                        student.become_panicked()
                else:
                    still_pending.append((tick, student, sender_name))
            self.pending_alerts = still_pending

        for b in self.buildings:
            b.evacuate_step()
            if getattr(b, 'interior_grid', None) is not None:
                b.interior_grid.step()

        self.schedule.step()

    def is_near_any_smoke(self, x, y):
        if not self.fire_started:
            return False
        for blob in self.smoke_blobs[::5]:
            d = math.sqrt((x-blob['x'])**2 + (y - blob['y'])**2)
            if d<blob['size']:
                return True
        return False

    def move_avoid_buildings(self, x, y, tx, ty, speed, buildings_shape):
        dx = tx - x
        dy = ty - y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 0.1:
            return x,y
        new_nx = x + (dx/dist)*speed
        new_ny = y + (dy/dist)*speed
        if buildings_shape and buildings_shape.contains(Point(new_nx, new_ny)):
            base_angle = math.atan2(dy, dx)
            for i in [0.3, -0.3, 0.6, -0.6, 0.9, -0.9, math.pi/2, -math.pi/2]:
                alt_angle = base_angle + i
                alt_x = x + math.cos(alt_angle)*speed
                alt_y = y + math.sin(alt_angle)*speed
                if not buildings_shape.contains(Point(alt_x, alt_y)):
                    return alt_x, alt_y
            return x, y
        return new_nx, new_ny
