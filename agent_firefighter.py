from mesa import Agent
import networkx as nx
import osmnx as ox
import math
import random
from faker import Faker
from shapely.geometry import Point
from typing import TYPE_CHECKING
from config import FF_SHOOT_INTERVAL_MIN, FF_SHOOT_INTERVAL_MAX, FF_BURST_MAX_MIN, FF_BURST_MAX_MAX, FF_BURST_PAUSE_MIN, FF_BURST_PAUSE_MAX, FF_STAIR_TICKS_PER_FLOOR, FIREFIGHTER_STANDOFF, MIN_FIREFIGHTER_DIST, WATER_SPREAD_ANGLE, WATER_PARTICLES_PER_FRAME, WATER_PARTICLE_SPEED, WATER_EXTINGUISH_POWER, FIREFIGHTER_RETREAT_DIST, FIREFIGHTER_ADVANCE_DIST
from pathfinder import DStarLite
from interior_agent import InteriorAgent
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
        self.shoot_interval = random.randint(FF_SHOOT_INTERVAL_MIN, FF_SHOOT_INTERVAL_MAX)
        self.burst_count = 0
        self.burst_max = random.randint(FF_BURST_MAX_MIN, FF_BURST_MAX_MAX)
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
        self.indoor_state = None
        self.indoor_building = None
        self.indoor_agent = None
        self.indoor_target_floor = None
        self.post_computed = False

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

    def is_too_close(self, x, y, min_dist=None):
        if min_dist is None:
            min_dist = MIN_FIREFIGHTER_DIST
        for agent in self.model.active_agents_cache:
            if agent is self:
                continue
            if agent is self.truck:
                continue
            if type(agent).__name__ in ('Firefighter', 'Firetruck'):
                if (x - agent.x) ** 2 + (y - agent.y) ** 2 <= min_dist ** 2:
                    return True
        return False

    def move(self):
        if self.indoor_state is not None:
            self.move_indoor()
            return
        if self.model.fire_started and not self.is_returning:
            if self.check_enter_building():
                return

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

        if not getattr(self.model, 'interior_fire_only', False):
            dist_to_fire = math.sqrt(
                (self.x - self.model.fire_center_x) ** 2 + (self.y - self.model.fire_center_y) ** 2)
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
                self.post_computed = False
                self.pos_history = []
            else:
                self.shoot_water()
                return

        if not getattr(self, 'post_computed', False):
            self.compute_standoff_post()
            self.post_computed = True

        dist_to_post = math.sqrt((self.target_x - self.x) ** 2 + (self.target_y - self.y) ** 2)

        if dist_to_post <= 3.0:
            self.has_arrived = True
            self.post_computed = False
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

        if not hasattr(self, 'pos_history'):
            self.pos_history = []
        self.pos_history.append((self.x, self.y))
        if len(self.pos_history) > 25:
            self.pos_history.pop(0)

        if len(self.pos_history) >= 25 and not self.is_returning:
            old_x, old_y = self.pos_history[0]
            total_moved = math.sqrt((self.x - old_x) ** 2 + (self.y - old_y) ** 2)
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
            self.burst_paused = random.randint(FF_BURST_PAUSE_MIN, FF_BURST_PAUSE_MAX)
        else:
            self.shoot_cooldown = self.shoot_interval

    def check_enter_building(self):
        for b in self.model.buildings:
            if not getattr(b, 'interior_grid', None):
                continue
            ig = b.interior_grid
            if getattr(ig, 'fire_extinguished_indoors', False):
                continue
            if not ig.fire_floors:
                continue

            door_x, door_y = b.door_coords
            my_dist = math.sqrt((door_x - self.x) ** 2 + (door_y - self.y) ** 2)
            if my_dist <= 40.0:
                self.x = door_x
                self.y = door_y
                self.start_coords = b.door_coords
                self.start_indoor(b, ig)
                return True
        return False

    def start_indoor(self, building, ig):
        if not ig.fire_centers:
            return
        target_floor = max(ig.fire_centers.keys(), key=lambda f:ig.fire_centers[f]['radius'])
        self.indoor_building = building
        self.indoor_target_floor = target_floor
        self.indoor_state = 'entering'
        self.is_hidden = True

        stair = random.choice(ig.stair_nodes) if ig.stair_nodes else (ig.grid_nodes[0] if ig.grid_nodes else (50.0, 50.0))
        offset_x = random.uniform(-0.5, 0.5)
        offset_y = random.uniform(-0.5, 0.5)

        ia = InteriorAgent(
            agent_id = f"FF_{self.unique_id}",
            x=stair[0] + offset_x, y=stair[1] + offset_y,
            floor=0,
            map_student=None
        )
        ia.assigned_stair = stair
        ia.is_firefighter = True
        ia.firefighter_ref = self
        ia.is_exiting = False
        ia.target_floor = target_floor
        ia.is_aware_of_fire = True
        ig.floors[0].append(ia)
        self.indoor_agent = ia

    def move_indoor(self):
        ia = self.indoor_agent
        ig = self.indoor_building.interior_grid if self.indoor_building else None

        if ia is None or ig is None:
            self.indoor_state = None
            self.is_hidden = False
            return

        if self.indoor_state == 'entering':
            ia.target_floor = self.indoor_target_floor
            ia.is_exiting = False

            best_stair = getattr(ia, 'assigned_stair', None) or min(ig.stair_nodes, key=lambda s: (ia.x - s[0]) ** 2 + (ia.y - s[1]) ** 2)
            dx = best_stair[0] - ia.x
            dy = best_stair[1] - ia.y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist > 1.5:
                ia.x += (dx / dist) * 2.0
                ia.y += (dy / dist) * 2.0
            else:
                ia.x, ia.y = best_stair
                ia.in_transit = True
                floors_diff = abs(self.indoor_target_floor - ia.floor)
                ia.stair_timer = FF_STAIR_TICKS_PER_FLOOR * max(1, floors_diff)
                ia.stair_timer_total = ia.stair_timer
                ia.firefighter_climb_start_floor = ia.floor
                ia.firefighter_climb_target_floor = self.indoor_target_floor
                self.indoor_state = 'climbing'

        elif self.indoor_state == 'climbing':
            if ia.stair_timer > 0:
                ia.stair_timer -= 1
                return

            ia.in_transit = False
            target_f = getattr(ia, 'firefighter_climb_target_floor', self.indoor_target_floor)
            ig.move_agent_to_floor(ia, target_f)
            ia.path = []
            self.indoor_state = 'fighting'


        elif self.indoor_state == 'fighting':
            fc = ig.fire_centers.get(self.indoor_target_floor)
            if not fc:
                self.indoor_state = 'descending'
                ia.path = []
                return

            fx, fy = fc['x'], fc['y']
            dx_fire = fx - ia.x
            dy_fire = fy - ia.y
            dist_to_fire = max(0.1, math.sqrt(dx_fire ** 2 + dy_fire ** 2))
            if not hasattr(ia, 'fight_post') or ia.fight_post is None:
                angle_personal = (hash(str(self.unique_id)) % 360) * (math.pi / 180.0)
                safe_nodes = [n for n in ig.grid_nodes if n not in ig.burned_nodes and ig.graph.degree(n) > 0]
                if safe_nodes:
                    closest_cur = min(safe_nodes, key=lambda n: (n[0] - ia.x) ** 2 + (n[1] - ia.y) ** 2)
                    try:
                        reachable_set = nx.node_connected_component(ig.graph, closest_cur)
                        reachable = [n for n in safe_nodes if n in reachable_set]
                    except Exception:
                        reachable = safe_nodes
                    if not reachable:
                        reachable = safe_nodes
                    tx = fx + math.cos(angle_personal) * 7.0
                    ty = fy + math.sin(angle_personal) * 7.0
                    dest = min(reachable, key=lambda n: (n[0] - tx) ** 2 + (n[1] - ty) ** 2)
                    ia.fight_post = dest
                else:
                    ia.fight_post = None

            if dist_to_fire < 3.5:
                ia.fight_post = None
                if not ia.path:
                    safe_nodes = [n for n in ig.grid_nodes if n not in ig.burned_nodes]
                    if safe_nodes:
                        closest = min(safe_nodes, key=lambda n: (n[0] - ia.x) ** 2 + (n[1] - ia.y) ** 2)
                        far = max(safe_nodes, key=lambda n: (n[0] - fx) ** 2 + (n[1] - fy) ** 2)
                        try:
                            full = nx.shortest_path(ig.graph, closest, far)
                            ia.path = full[1:] if len(full) > 1 else [far]
                        except Exception:
                            ia.path = [far]

                if ia.path:
                    next_pt = ia.path[0]
                    dx = next_pt[0] - ia.x
                    dy = next_pt[1] - ia.y
                    d = max(0.1, math.sqrt(dx ** 2 + dy ** 2))
                    if d < 1.2:
                        ia.x, ia.y = next_pt
                        ia.path.pop(0)
                    else:
                        nx_ = ia.x + (dx / d) * 1.5
                        ny_ = ia.y + (dy / d) * 1.5
                        if ig.polygon.contains(Point(nx_, ny_)):
                            ia.x, ia.y = nx_, ny_
                        else:
                            ia.x, ia.y = next_pt
                            ia.path.pop(0)
                return

            if ia.fight_post is not None:
                dp = math.sqrt((ia.x - ia.fight_post[0]) ** 2 + (ia.y - ia.fight_post[1]) ** 2)
                if dp > 2.0:
                    if not ia.path:
                        safe_nodes = [n for n in ig.grid_nodes if n not in ig.burned_nodes]
                        closest = min(safe_nodes, key=lambda n: (n[0] - ia.x) ** 2 + (n[1] - ia.y) ** 2)
                        try:
                            full = nx.shortest_path(ig.graph, closest, ia.fight_post)
                            ia.path = full[1:] if len(full) > 1 else [ia.fight_post]
                        except Exception:
                            try:
                                reachable_set = nx.node_connected_component(ig.graph, closest)
                                reachable = [n for n in safe_nodes if n in reachable_set]
                            except Exception:
                                reachable = []
                            if reachable:
                                fallback = min(reachable, key=lambda n: (n[0] - ia.fight_post[0]) ** 2 + (
                                            n[1] - ia.fight_post[1]) ** 2)
                                ia.path = [fallback]
                                ia.fight_post = fallback
                            else:
                                ia.fight_post = None
                                ia.path = []
                    if ia.path:
                        next_pt = ia.path[0]
                        dx = next_pt[0] - ia.x
                        dy = next_pt[1] - ia.y
                        d = max(0.1, math.sqrt(dx ** 2 + dy ** 2))
                        if d < 1.2:
                            ia.x, ia.y = next_pt
                            ia.path.pop(0)
                        else:
                            nx_ = ia.x + (dx / d) * 1.5
                            ny_ = ia.y + (dy / d) * 1.5
                            if ig.polygon.contains(Point(nx_, ny_)):
                                ia.x, ia.y = nx_, ny_
                            else:
                                ia.x, ia.y = next_pt
                                ia.path.pop(0)
                                if not ia.path:
                                    ia.fight_post = None
                    return

            ia.path = []
            extinguish_rate = 0.015 + (fc['radius'] / 60.0) * 0.01
            fc['radius'] = max(0.0, fc['radius'] - extinguish_rate)
            if self.burst_paused > 0:
                self.burst_paused -= 1
            elif self.shoot_cooldown > 0:
                self.shoot_cooldown -= 1

            else:
                if not hasattr(ig, 'interior_water_particles'):
                    ig.interior_water_particles = []
                base_angle = math.atan2(dy_fire, dx_fire)
                exact_life = int(dist_to_fire / WATER_PARTICLE_SPEED) + random.randint(0, 2)
                for _ in range(5):
                    ang = base_angle + random.uniform(-WATER_SPREAD_ANGLE, WATER_SPREAD_ANGLE)
                    ig.interior_water_particles.append({
                        'x': ia.x, 'y': ia.y,
                        'vx': math.cos(ang) * WATER_PARTICLE_SPEED,
                        'vy': math.sin(ang) * WATER_PARTICLE_SPEED,
                        'floor': ia.floor, 'life': exact_life
                    })
                self.burst_count += 1

                if self.burst_count >= self.burst_max:
                    self.burst_count = 0
                    self.burst_paused = random.randint(8, 14)
                else:
                    self.shoot_cooldown = self.shoot_interval
            if not hasattr(ia, 'stuck_hist'):
                ia.stuck_hist = []
            ia.stuck_hist.append((ia.x, ia.y))

            if len(ia.stuck_hist) > 20:
                ia.stuck_hist.pop(0)
                old_x, old_y = ia.stuck_hist[0]
                if math.sqrt((ia.x - old_x) ** 2 + (ia.y - old_y) ** 2) < 1.5:
                    ia.fight_post = None
                    ia.path = []
                    ia.stuck_hist = []
                    snap_nodes = [n for n in ig.grid_nodes if n not in ig.burned_nodes and ig.graph.degree(n) > 0]
                    if snap_nodes:
                        snap = min(snap_nodes, key=lambda n: (n[0] - ia.x) ** 2 + (n[1] - ia.y) ** 2)
                        ia.x, ia.y = snap


            if fc['radius'] <= 0:
                ig.fire_floors.discard(self.indoor_target_floor)
                ig.fire_centers.pop(self.indoor_target_floor, None)
                ig.fire_blobs = [b for b in ig.fire_blobs if b['floor'] != self.indoor_target_floor]
                ig.fire_extinguished_indoors = True
                self.model.fire_started = False
                self.model.current_fire_radius = 0.0
                if self.truck:
                    for ff in self.truck.firefighters:
                        ff.is_returning = True
                self.indoor_state = 'descending'
                ia.path = []

        elif self.indoor_state == 'descending':
            best_stair = getattr(ia, 'assigned_stair', None) or min(ig.stair_nodes, key=lambda s: (ia.x - s[0]) ** 2 + (ia.y - s[1]) ** 2)
            dist_to_stair = math.sqrt((ia.x - best_stair[0]) ** 2 + (ia.y - best_stair[1]) ** 2)

            if dist_to_stair > 1.5:
                if not getattr(ia, 'path', None):
                    closest = min(ig.grid_nodes, key=lambda n: (n[0] - ia.x) ** 2 + (n[1] - ia.y) ** 2)
                    try:
                        full_path = nx.shortest_path(ig.graph, closest, best_stair)
                        ia.path = full_path[1:] if len(full_path) > 1 else [best_stair]
                    except:
                        ia.path = [best_stair]
                if ia.path:
                    next_pt = ia.path[0]
                    dx = next_pt[0] - ia.x
                    dy = next_pt[1] - ia.y
                    d = max(0.1, math.sqrt(dx * dx + dy * dy))
                    if d < 1.2:
                        ia.path.pop(0)
                    else:
                        nx_ = ia.x + (dx / d) * 2.0
                        ny_ = ia.y + (dy / d) * 2.0
                        if ig.polygon.contains(Point(nx_, ny_)):
                            ia.x, ia.y = nx_, ny_
                        else:
                            ia.x, ia.y = next_pt[0], next_pt[1]
                            ia.path.pop(0)

            else:
                ia.x, ia.y = best_stair
                ia.in_transit = True
                ia.stair_timer = 25 * max(1, ia.floor)
                ia.stair_timer_total = ia.stair_timer
                ia.firefighter_climb_start_floor = ia.floor
                ia.firefighter_climb_target_floor = 0
                self.indoor_state = 'descending_stairs'

        elif self.indoor_state == 'descending_stairs':
            if ia.stair_timer > 0:
                ia.stair_timer -= 1
                return
            ia.in_transit = False
            ig.move_agent_to_floor(ia, 0)
            self.indoor_state = 'exiting'

        elif self.indoor_state == 'exiting':
            self.exit_indoor()

        if self.indoor_state in ['entering', 'fighting', 'descending'] and ia and ig:
            if not hasattr(ia, 'global_stuck_hist'):
                ia.global_stuck_hist = []
            ia.global_stuck_hist.append((ia.x, ia.y))
            if len(ia.global_stuck_hist) > 40:
                old_x, old_y = ia.global_stuck_hist.pop(0)
                if math.sqrt((ia.x - old_x) ** 2 + (ia.y - old_y) ** 2) < 1.0:
                    ia.fight_post = None
                    ia.path = []
                    ia.global_stuck_hist = []
                    snap_nodes = [n for n in ig.grid_nodes if n not in ig.burned_nodes and ig.graph.degree(n) > 0]
                    if snap_nodes:
                        snap = min(snap_nodes, key=lambda n: (n[0] - ia.x) ** 2 + (n[1] - ia.y) ** 2)
                        ia.x, ia.y = snap
                    else:
                        ia.x, ia.y = ig.grid_center

    def is_too_close_indoor(self, ia, x, y, min_dist=2.0):
        ig = self.indoor_building.interior_grid if self.indoor_building else None
        if not ig:
            return False
        for floor_agents in ig.floors.values():
            for other_ia in floor_agents:
                if other_ia is ia:
                    continue
                if not getattr(other_ia, 'is_firefighter', False):
                    continue
                dist_sq = (x - other_ia.x) ** 2 + (y - other_ia.y) ** 2
                if dist_sq <= min_dist ** 2:
                    return True
        return False

    def switch_to_floor(self, ig, ia, new_floor):
        self.indoor_target_floor = new_floor
        ia.target_floor = new_floor
        current = ia.floor
        best_stair = min(ig.stair_nodes, key=lambda s: (ia.x-s[0])**2 + (ia.y-s[1])**2)
        ia.x, ia.y = best_stair
        ia.in_transit = True
        floors_diff = abs(new_floor - current)
        ia.stair_timer = FF_STAIR_TICKS_PER_FLOOR * max(1, floors_diff)
        ia.stair_timer_total = ia.stair_timer
        ia.firefighter_climb_start_floor = current
        ia.firefighter_climb_target_floor = new_floor
        self.indoor_state = 'climbing'

    def exit_indoor(self):
        ia = self.indoor_agent
        ig = self.indoor_building.interior_grid if self.indoor_building else None

        if ia and ig:
            for floor in ig.floors.values():
                if ia in floor:
                    floor.remove(ia)
            for queue in ig.stair_queues.values():
                if ia in queue:
                    queue.remove(ia)

        if self.indoor_building:
            self.x = self.indoor_building.door_coords[0]
            self.y = self.indoor_building.door_coords[1]

        self.is_hidden = False
        self.indoor_state = None
        self.indoor_building = None
        self.indoor_agent = None
        self.indoor_target_floor = None
        self.is_returning = True
        self.path = []
        self.edge_waypoints = []
        self.frames_current = 0
        self.frames_total = 1
        self.has_arrived = False

    def step(self):
        self.move()