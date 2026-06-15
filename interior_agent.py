import math
import random
import networkx as nx
from shapely import Point
from config import STUDENT_STAIR_TICKS, INTERIOR_FIRE_DEATH_TICKS, STAIR_CAPACITY, INTERIOR_CALM_SPEED, INTERIOR_EVAC_SPEED, INTERIOR_SLOW_EVAC_SPEED, INTERIOR_QUEUE_MOVE

class InteriorAgent:
    def __init__(self, agent_id, x, y, floor, map_student=None):
        self.agent_id = agent_id
        self.x = x
        self.y = y
        self.floor = floor
        self.map_student = map_student
        self.rest_timer = random.randint(0, 60)
        self.path = []
        self.target_floor = getattr(map_student, 'target_floor', floor)
        self.is_exiting = False
        self.stair_timer = 0
        self.in_transit = False
        self._waiting_at_stair = False
        self.idle_timer = 0

        self.is_aware_of_fire = False
        self.panic_wander_dir = None
        self.panic_wander_timer = 0
        self.path_failed = False
        self.building_nodes = None
        self.fire_death_timer = 0
        self.alarm_response_timer = -1
        self.is_firefighter = False
        self.cached_queue_idx = 0
        self.queue_idx_timer = 0

    def move(self, speed):
        if getattr(self.map_student, 'is_dead', False):
            return False
        if not self.path:
            return True
        target = self.path[0]
        dx = target[0] - self.x
        dy = target[1] - self.y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist <= speed:
            self.x, self.y = target
            self.path.pop(0)
        else:
            self.x += (dx / dist) * speed
            self.y += (dy / dist) * speed
        return False

    def check_fire_awareness(self, building_grid):
        if self.is_aware_of_fire:
            return

        for floor in [self.floor - 1, self.floor, self.floor + 1]:
            fc_floor = building_grid.fire_centers.get(floor)

            if fc_floor:
                if floor != self.floor and fc_floor['radius'] <= 20.0:
                    continue
                dist = math.sqrt((self.x - fc_floor['x']) ** 2 + (self.y - fc_floor['y']) ** 2)
                base_range = 5.0 + (fc_floor['radius'] * 1.5)
                detection_range = base_range if floor == self.floor else fc_floor['radius'] * 0.5
                if dist < detection_range and (floor == self.floor or fc_floor['radius'] > 20.0):
                    self.trigger_evacuation(building_grid)
                    return

    def trigger_evacuation(self, building_grid):
        self.is_aware_of_fire = True
        if self.map_student:
            self.map_student.is_panicked = True
            self.map_student.color = 'red'
            campus_model = getattr(building_grid.building, 'model', None)
            if campus_model and not getattr(campus_model, 'alarm_triggered', False):
                campus_model.alarm_triggered = True
                campus_model.truck_timer = random.randint(160, 250)
                campus_model.hero_name = self.map_student.full_name
                self.map_student.is_calling_112 = True
                self.map_student.call_timer = 70

        self.is_exiting = True
        self.target_floor = 0
        self.path = []
        self._waiting_at_stair = False
        self.path_failed = False

    def update_stair_queue(self, best_stair, safe_polygon, building_grid):
        queue = building_grid.stair_queues.setdefault(best_stair, [])
        if self not in queue:
            if getattr(self, 'is_firefighter', False):
                queue.insert(0, self)
            else:
                queue.append(self)
            self.cached_queue_idx = 0 if getattr(self, 'is_firefighter', False) else len(queue) - 1
            self.queue_idx_timer = 0

        if self.queue_idx_timer <= 0:
            try:
                self.cached_queue_idx = queue.index(self)
            except ValueError:
                self.cached_queue_idx = 0
            self.queue_idx_timer = 10
        else:
            self.queue_idx_timer -= 1
        idx = self.cached_queue_idx
        sx, sy = best_stair

        if idx == 0:
            target_x, target_y = sx, sy
        else:
            steps = getattr(building_grid.building, 'model', None).schedule.steps if hasattr(building_grid.building, 'model') else random.randint(0, 100)
            base_angle = (hash(str(self.agent_id)) % 360) * (math.pi / 180.0)
            dynamic_angle = base_angle + math.sin(steps * 0.05 + self.agent_id) * 0.4
            row = (idx - 1) // 4
            dist_offset = 2.2 + row * 1.5 + random.uniform(-0.2, 0.2)
            target_x = sx + math.cos(dynamic_angle) * dist_offset
            target_y = sy + math.sin(dynamic_angle) * dist_offset

        ddx = target_x - self.x
        ddy = target_y - self.y
        dist = math.sqrt(ddx * ddx + ddy * ddy)

        if dist > 0.3:
            nx_ = self.x + (ddx / dist) * INTERIOR_QUEUE_MOVE
            ny_ = self.y + (ddy / dist) * INTERIOR_QUEUE_MOVE
            if safe_polygon.contains(Point(nx_, ny_)):
                self.x, self.y = nx_, ny_
        else:
            jx = self.x + random.uniform(-0.6, 0.6)
            jy = self.y + random.uniform(-0.6, 0.6)
            if safe_polygon.contains(Point(jx, jy)):
                self.x, self.y = jx, jy

    def panic_wander(self, polygon):
        if self.panic_wander_timer <= 0:
            self.panic_wander_dir = random.uniform(0, 2 * math.pi)
            self.panic_wander_timer = random.randint(15, 45)
        self.panic_wander_timer -= 1

        nx_ = self.x + math.cos(self.panic_wander_dir) * 0.4
        ny_ = self.y + math.sin(self.panic_wander_dir) * 0.4

        if polygon.contains(Point(nx_, ny_)):
            self.x, self.y = nx_, ny_
        else:
            self.panic_wander_dir = self.panic_wander_dir + math.pi
            self.panic_wander_timer = 0

            if not polygon.contains(Point(self.x, self.y)):
                bn = getattr(self, 'building_nodes', None)
                nearest_node = min(bn, key=lambda n: (self.x - n[0]) ** 2 + (self.y - n[1]) ** 2) if bn else None
                if nearest_node:
                    self.x, self.y = nearest_node

    @property
    def display_floor(self):
        if not self.in_transit:
            return self.floor

        start_f = getattr(self, 'firefighter_climb_start_floor', None)
        target_f = getattr(self, 'firefighter_climb_target_floor', None)
        if start_f is not None and target_f is not None:
            total = getattr(self, 'stair_timer_total', None)
            if total is None:
                floors_diff = max(1, abs(target_f - start_f))
                total = 20 * floors_diff
            if total > 0:
                progress = 1.0 - (self.stair_timer / total)
                return round(start_f + (target_f - start_f) * progress)
            return target_f

        total = getattr(self, 'stair_timer_total', 35.0)
        progress = 1.0 - (self.stair_timer / total) if total > 0 else 1.0
        diff = self.target_floor - self.floor
        return round(self.floor + diff * progress)

    def step(self, polygon, safe_polygon, grid_nodes, stair_nodes, building_grid):
        self.building_nodes = grid_nodes

        if getattr(self, 'is_firefighter', False):
            return

        if getattr(self.map_student, 'is_dead', False):
            return

        fc = building_grid.fire_centers.get(self.floor)
        if fc:
            dist = math.sqrt((self.x - fc['x'])**2 + (self.y - fc['y'])**2)
            if dist < fc['radius'] * 0.6:
                self.fire_death_timer += 1

                campus_model = getattr(self.map_student, 'model', None)
                death_threshold = INTERIOR_FIRE_DEATH_TICKS
                if campus_model and getattr(campus_model, 'interior_fire_death_ticks_override', None) is not None:
                    death_threshold = campus_model.interior_fire_death_ticks_override

                if self.fire_death_timer > death_threshold:
                    if self.map_student:
                        self.map_student.is_dead = True
                        if self.map_student.current_building:
                            if self.map_student in self.map_student.current_building.inventory:
                                self.map_student.current_building.inventory.remove(self.map_student)

                        building = self.map_student.current_building
                        if building and hasattr(building, 'polygon_coords') and building.polygon_coords:
                            xs = [p[0] for p in building.polygon_coords]
                            ys = [p[1] for p in building.polygon_coords]
                            min_x, max_x = min(xs), max(xs)
                            min_y, max_y = min(ys), max(ys)

                            global_x = min_x + (self.x / 100.0) * (max_x - min_x)
                            global_y = min_y + (self.y / 100.0) * (max_y - min_y)
                        else:
                            if building:
                                global_x, global_y = building.door_coords
                            else:
                                global_x, global_y = self.x, self.y

                        if self.map_student.model:
                            b_name = building.name if building else "Unknown"
                            self.map_student.model.death_log.append({
                                'x': global_x,
                                'y': global_y,
                                'floor': self.floor,
                                'building': b_name
                            })

                        if campus_model and hasattr(campus_model, 'death_log'):
                            b_name = "Unknown"
                            if getattr(self.map_student, 'current_building', None):
                                b_name = self.map_student.current_building.name
                            campus_model.death_log.append({
                                'x': self.x,
                                'y': self.y,
                                'floor': self.floor,
                                'building': b_name
                            })
                    if self.in_transit and stair_nodes:
                        self.in_transit = False
                        best_stair = min(stair_nodes, key=lambda s: (self.x - s[0]) ** 2 + (self.y - s[1]) ** 2)
                        if best_stair in building_grid.stair_occupancy:
                            building_grid.stair_occupancy[best_stair] = max(0, building_grid.stair_occupancy[
                                best_stair] - 1)

                    for q in building_grid.stair_queues.values():
                        if self in q:
                            q.remove(self)
                    return
            else:
                self.fire_death_timer = max(0, self.fire_death_timer - 1)

        self.check_fire_awareness(building_grid)

        if getattr(building_grid, 'fire_alarm_active', False):
            if not self.is_aware_of_fire:
                self.is_aware_of_fire = True
                if self.map_student:
                    self.map_student.informed_by = "Fire Alarm"
                    self.map_student.color = 'red'
                    self.map_student.is_panicked = True

            if not self.is_exiting:
                if self.alarm_response_timer == -1:
                    campus_model = getattr(self.map_student, 'model', None)
                    if campus_model and getattr(campus_model, 'alarm_response_mode', 'realistic') == 'ideal':
                        self.alarm_response_timer = 0
                    else:
                        val = random.random()
                        if val < 0.3:
                            self.alarm_response_timer = random.randint(0, 20)
                        elif val < 0.7:
                            self.alarm_response_timer = random.randint(40, 100)
                        else:
                            self.alarm_response_timer = random.randint(120, 250)

                if self.alarm_response_timer > 0:
                    self.alarm_response_timer -= 1
                elif self.alarm_response_timer == 0:
                    self.trigger_evacuation(building_grid)
                    self.alarm_response_timer -= 1

        if getattr(building_grid, 'mass_evacuation', False) and not self.is_exiting:
            self.trigger_evacuation(building_grid)

        if self.is_aware_of_fire and self.map_student:
            self.map_student.color = 'red'
            self.map_student.is_panicked = True

        if self.stair_timer > 0:
            self.stair_timer -= 1
            if self.stair_timer <= 0:
                self.in_transit = False
                self._waiting_at_stair = False

                best_stair = min(stair_nodes, key=lambda s: (self.x - s[0]) ** 2 + (self.y - s[1]) ** 2)
                building_grid.stair_occupancy[best_stair] -= 1

                building_grid.move_agent_to_floor(self, self.target_floor)
                if self.is_exiting and self.target_floor == 0:
                    building_grid.exit_building(self)
                else:
                    self.path = []
            return

        if self.is_exiting:
            if not stair_nodes:
                if not getattr(self, 'is_firefighter', False):
                    self.panic_wander(polygon)
                return

            best_stair = min(stair_nodes, key=lambda s: (self.x - s[0]) ** 2 + (self.y - s[1]) ** 2)
            dist_to_stair = math.sqrt((best_stair[0] - self.x) ** 2 + (best_stair[1] - self.y) ** 2)

            if dist_to_stair < 9.0 or self._waiting_at_stair:
                self._waiting_at_stair = True
                self.path = []

                self.update_stair_queue(best_stair, safe_polygon, building_grid)

                cooldown = getattr(building_grid, 'stair_cooldown', {}).get(best_stair, 0)
                queue = building_grid.stair_queues.get(best_stair, [])
                current_on_stairs = building_grid.stair_occupancy.get(best_stair, 0)

                if cooldown <= 0 and queue and queue[0] == self and current_on_stairs < STAIR_CAPACITY:
                    queue.remove(self)
                    self._waiting_at_stair = False
                    self.in_transit = True
                    self.stair_timer = STUDENT_STAIR_TICKS

                    self.x, self.y = best_stair
                    building_grid.stair_occupancy[best_stair] += 1
                    building_grid.stair_cooldown[best_stair] = 5
                return

            if not self.path:
                if self.path_failed:
                    self.panic_wander(polygon)
                    if random.random() < 0.04:
                        self.path_failed = False
                    return

                closest = building_grid.get_nearest_node(self.x, self.y)
                if self.is_exiting:
                    if hasattr(building_grid, 'evac_paths') and closest in building_grid.evac_paths:
                        self.path = building_grid.evac_paths[closest][1:]
                        if not self.path:
                            self.path = [closest]
                    else:
                        best_stair = min(stair_nodes, key=lambda s: (self.x - s[0]) ** 2 + (self.y - s[1]) ** 2)
                        try:
                            full_path = building_grid.get_cached_path(closest, best_stair)
                            self.path = full_path if full_path else [best_stair]
                        except Exception:
                            self.path = [best_stair]
                else:
                    target = random.choice(grid_nodes)
                    try:
                        full_path = building_grid.get_cached_path(closest, target)
                        self.path = full_path if full_path else [target]
                    except Exception:
                        self.path = [target]

            speed = INTERIOR_EVAC_SPEED if (self.map_student and self.map_student.is_panicked) else INTERIOR_SLOW_EVAC_SPEED
            self.move(speed)
            return

        if self.idle_timer > 0:
            self.idle_timer -= 1
            return

        if random.random() < 0.005:
            self.idle_timer = random.randint(100, 500)
            return

        if random.random() < 0.002:
            if random.random() < 0.6:
                self.target_floor = random.randint(0, building_grid.n_floor - 1)
                self.path = []
            else:
                self.is_exiting = True
                self.target_floor = 0

        if self.rest_timer > 0:
            self.rest_timer -= 1
            return

        if not self.path:
            target = random.choice(grid_nodes)
            closest = building_grid.get_nearest_node(self.x, self.y)
            try:
                full_path = building_grid.get_cached_path(closest, target)
                self.path = full_path if full_path else [target]
            except Exception:
                self.path = [target]

        if self.move(INTERIOR_CALM_SPEED):
            self.rest_timer = random.randint(30, 120)

        if not polygon.contains(Point(self.x, self.y)):
            nearest = min(grid_nodes, key=lambda n: (self.x - n[0]) ** 2 + (self.y - n[1]) ** 2)
            self.x, self.y = nearest