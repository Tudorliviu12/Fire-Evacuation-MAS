import pandas as pd
import random
import os
import config
from map_loader import load_campus_map
from simulation_model import CampusModel

SCENARIO_TO_RUN = 3
N_RUNS_BIG = 300
N_RUNS_SMALL = 50
MAX_TICKS = 6000
OUTPUT_DIRECTORY = "results"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

print("Loading campus map")
G_all, G_drive, nodes, edges, buildings, doors, safe = load_campus_map()
print("Map loaded")

if SCENARIO_TO_RUN == 1:
    print(f"Scenario {SCENARIO_TO_RUN} started: {N_RUNS_SMALL} iterations on day vs night evacuation in dorm T17")
    results = []
    t17_index = 16
    MAX_TICKS_S2 = 10000

    import config

    orig_evac = config.INTERIOR_EVAC_SPEED
    orig_slow = config.INTERIOR_SLOW_EVAC_SPEED
    orig_stair = config.STUDENT_STAIR_TICKS

    for i in range(N_RUNS_SMALL):
        print(f"Daytime Run no. {i + 1} out of {N_RUNS_SMALL}")

        run_base_delay = random.randint(50, 250)

        config.INTERIOR_EVAC_SPEED = random.uniform(0.35, 0.45)
        config.INTERIOR_SLOW_EVAC_SPEED = random.uniform(0.20, 0.30)
        config.STUDENT_STAIR_TICKS = random.randint(30, 45)

        model = CampusModel(G_all, G_drive, nodes, buildings, doors,
                            n_students=200, alarm_response_mode='ideal',
                            interior_fire_death_ticks_override=5000)
        model.safe_nodes = safe
        for j, (idx, row) in enumerate(buildings.iterrows()):
            if j < 22:
                model.buildings[j].polygon_coords = list(row.geometry.exterior.coords)

        t17 = model.buildings[t17_index]
        for a in model.schedule.agents:
            if type(a).__name__ == 'Student':
                a.home_dorm = t17.name
                a.current_building = t17
                a.indoors = True

        t17.init_interior()

        gn = t17.interior_grid.grid_nodes
        cx = sum(n[0] for n in gn) / len(gn)
        cy = sum(n[1] for n in gn) / len(gn)
        cn = min(gn, key=lambda n: (n[0] - cx) ** 2 + (n[1] - cy) ** 2)
        t17.interior_grid.set_fire_on_floor(0, x=cn[0], y=cn[1])

        t17.interior_grid.fire_alarm_active = True

        for floor_agents in t17.interior_grid.floors.values():
            for agent in floor_agents:
                agent.alarm_response_timer = run_base_delay + random.randint(50, 200)

        evacuation_time = MAX_TICKS_S2
        for tick in range(MAX_TICKS_S2):
            model.step()
            alive_inside = sum(1 for a in t17.inventory if not getattr(a, 'is_dead', False))
            if alive_inside == 0:
                evacuation_time = tick
                break
        results.append({'run_id': i, 'time_of_day': 'Day', 'evacuation_time': evacuation_time})

    for i in range(N_RUNS_SMALL):
        print(f"Nighttime Run no. {i + 1} out of {N_RUNS_SMALL}")

        run_base_delay = random.randint(300, 500)
        config.INTERIOR_EVAC_SPEED = random.uniform(0.20, 0.30)
        config.INTERIOR_SLOW_EVAC_SPEED = random.uniform(0.10, 0.20)
        config.STUDENT_STAIR_TICKS = random.randint(55, 75)

        model = CampusModel(G_all, G_drive, nodes, buildings, doors,
                            n_students=600, alarm_response_mode='realistic',
                            interior_fire_death_ticks_override=5000)
        model.safe_nodes = safe
        for j, (idx, row) in enumerate(buildings.iterrows()):
            if j < 22:
                model.buildings[j].polygon_coords = list(row.geometry.exterior.coords)

        t17 = model.buildings[t17_index]
        for a in model.schedule.agents:
            if type(a).__name__ == 'Student':
                a.home_dorm = t17.name
                a.current_building = t17
                a.indoors = True

        t17.init_interior()

        gn = t17.interior_grid.grid_nodes
        cx = sum(n[0] for n in gn) / len(gn)
        cy = sum(n[1] for n in gn) / len(gn)
        cn = min(gn, key=lambda n: (n[0] - cx) ** 2 + (n[1] - cy) ** 2)
        t17.interior_grid.set_fire_on_floor(0, x=cn[0], y=cn[1])

        t17.interior_grid.fire_alarm_active = True

        for floor_agents in t17.interior_grid.floors.values():
            for agent in floor_agents:
                agent.alarm_response_timer = run_base_delay + random.randint(150, 400)

        evacuation_time = MAX_TICKS_S2
        for tick in range(MAX_TICKS_S2):
            model.step()
            alive_inside = sum(1 for a in t17.inventory if not getattr(a, 'is_dead', False))
            if alive_inside == 0:
                evacuation_time = tick
                break
        results.append({'run_id': i, 'time_of_day': 'Night', 'evacuation_time': evacuation_time})

    config.INTERIOR_EVAC_SPEED = orig_evac
    config.INTERIOR_SLOW_EVAC_SPEED = orig_slow
    config.STUDENT_STAIR_TICKS = orig_stair

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIRECTORY, "results_scenario_2_T17_day_vs_night.csv"), index=False)
    print(f"Scenario {SCENARIO_TO_RUN} finished")


elif SCENARIO_TO_RUN == 2:
    print(f"Scenario {SCENARIO_TO_RUN} started: {N_RUNS_SMALL} iterations on ideal vs realistic evacuation")
    all_data = []
    t17_index = 16

    for mode in ['ideal', 'realistic']:
        for i in range(N_RUNS_SMALL):
            print(f"{mode.upper()} Run no. {i+1} out of {N_RUNS_SMALL}")
            model = CampusModel(G_all, G_drive, nodes, buildings, doors, n_students=200, alarm_response_mode=mode)
            model.safe_nodes = safe
            for j, (idx, row) in enumerate(buildings.iterrows()):
                if j < 22:
                    model.buildings[j].polygon_coords = list(row.geometry.exterior.coords)

            t17 = model.buildings[t17_index]
            t17.inventory.clear()

            students_to_move = [a for a in model.schedule.agents if type(a).__name__ == 'Student']
            for student in students_to_move:
                for b in model.buildings:
                    if student in b.inventory:
                        b.inventory.remove(student)

                t17.inventory.append(student)
                student.current_building = t17
                student.home_dorm = t17.name
                student.indoors = True
                student.is_hidden = True
                student.waiting_timer = random.randint(100, 500)


            t17.init_interior()
            t17.interior_grid.set_fire_on_floor(0, x=t17.interior_grid.grid_nodes[0][0], y=t17.interior_grid.grid_nodes[0][1])

            evacuation_time = MAX_TICKS
            people_in_building = []
            alarm_handled = False
            for tick in range(MAX_TICKS):
                model.step()
                if getattr(t17.interior_grid, 'fire_alarm_active', False) and not alarm_handled:
                    alarm_handled = True
                    for floor_agents in t17.interior_grid.floors.values():
                        for agent in floor_agents:
                            if mode == 'ideal':
                                agent.alarm_response_timer = 0
                            else:
                                agent.alarm_response_timer = random.randint(100, 400)

                people_inside = len(t17.inventory)
                people_in_building.append({
                    'run_id': i,
                    'mode': mode,
                    'tick': tick,
                    'people_inside': people_inside
                })

                if people_inside == 0:
                    for extra_tick in range(tick + 1, tick + 2000):
                        people_in_building.append({
                            'run_id': i,
                            'mode': mode,
                            'tick': extra_tick,
                            'people_inside': 0
                        })
                    break

            all_data.extend(people_in_building)

    df = pd.DataFrame(all_data)
    df.to_csv(os.path.join(OUTPUT_DIRECTORY, "results_scenario_3_T17_ideal_vs_realistic.csv"), index=False)
    print(f"Scenario {SCENARIO_TO_RUN} finished")


elif SCENARIO_TO_RUN == 3:
    print(f"Scenario {SCENARIO_TO_RUN} started: {N_RUNS_SMALL} iterations on staircaise bottleneck evolution in T17")
    all_data = []
    t17_index = 16

    for i in range (N_RUNS_SMALL):
        print(f"Run no. {i+1} out of {N_RUNS_SMALL}")
        model = CampusModel(G_all, G_drive, nodes, buildings, doors,
                            n_students=600,
                            data_collection_mode='stair_bottleneck',
                            interior_fire_death_ticks_override=120)
        model.safe_nodes = safe
        for j, (idx, row) in enumerate(buildings.iterrows()):
            if j < 22:
                model.buildings[j].polygon_coords = list(row.geometry.exterior.coords)

        t17 = model.buildings[t17_index]

        t17.inventory.clear()

        for a in model.schedule.agents:
            if type(a).__name__ == 'Student':
                a.home_dorm = t17.name
                a.current_building = t17
                a.indoors = True
                t17.inventory.append(a)

        t17.init_interior()

        xs = [n[0] for n in t17.interior_grid.grid_nodes]
        ys = [n[1] for n in t17.interior_grid.grid_nodes]
        center_x = sum(xs) / len(xs)
        center_y = sum(ys) / len(ys)
        center_node = min(t17.interior_grid.grid_nodes, key=lambda n: (n[0] - center_x) ** 2 + (n[1] - center_y) ** 2)

        t17.interior_grid.set_fire_on_floor(2, x=center_node[0], y=center_node[1])

        for tick in range(MAX_TICKS):
            model.step()
            if hasattr(t17.interior_grid, 'stair_queues'):
                for stair in t17.interior_grid.stair_queues:
                    queue = t17.interior_grid.stair_queues[stair]
                    t17.interior_grid.stair_queues[stair] = [
                        a for a in queue
                        if not getattr(a.map_student, 'is_dead', False)
                           and not a.path_failed
                    ]
            if hasattr(t17.interior_grid, 'stair_queues'):
                bottleneck = sum(len(queue) for queue in t17.interior_grid.stair_queues.values())
            else:
                bottleneck = 0

            all_data.append({
                'run_id': i,
                'tick': tick,
                'people_in_queue': bottleneck,
                'fire_radius': model.current_fire_radius if model.fire_started else 0.0
            })

            alive_inside = sum(1 for a in t17.inventory if not getattr(a, 'is_dead', False))
            dead_inside = sum(
                1 for a in t17.inventory
                if getattr(a, 'is_dead', False)
            )
            # Curăță morții din inventory
            t17.inventory = [
                a for a in t17.inventory
                if not getattr(a, 'is_dead', False)
            ]

            if alive_inside == 0:
                break

            if len(t17.inventory) == 0:
                break

    df = pd.DataFrame(all_data)
    df.to_csv(os.path.join(OUTPUT_DIRECTORY, "results_scenario_4_bottleneck.csv"), index=False)
    print(f"Scenario {SCENARIO_TO_RUN} finished")

elif SCENARIO_TO_RUN == 4:
    print(f"Scenario {SCENARIO_TO_RUN} started: {N_RUNS_SMALL} iterations on Firefighters Delay Impact")
    all_data = []
    delay_modes = {
        'Fast': (20, 40),
        'Normal': (80, 150),
        'Slow': (300, 500)
    }

    fire_target_x = buildings.iloc[0].geometry.centroid.x
    fire_target_y = buildings.iloc[0].geometry.centroid.y

    for mode_name, (d_min, d_max) in delay_modes.items():
        for i in range(N_RUNS_SMALL):
            print(f"Run no. {i + 1} out of {N_RUNS_SMALL} - Mode: {mode_name}")
            model = CampusModel(G_all, G_drive, nodes, buildings, doors, n_students=200, truck_delay_min_override=d_min,
                                truck_delay_max_override=d_max)
            model.safe_nodes = safe
            for j, (idx, row) in enumerate(buildings.iterrows()):
                if j < 22:
                    model.buildings[j].polygon_coords = list(row.geometry.exterior.coords)

            model.ignite_fire(fire_target_x, fire_target_y)

            max_steps = 0
            while model.fire_ever_started and model.fire_started:
                model.step()
                max_steps += 1
                all_data.append({
                    'run_id': i,
                    'delay_mode': mode_name,
                    'tick': model.schedule.steps,
                    'fire_radius': model.current_fire_radius if model.fire_started else 0.0
                })
                if max_steps > 8000:
                    break

    df = pd.DataFrame(all_data)
    df.to_csv(os.path.join(OUTPUT_DIRECTORY, "results_scenario_5_firefighter_delay.csv"), index=False)
    print(f"Scenario {SCENARIO_TO_RUN} finished")


elif SCENARIO_TO_RUN == 5:
    print(f"Starting Scenario {SCENARIO_TO_RUN}: Campus Congestion Heatmap")
    all_data = []

    for i in range(N_RUNS_SMALL):
        print(f"Run no. {i + 1} out of {N_RUNS_SMALL}")
        model = CampusModel(G_all, G_drive, nodes, buildings, doors, n_students=2500, data_collection_mode='campus_heatmap')
        model.safe_nodes = safe
        for j, (idx, row) in enumerate(buildings.iterrows()):
            if j < 22:
                model.buildings[j].polygon_coords = list(row.geometry.exterior.coords)

        random_node_id = random.choice(list(nodes.index))
        fire_x = nodes.loc[random_node_id]['x']
        fire_y = nodes.loc[random_node_id]['y']

        model.ignite_fire(fire_x, fire_y)
        for tick in range(MAX_TICKS):
            model.step()

            if getattr(model, 'fire_ever_started', False) and not model.fire_started:
                break

        for row in model.exterior_positions_log:
            row['run_id'] = i
            row['fire_start_x'] = fire_x
            row['fire_start_y'] = fire_y
            all_data.append(row)

    df = pd.DataFrame(all_data)
    df.to_csv(os.path.join(OUTPUT_DIRECTORY, "results_scenario_6_campus_congestion.csv"), index=False)
    print(f"Scenario {SCENARIO_TO_RUN} finished")
