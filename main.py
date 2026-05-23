import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import random
import numpy as np
import contextily as ctx
from matplotlib.lines import Line2D
from matplotlib.widgets import Button
from matplotlib.patches import Polygon
from map_loader import *
from shapely.geometry import Point, Polygon as ShapelyPolygon
from simulation_model import CampusModel
from config import NUM_DORMS, TARGET_POPULATION_MIN, TARGET_POPULATION_MAX, SMOKE_LIFESPAN

if __name__ == '__main__':

    G_all, G_drive, nodes, edges, buildings, doors, safe = load_campus_map()

    n_stud = random.randint(TARGET_POPULATION_MIN, TARGET_POPULATION_MAX)
    model = CampusModel(G_all, G_drive, nodes, buildings, doors, n_students=n_stud)
    model.safe_nodes = safe
    is_paused = False
    is_fire_mode = False

    plt.rcParams['keymap.fullscreen'].remove('f')

    fig = plt.figure(figsize=(15,9))
    gs = gridspec.GridSpec(3, 2, width_ratios=[4, 1.2], height_ratios=[1.2, 1.4, 0.10], hspace=0.05, wspace=0.05)
    ax = fig.add_subplot(gs[:, 0])
    ax_info = fig.add_subplot(gs[0, 1])
    ax_inter = fig.add_subplot(gs[1, 1])
    ax_btns = fig.add_subplot(gs[2, 1])
    ax_info.set_axis_off()
    ax_inter.set_axis_off()
    ax_inter.set_facecolor('#f0f0f0')
    ax_btns.set_axis_off()

    pos = ax_btns.get_position()
    btn_h = pos.height*0.85
    btn_y = pos.y0 + pos.height * 0.075
    btn_w = pos.width * 0.44

    ax_alerts = fig.add_axes([0.02, 0.75, 0.2, 0.22])
    ax_alerts.set_axis_off()
    alert_panel = ax_alerts.text(0.0, 1.0, "", transform=ax_alerts.transAxes, va='top', fontsize=9, fontweight='bold', bbox=dict(boxstyle='round, pad=0.3', fc='#ff4d4d', ec='black', alpha=0.9), zorder=998)
    alert_panel.set_visible(False)
    fire_panel = ax_alerts.text(0.0, 1.0, "", transform=ax_alerts.transAxes, va='top', fontsize=8, fontweight='bold', bbox=dict(boxstyle='round, pad=0.3', fc='#ffcccc', ec='black', alpha=0.9), zorder=998)

    bottleneck_panel = ax_alerts.text(0.0, -0.9, "", transform=ax_alerts.transAxes, va='top', fontsize=8, bbox=dict(boxstyle='round, pad=0.3', fc='#f0f0f0', ec='gray', alpha=0.9), zorder=998)
    bottleneck_panel.set_visible(False)

    menu_text = ax_info.text(0.05, 0.98, "", transform=ax_info.transAxes, va='top', fontsize=8, bbox=dict(boxstyle='round, pad=0.3', fc='#e6f2ff', ec='gray', alpha=0.9))

    selected_building = None
    current_floor = 0
    ax_btn_down = fig.add_axes([pos.x0, btn_y, btn_w, btn_h])
    ax_btn_up = fig.add_axes([pos.x0 + pos.width * 0.56, btn_y, btn_w, btn_h])
    btn_down = Button(ax_btn_down, 'Down', color='#dde', hovercolor='#aac')
    btn_up = Button(ax_btn_up, 'Up', color='#dde', hovercolor='#aac')
    ax_btn_down.set_visible(False)
    ax_btn_up.set_visible(False)

    def floor_up(event):
        global current_floor
        if selected_building and current_floor < 4:
            current_floor += 1
            render_interior(selected_building, current_floor)

    def floor_down(event):
        global current_floor
        if selected_building and current_floor > 0:
            current_floor -= 1
            render_interior(selected_building, current_floor)

    btn_up.on_clicked(floor_up)
    btn_down.on_clicked(floor_down)

    if not buildings.empty:
        dorms = buildings[buildings['is_dorm'] == True]
        dorms.plot(ax=ax, color='#a67c52', edgecolor='black', alpha=0.7, label='Dorms')

    dorms_gdf_3857 = gpd.read_file("camine_tuiasi.geojson").to_crs("EPSG:3857")
    for i, (idx, row) in enumerate(dorms_gdf_3857.iterrows()):
        if i<NUM_DORMS:
            model.buildings[i].polygon_coords = list(row.geometry.exterior.coords)

    def find_building(x, y):
        pt = Point(x, y)
        for i, (idx, row) in enumerate(dorms_gdf_3857.iterrows()):
            if i>= NUM_DORMS:
                break
            if row.geometry.contains(pt) or row.geometry.distance(pt) < 15:
                return model.buildings[i]
        return None

    edges.plot(ax=ax, color='#bdc3c7', linewidth=0.5, alpha=0.5, zorder=1)

    try:
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zorder=0)
    except Exception as e:
        print("No internet connection for downloading the map")
        ax.set_facecolor('#e8e5e1')

    scat = ax.scatter([], [], c='blue', s=20, zorder=10, label='Students', picker=5)

    fire_glow = ax.scatter([], [], c='red', s=40, edgecolors='none', alpha=0.2, zorder=5)
    fire_core = ax.scatter([], [], c='orange', s=40, edgecolors='none', alpha=0.8, zorder=6)
    smoke_scatter = ax.scatter([], [], c='gray', alpha=0.4, marker='o', edgecolors='none', zorder=8)

    selected_agent_id = None
    high_scat = ax.scatter([],[],c='lime', s=80, edgecolors='white', linewidth=2, zorder=11)
    truck_scat = ax.scatter([], [], c='yellow', s=80, marker='s', edgecolors='black', linewidth=2, zorder=12, label='Firetruck')
    ff_scat = ax.scatter([], [], c='brown', s=40, marker='o', edgecolors='black', linewidth=1.5, zorder=13, label='Firefighter', picker=5)
    water_scat = ax.scatter([], [], c='cyan', s=8, alpha=0.7, edgecolors='black', zorder=14, label='Water')

    info_panel = ax.text(0.80, 0.95, "", transform=ax.transAxes,
                         verticalalignment='top', horizontalalignment='center',
                         fontsize=10, fontweight='bold', multialignment='center',
                         bbox=dict(boxstyle='round',pad=0.5, facecolor='white', edgecolor='black', alpha=0.9), zorder=999)

    def update_info_display(agent):
        if agent is None:
            info_panel.set_text("")
            fig.canvas.draw_idle()
            return

        if type(agent).__name__ == 'Firefighter':
            if not agent.model.fire_started or getattr(agent, 'is_returning', False):
                mission = "Return to truck"
            else:
                mission = "Extinguish fire"

            truck_id = agent.truck.unique_id if getattr(agent, 'truck', None) else "N/A"
            name = getattr(agent, 'full_name', f"Firefighter #{agent.unique_id}")

            text = (f"{name}\n"
                    f"Mission: {mission}\n"
                    f"Truck ID: {truck_id}\n")

            info_panel.set_text(text)
            fig.canvas.draw_idle()
            return


        status = "Panicked" if agent.is_panicked else "Calm"
        if agent.is_dead:
            status = "Dead"

        dest = getattr(agent, 'target_name', "None")

        if getattr(agent, 'is_hidden', False):
            if agent.current_building and getattr(agent.current_building, 'interior_grid', None):
                interior_agent = None
                for floor_idx, f_agents in agent.current_building.interior_grid.floors.items():
                    for a in f_agents:
                        if a.map_student == agent:
                            interior_agent = a
                            break
                    if interior_agent: break
                if interior_agent:
                    if interior_agent.is_exiting:
                        pass
                    else:
                        dest = f"{agent.current_building.name} (floor {interior_agent.target_floor})"
            else:
                dest += " (inside)"
        else:
            if "Cămin" in dest and getattr(agent, 'target_floor', None) is not None:
                dest += f" (floor {agent.target_floor})"

        text = (f"Name: {agent.full_name}\n"
                f"Home Dormitory: {agent.home_dorm}\n"
                f"Destination: {dest}\n"
                f"Status: {status}\n")
        if getattr(agent, 'is_aware', False):
            if getattr(agent, 'informed_by', None) == "Fire Alarm":
                text += "Heard Fire Alarm\n"
            elif getattr(agent, 'informed_by', None):
                text += f"Alerted by: {agent.informed_by}\n"
            else:
                text += f"Saw fire directly\n"
        text += f"ID: {agent.unique_id}\n"
        info_panel.set_text(text)
        fig.canvas.draw_idle()

    def update_status_title():
        p_txt = "Paused" if is_paused else "Running"
        f_txt = "Fire Mode On" if is_fire_mode else "Fire Mode Off"
        ax.set_title(f"Simulation - {p_txt} - {f_txt}", fontsize=13, fontweight='bold', pad=10)
        fig.canvas.draw_idle()

    last_drawn_building = None
    last_drawn_floor = -1
    inter_norm_scat = None
    inter_trans_scat = None
    inter_high_scat = None
    inter_title_text = None
    fire_glow_inter = None
    fire_core_inter = None
    inter_ff_scat = None
    inter_water_scat = None

    def render_interior(building, floor):
        global inter_water_scat, inter_ff_scat, interior_alarm_panel, fire_core_inter, fire_glow_inter, selected_agent_id, last_drawn_building, last_drawn_floor, inter_title_text, inter_high_scat, inter_trans_scat, inter_norm_scat
        if building is None or getattr(building, 'interior_grid', None) is None:
            ax_inter.cla()
            ax_btn_down.set_visible(False)
            ax_btn_up.set_visible(False)
            ax_inter.text(0.5, 0.5, 'Click on a dormitory', ha='center', va='center', transform=ax_inter.transAxes, fontsize=9, color='gray')
            fig.canvas.draw_idle()
            last_drawn_building = None
            return

        ig = building.interior_grid
        xs = [c[0] for c in building.local_coords]
        xy = [c[1] for c in building.local_coords]
        margin = 5

        if building != last_drawn_building:
            ax_inter.cla()
            ax_inter.set_axis_off()
            ax_inter.set_facecolor('#e8e0d0')

            poly_patch = Polygon(building.local_coords, closed=True, edgecolor='#333333', facecolor='#e8dcc8', linewidth=2, zorder=2)
            ax_inter.add_patch(poly_patch)

            stairs = ig.get_stair_nodes()
            if stairs:
                ax_inter.scatter([s[0] for s in stairs], [s[1] for s in stairs], s=120, c='#e67e22', marker='s', edgecolors='black', linewidth=1, zorder=5)

            fire_glow_inter = ax_inter.scatter([], [], c='red', alpha=0.25, edgecolors='none', zorder=4)
            fire_core_inter = ax_inter.scatter([], [], c='orange', alpha=0.9, edgecolors='none', zorder=5)
            inter_norm_scat = ax_inter.scatter([], [], s=25, c='#2980b9', edgecolors='white', linewidths=0.5, zorder=6)
            inter_trans_scat = ax_inter.scatter([], [], s=30, c='#95a5a6', edgecolors='black', linewidths=1, zorder=7, alpha=0.8)
            inter_high_scat = ax_inter.scatter([], [], c='lime', s=80, edgecolors='white', linewidth=2, zorder=10)
            inter_title_text = ax_inter.text(0.5, 1.02, "", transform=ax_inter.transAxes, ha='center', va='bottom', fontsize=9, fontweight='bold', bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.8))

            interior_alarm_panel = ax_inter.text(0.5, 0.9, "", transform=ax_inter.transAxes, ha='center', va='top', fontsize=11, fontweight='bold', color='white', bbox=dict(boxstyle='round', fc='red', ec='black', alpha=0.9), zorder=999)
            interior_alarm_panel.set_visible(False)

            inter_ff_scat = ax_inter.scatter([], [], s=40, c='brown', edgecolors='black', linewidths=1.5, marker='o', zorder=12)
            inter_water_scat = ax_inter.scatter([], [], s=8, c='#3498db', edgecolors='none', alpha=0.7, zorder=11)
            ax_inter.set_xlim(min(xs) - margin, max(xs) + margin)
            ax_inter.set_ylim(min(xy) - margin, max(xy) + margin)
            ax_inter.set_aspect('equal', adjustable='box')
            last_drawn_building = building

        last_drawn_floor = floor

        agents = [a for a in ig.get_agents_on_floor(floor) if a.map_student is not None]

        floor_blobs = [b for b in getattr(ig, 'fire_blobs', []) if b['floor'] == floor]
        if floor_blobs:
            coords = np.c_[[b['x'] for b in floor_blobs], [b['y'] for b in floor_blobs]]
            sizes_glow = np.random.uniform(25, 55, len(floor_blobs))
            sizes_core = np.full(len(floor_blobs), 15)
            fire_glow_inter.set_offsets(coords)
            fire_glow_inter.set_sizes(sizes_glow)
            fire_core_inter.set_offsets(coords)
            fire_core_inter.set_sizes(sizes_core)
        else:
            fire_glow_inter.set_offsets(np.empty((0, 2)))
            fire_core_inter.set_offsets(np.empty((0, 2)))

        if agents:
            norm_ag = [a for a in agents if not a.in_transit]
            trans_ag = [a for a in agents if a.in_transit]
            if norm_ag:
                inter_norm_scat.set_offsets(np.c_[[a.x for a in norm_ag], [a.y for a in norm_ag]])
                colors = []
                for a in norm_ag:
                    if getattr(a.map_student, 'is_dead', False):
                        colors.append('black')
                    elif getattr(a, 'is_aware_of_fire', False):
                        colors.append('red')
                    else:
                        colors.append('#2980b9')
                inter_norm_scat.set_color(colors)
            else:
                inter_norm_scat.set_offsets(np.empty((0, 2)))

            if trans_ag:
                inter_trans_scat.set_offsets(np.c_[[a.x for a in trans_ag], [a.y for a in trans_ag]])
                inter_trans_scat.set_color(['#bdc3c7' for _ in trans_ag])
                inter_trans_scat.set_alpha(0.6)
            else:
                inter_trans_scat.set_offsets(np.empty((0, 2)))

            high_coords = []
            if selected_agent_id is not None:
                for a in agents:
                    if a.map_student and a.map_student.unique_id == selected_agent_id:
                        high_coords = [[a.x, a.y]]
                        break

            if not high_coords and selected_agent_id is not None:
                for a in ig.get_agents_on_floor(floor):
                    if getattr(a, 'is_firefighter', False):
                        ff_ref = getattr(a, 'firefighter_ref', None)
                        if ff_ref and getattr(ff_ref, 'unique_id', None) == selected_agent_id:
                            high_coords = [[a.x, a.y]]
                            break
                    elif a.map_student and a.map_student.unique_id == selected_agent_id:
                        high_coords = [[a.x, a.y]]
                        break

            if high_coords:
                inter_high_scat.set_offsets(high_coords)
            else:
                inter_high_scat.set_offsets(np.empty((0,2)))

            inter_title_text.set_text(f"{building.name} - {'Ground Floor' if floor==0 else f'Floor no {floor}'} ({len(agents)} people)")

        else:
            inter_norm_scat.set_offsets(np.empty((0,2)))
            inter_trans_scat.set_offsets(np.empty((0,2)))
            inter_high_scat.set_offsets(np.empty((0,2)))
            inter_title_text.set_text(f"{building.name} - {'Ground Floor' if floor==0 else f'Floor no {floor}'} (0 people)")

        if inter_ff_scat is not None:
            ff_agents = [a for a in ig.get_agents_on_floor(floor) if getattr(a, 'is_firefighter', False)]
            if ff_agents:
                inter_ff_scat.set_offsets(np.c_[[a.x for a in ff_agents], [a.y for a in ff_agents]])
            else:
                inter_ff_scat.set_offsets(np.empty((0, 2)))

        if inter_water_scat is not None:
            ig.interior_water_particles = [p for p in getattr(ig, 'interior_water_particles', []) if p['life'] > 0]
            floor_water = [p for p in ig.interior_water_particles if p['floor'] == floor]
            if floor_water:
                inter_water_scat.set_offsets(np.c_[[p['x'] for p in floor_water], [p['y'] for p in floor_water]])
            else:
                inter_water_scat.set_offsets(np.empty((0, 2)))

        if getattr(ig, 'fire_alarm_active', False):
            interior_alarm_panel.set_text(f"Fire Alarm in {building.name}")
            interior_alarm_panel.set_visible(True)
        else:
            if 'interior_alarm_panel' in globals():
                interior_alarm_panel.set_visible(False)

        ax_btn_down.set_visible(True)
        ax_btn_up.set_visible(True)
        fig.canvas.draw_idle()

    tracked_stair = None
    def on_click(event):
        global tracked_stair, selected_building, current_floor, selected_agent_id, last_drawn_floor

        if event.inaxes == ax:
            cx, cy = event.xdata, event.ydata
            click_pt = Point(cx, cy)

            if is_fire_mode:
                if getattr(model, 'fire_ever_started', False):
                    return
                for b in model.buildings:
                    if getattr(b, 'polygon_coords', None) and ShapelyPolygon(b.polygon_coords).contains(click_pt):
                        return
                x, y = model.snap_to_nearest_hotspot(event.xdata, event.ydata)
                model.ignite_fire(x, y)
                model.fire_ever_started = True
            else:
                b = find_building(event.xdata, event.ydata)
                if b is not None:
                    building_on_fire = None
                    for build in model.buildings:
                        if build.interior_grid and hasattr(build.interior_grid, 'fire_floors') and build.interior_grid.fire_floors:
                            building_on_fire = build
                            break
                    if building_on_fire is not None and b!=building_on_fire:
                        return

                    selected_building = b
                    current_floor = 0
                    b.init_interior()
                    render_interior(b, current_floor)

        elif event.inaxes == ax_inter and selected_building:
            ig = selected_building.interior_grid
            if not ig:
                return
            cx, cy = event.xdata, event.ydata
            if cx is None:
                return

            for stair in ig.get_stair_nodes():
                if (cx - stair[0])**2 + (cy - stair[1])**2 < 12.0:
                    tracked_stair = stair
                    return

            if is_fire_mode:
                if not getattr(model, 'fire_ever_started', False):
                    ig.set_fire_on_floor(current_floor, cx, cy)
                    model.interior_fire_only = True
                    model.fire_started = True
                    model.fire_ever_started = True
                    model.fire_center_x = selected_building.door_coords[0]
                    model.fire_center_y = selected_building.door_coords[1]
                    last_drawn_floor = -1
                    render_interior(selected_building, current_floor)
                    return

            agents = ig.get_agents_on_floor(current_floor)
            best, best_d = None, float('inf')
            for agent in agents:
                d = (agent.x-cx)**2 + (agent.y-cy)**2
                if d < best_d:
                    best_d = d
                    best = agent
            if best and best.map_student and best_d < 100:
                selected_agent_id = best.map_student.unique_id
                update_info_display(best.map_student)
                render_interior(selected_building, current_floor)
            else:
                ff_best, ff_best_d = None, float('inf')
                for agent in agents:
                    if not getattr(agent, 'is_firefighter', False):
                        continue
                    d = (agent.x - cx)**2 + (agent.y - cy)**2
                    if d < ff_best_d:
                        ff_best_d = d
                        ff_best = agent
                if ff_best and ff_best_d < 100:
                    ff_ref = getattr(ff_best, 'firefighter_ref', None)
                    if ff_ref:
                        selected_agent_id = ff_ref.unique_id
                        update_info_display(ff_ref)
                        render_interior(selected_building, current_floor)

    def on_key(event):
        global is_paused, is_fire_mode
        if event.key == ' ':
            is_paused = not is_paused
            if is_paused:
                ani.event_source.stop()
            else:
                ani.event_source.start()

        if event.key in ['f', 'F']:
            is_fire_mode = not is_fire_mode

        update_status_title()

    def on_pick(event):
        global selected_agent_id
        index = event.ind[0]

        if event.artist == scat:
            all_agents = [a for a in model.schedule.agents if type(a).__name__ == 'Student' and getattr(a, 'is_active', False) and not getattr(a, 'is_hidden', False)]
            if index<len(all_agents):
                agent = all_agents[index]
                selected_agent_id = agent.unique_id
                update_info_display(agent)
                high_scat.set_offsets([(agent.x, agent.y)])
                fig.canvas.draw_idle()

        elif event.artist == ff_scat:
            all_firefighters = [a for a in model.schedule.agents if type(a).__name__ == 'Firefighter' and not getattr(a, 'is_hidden', False)]
            if index < len(all_firefighters):
                agent = all_firefighters[index]
                selected_agent_id = agent.unique_id
                update_info_display(agent)
                high_scat.set_offsets([(agent.x, agent.y)])
                fig.canvas.draw_idle()

    def update(frame):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        model.step()
        if selected_building and selected_building.interior_grid:
            render_interior(selected_building, current_floor)

        agents, trucks, firefighters = [], [], []
        for a in model.schedule.agents:
            t = type(a).__name__
            if t == 'Student' and getattr(a, 'is_active', False) and not getattr(a, 'is_hidden', False):
                agents.append(a)
            elif t == 'Firetruck':
                trucks.append(a)
            elif t == 'Firefighter' and not getattr(a, 'is_hidden', False):
                firefighters.append(a)
        if agents:
            offsets = [(a.x, a.y) for a in agents]
            colors = [a.color for a in agents]
            scat.set_offsets(offsets)
            scat.set_array(None)
            scat.set_color(colors)

        if model.fire_started and model.fire_blobs:
            coords = np.array([[b['x'], b['y']] for b in model.fire_blobs])
            fire_core.set_offsets(coords)
            fire_glow.set_offsets(coords)
            fire_glow.set_sizes([model.current_fire_radius * 10] * len(coords))
        else:
            fire_core.set_offsets(np.empty((0, 2)))
            fire_glow.set_offsets(np.empty((0, 2)))

        if model.smoke_blobs:
            sx = [b['x'] for b in model.smoke_blobs]
            sy = [b['y'] for b in model.smoke_blobs]
            sizes = [b['size'] for b in model.smoke_blobs]
            alphas = [0.4 * (1.0 - (b['age']/SMOKE_LIFESPAN)) for b in model.smoke_blobs]
            smoke_colors = np.zeros((len(model.smoke_blobs), 4))
            smoke_colors[:, 0:3] = 0.5
            smoke_colors[:, 3] = alphas
            smoke_scatter.set_offsets(np.c_[sx, sy])
            smoke_scatter.set_sizes(sizes)
            smoke_scatter.set_color(smoke_colors)
        else:
            smoke_scatter.set_offsets(np.empty((0, 2)))

        if selected_agent_id is not None:
            selected_ag = None
            for a in model.schedule.agents:
                if getattr(a, 'unique_id', None) == selected_agent_id:
                    selected_ag = a
                    break

            if selected_ag:
                update_info_display(selected_ag)

                if not getattr(selected_ag, 'is_hidden', False):
                    high_scat.set_offsets(np.c_[[selected_ag.x], [selected_ag.y]])
                else:
                    high_scat.set_offsets(np.empty((0, 2)))
            else:
                high_scat.set_offsets(np.empty((0, 2)))
                info_panel.set_text("")
        else:
            high_scat.set_offsets(np.empty((0, 2)))

        sorted_buildings = sorted(model.buildings, key=lambda x:len(x.inventory), reverse=True)
        status_txt = "Campus Status:\n"
        for b in sorted_buildings[:12]:
            if len(b.inventory) > 0:
                status_txt += f"- {b.name}: {len(b.inventory)} people\n"
        menu_text.set_text(status_txt)
        menu_text.set_bbox(dict(boxstyle='round', facecolor='#e6f2ff', edgecolor='gray', alpha=0.9))

        burning_buildings = [b for b in model.buildings if b.is_on_fire or (b.interior_grid and b.interior_grid.fire_floors)]
        has_fire = len(burning_buildings) > 0
        if has_fire:
            fire_txt = f"FIRE DETECTED! Radius: {model.current_fire_radius:.1f}m\n\n"
            for b in burning_buildings[:3]:
                fire_txt += f"Fire at {b.name}\n ({len(b.inventory)} people inside)\n"
            if len(burning_buildings) > 3:
                fire_txt += f"...and other {len(burning_buildings) - 3} burning buildings\n"

            b_main = burning_buildings[0]
            if len(b_main.inventory) > 0:
                fire_txt += f"\nPeople stuck inside {b_main.name}:\n"
                names = [a.full_name for a in b_main.inventory[:6]]
                fire_txt += "\n".join(names) + "\n"
                if len(b_main.inventory) > 6:
                    fire_txt += f"...and other {len(b_main.inventory) - 6}\n"

            fire_panel.set_text(fire_txt)
            fire_panel.set_visible(True)
        else:
            fire_panel.set_visible(False)

        if tracked_stair and selected_building:
            ig = selected_building.interior_grid
            queue = ig.stair_queues.get(tracked_stair, [])

            on_stairs = []
            for floor_idx in ig.floors:
                for a in ig.floors[floor_idx]:
                    if not a.in_transit:
                        continue
                    if getattr(a, 'is_firefighter', False):
                        on_stairs.append(a)
                    elif abs(a.x - tracked_stair[0]) < 3 and abs(a.y - tracked_stair[1]) < 3:
                        on_stairs.append(a)

            info_txt = "Stair Bottleneck: \n"
            info_txt += f"People on stairs: {len(on_stairs)}\n"


            for a in on_stairs:
                if getattr(a, 'is_firefighter', False):
                    ff_ref = getattr(a, 'firefighter_ref', None)
                    name = getattr(ff_ref, 'full_name', 'Firefighter') if ff_ref else 'Firefighter'
                    start_f = getattr(a, 'firefighter_climb_start_floor', a.floor)
                    end_f = getattr(a, 'firefighter_climb_target_floor', a.target_floor)
                    total_time = getattr(a, 'stair_timer_total', 35.0)
                    progress = 1.0 - (a.stair_timer / total_time) if total_time > 0 else 1.0
                else:
                    name = a.map_student.full_name if getattr(a, 'map_student', None) else "Student"
                    start_f = a.floor
                    end_f = a.target_floor
                    total = getattr(a, 'stair_timer_total', 35.0)
                    progress = 1.0 - (a.stair_timer / total) if total > 0 else 1.0

                curr_f = start_f + (end_f - start_f) * progress
                info_txt += f"- {name} (floor {int(round(curr_f))})\n"


            info_txt += f"\nWaiting: {len(queue)}"
            for a in queue[:5]:
                name = a.map_student.full_name if a.map_student else "Firefighter"
                info_txt += f"- {name}\n"
            if len(queue) > 5:
                info_txt += f"...and {len(queue) - 5} more\n"

            bottleneck_panel.set_text(info_txt)
            bottleneck_panel.set_visible(True)

        has_alert = getattr(model, 'alarm_triggered', False) and not getattr(model, 'truck_dispatched', False)

        if trucks:
            tx = [t.x for t in trucks]
            ty = [t.y for t in trucks]
            truck_scat.set_offsets(np.c_[tx, ty])
        else:
            truck_scat.set_offsets(np.empty((0, 2)))

        cy = 1.0
        if has_alert:
            alert_panel.set_text(
                f"112 Emergency!\nReported by: {model.hero_name}\nISU arriving in {model.truck_timer} frames\n")
            alert_panel.set_position((0.0, cy))
            alert_panel.set_visible(True)
            cy -= 0.35
        else:
            alert_panel.set_visible(False)

        if has_fire:
            fire_panel.set_position((0.0, cy))
            fire_panel.set_visible(True)
        else:
            fire_panel.set_visible(False)

        menu_text.set_position((0.05, 0.98))

        if firefighters:
            fx = [f.x for f in firefighters]
            fy = [f.y for f in firefighters]
            ff_scat.set_offsets(np.c_[fx, fy])
        else:
            ff_scat.set_offsets(np.empty((0, 2)))

        if hasattr(model, 'water_particles') and model.water_particles:
            wx = [p['x'] for p in model.water_particles]
            wy = [p['y'] for p in model.water_particles]
            water_scat.set_offsets(np.c_[wx, wy])
        else:
            water_scat.set_offsets(np.empty((0, 2)))

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return scat, fire_glow, fire_core, smoke_scatter, high_scat, alert_panel, fire_panel, menu_text, truck_scat, ff_scat, water_scat

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('pick_event', on_pick)


    ax.set_axis_off()
    title_text = ax.set_title("Tudor Vladimirescu - Simulation\nRunning - Fire Mode Off", fontsize=13, fontweight='bold', pad=10)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=7, label='Citizen (unaware)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=7, label='Citizen (aware/panicked)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=7, label='Citizen (dead)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='yellow', markersize=8, label='Firetruck', markeredgecolor='black', markeredgewidth=1.0),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='brown', markersize=7, label='Firefighter', markeredgecolor='black', markeredgewidth=0.8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='Fire', markeredgecolor='none'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=6, label='Water', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Smoke', markeredgecolor='none', alpha=0.5),
    ]

    legend = ax.legend(
        handles=legend_elements,
        loc='lower left',
        fontsize=7.5,
        framealpha=0.7,
        facecolor='white',
        edgecolor='gray',
        borderpad=0.7,
        labelspacing=0.4,
        handletextpad=0.5,
        title='Legend',
        title_fontsize=8,
    )
    legend.set_zorder(1000)


    ani = animation.FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)

    manager = plt.get_current_fig_manager()
    try:
        manager.window.wm_geometry("+160+20")
    except Exception as e:
        pass


    plt.show()