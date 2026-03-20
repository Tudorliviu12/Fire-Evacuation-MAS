import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import numpy as np
import contextily as ctx
from jupyter_core.paths import is_hidden

from map_loader import *
from simulation_model import CampusModel
from config import TARGET_POPULATION_MIN, TARGET_POPULATION_MAX, SMOKE_LIFESPAN

if __name__ == '__main__':

    G_all, G_drive, nodes, edges, buildings, doors, safe = load_campus_map()

    n_stud = random.randint(TARGET_POPULATION_MIN, TARGET_POPULATION_MAX)
    model = CampusModel(G_all, G_drive, nodes, doors, n_students=n_stud)
    model.safe_nodes = safe
    is_paused = False
    is_fire_mode = False

    plt.rcParams['keymap.fullscreen'].remove('f')

    fig, ax = plt.subplots(figsize=(10,9))

    if not buildings.empty:
        buildings.plot(ax=ax, color='#a67c52', edgecolor='black', alpha=0.7, label='Dorms')

    edges.plot(ax=ax, color='#bdc3c7', linewidth=0.5, alpha=0.5, zorder=1)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zorder=0)

    scat = ax.scatter([], [], c='blue', s=20, zorder=10, label='Students', picker=5)

    fire_glow = ax.scatter([], [], c='red', s=40, edgecolors='none', alpha=0.2, zorder=5)
    fire_core = ax.scatter([], [], c='orange', s=40, edgecolors='none', alpha=0.8, zorder=6)
    smoke_scatter = ax.scatter([], [], c='gray', alpha=0.4, marker='o', edgecolors='none', zorder=8)

    selected_agent_id = None
    high_scat = ax.scatter([],[],c='lime', s=80, edgecolors='white', linewidth=2, zorder=11)
    info_panel = ax.text(0.80, 0.95, "", transform=ax.transAxes,
                         verticalalignment='top', horizontalalignment='center',
                         fontsize=10, fontweight='bold', multialignment='center',
                         bbox=dict(boxstyle='round',pad=0.5, facecolor='white', edgecolor='black', alpha=0.9))

    def update_info_display(agent):
        if agent:
            status = "Panicked" if agent.is_panicked else "Calm"
            if agent.is_dead:
                status = "Dead"

            destinatie = getattr(agent, 'target_name', "None")

            if getattr(agent, 'is_hidden', False):
                destinatie += " - inside"

            text = (f"Name: {agent.full_name}\n"
                    f"Home Dormitory: {agent.home_dorm}\n"
                    f"Destination: {destinatie}\n"
                    f"Status: {status}\n"
                    f"ID: {agent.unique_id}\n")
            info_panel.set_text(text)
            fig.canvas.draw_idle()

    def update_status_title():
        p_txt = "Paused" if is_paused else "Running"
        f_txt = "Fire Mode On" if is_fire_mode else "Fire Mode Off"
        title_text.set_text(f"Simulation - {p_txt} - {f_txt}")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes == ax:
            if is_fire_mode:
                model.ignite_fire(event.xdata, event.ydata)

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
        all_agents = [a for a in model.schedule.agents if a.is_active]

        if index<len(all_agents):
            agent = all_agents[index]
            selected_agent_id = agent.unique_id
            update_info_display(agent)
            high_scat.set_offsets([(agent.x, agent.y)])
            fig.canvas.draw_idle()

    def update(frame):
        model.step()
        agents = [a for a in model.schedule.agents if a.is_active]
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
            target = next((a for a in model.schedule.agents if a.unique_id == selected_agent_id), None)
            if target and target.is_active:
                high_scat.set_offsets([(target.x, target.y)])
                update_info_display(target)
            else:
                high_scat.set_offsets(np.empty((0, 2)))
                info_panel.set_text("")

        return scat, fire_glow, fire_core, smoke_scatter, high_scat

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('pick_event', on_pick)


    annot = ax.annotate("", xy=(0,0), xytext=(10,10),
                        textcoords='offset points',
                        bbox=dict(boxstyle='round', fc='white',
                        edgecolor='black',alpha=0.8,),
                        arrowprops=dict(arrowstyle='->'))
    annot.set_visible(False)

    ax.set_axis_off()
    title_text = ax.set_title("Tudor Vladimirescu - Simulation\nRunning - Fire Mode Off", fontsize=13, fontweight='bold', pad=10)

    ani = animation.FuncAnimation(fig, update, frames=500, interval=50, blit=False)


    manager = plt.get_current_fig_manager()
    try:
        manager.window.wm_geometry("+160+20")
    except Exception as e:
        pass

    plt.tight_layout()

    try:
        plt.show()
    except KeyboardInterrupt:
        plt.close('all')