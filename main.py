import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import contextily as ctx
from map_loader import *
from simulation_model import CampusModel
from config import TARGET_POPULATION_MIN, TARGET_POPULATION_MAX

if __name__ == '__main__':

    G_all, G_drive, nodes, edges, buildings, doors, safe = load_campus_map()

    n_stud = random.randint(TARGET_POPULATION_MIN, TARGET_POPULATION_MAX)
    model = CampusModel(G_all, G_drive, nodes, doors, n_students=n_stud)

    fig, ax = plt.subplots(figsize=(12,12))

    if not buildings.empty:
        buildings.plot(ax=ax, color='#a67c52', edgecolor='black', alpha=0.7, label='Dorms')

    edges.plot(ax=ax, color='#bdc3c7', linewidth=0.5, alpha=0.5, zorder=1)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zorder=0)

    scat = ax.scatter([], [], c='blue', s=20, zorder=5, label='Students')

    def update(frame):
        model.step()
        active_agents = [a for a in model.schedule.agents if a.is_active and not a.is_dead]
        if active_agents:
            offsets = [(a.x, a.y) for a in active_agents]
            scat.set_offsets(offsets)
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=500, interval=50, blit=True)
    plt.title("Simulation")
    plt.legend()
    plt.show()