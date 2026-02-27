import matplotlib.pyplot as plt
import contextily as ctx
from map_loader import *

if __name__ == '__main__':
    #preview harta, de facut functii separate mai incolo (Mesa)
    #momentan doar pt testare

    G_w, G_d, nodes, edges, n_list, buildings, doors, safe = load_campus_map()

    fig, ax = plt.subplots(figsize=(10,10))
    if not buildings.empty:
        buildings.plot(ax=ax, color='#a67c52', edgecolor='black', alpha=0.7, label='Dorms')
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zorder=0)
    edges.plot(ax=ax, color='#bdc3c7', linewidth=0.5, alpha=0.5, label='Streets', zorder=1)
    door_x=[coords[0] for coords in doors.values()]
    door_y=[coords[1] for coords in doors.values()]
    ax.scatter(door_x, door_y, color='red', s=10, label='Doors', zorder=5)
    safe_x = nodes.loc[safe].geometry.x
    safe_y = nodes.loc[safe].geometry.y
    ax.scatter(safe_x, safe_y, color='green', s=50, marker='X', label='Safe', zorder=6)
    plt.title("Map Preview - TUIASI Campus")
    plt.legend()
    plt.show()