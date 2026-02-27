import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, MultiPoint
from shapely.ops import nearest_points
from config import MAP_CENTER, MAP_DIST, GEOJSON_FILE

def load_campus_map():
    G_walk_raw = ox.graph_from_point(MAP_CENTER, dist = MAP_DIST, network_type='all')
    G_walk = ox.project_graph(G_walk_raw, to_crs="EPSG:3857")
    G_walk = G_walk.to_undirected()

    nodes_walk_proj, edges_walk_proj = ox.graph_to_gdfs(G_walk)
    walk_nodes_list = list(G_walk.nodes())

    G_drive_raw = ox.graph_from_point(MAP_CENTER, dist = MAP_DIST, network_type='drive')
    G_drive = ox.project_graph(G_drive_raw, to_crs="EPSG:3857")
    G_drive = G_drive.to_undirected()

    print("Street map load successfully\n")

    buildings_gdf = gpd.read_file(GEOJSON_FILE)
    buildings_gdf = buildings_gdf.to_crs("EPSG:3857")
    print(f"Buildings map load successfully ({len(buildings_gdf)} buildings)\n")


    all_x = nodes_walk_proj.geometry.x
    all_y = nodes_walk_proj.geometry.y
    safe_nodes = [
        nodes_walk_proj.index[all_y.argmax()],
        nodes_walk_proj.index[all_y.argmin()],
        nodes_walk_proj.index[all_x.argmax()],
        nodes_walk_proj.index[all_x.argmin()],
    ]

    street_points_cloud = MultiPoint([Point(x,y) for x, y in zip(all_x, all_y)])
    building_doors = {}
    for idx, row in buildings_gdf.iterrows():
        poly = row.geometry
        temp_nearest_street_node = nearest_points(poly, street_points_cloud)[1]
        door_point = nearest_points(poly.boundary, temp_nearest_street_node)[0]
        building_doors[idx] = (door_point.x, door_point.y)


    return G_walk, G_drive, nodes_walk_proj, edges_walk_proj, walk_nodes_list, buildings_gdf, building_doors, safe_nodes