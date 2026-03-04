import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, MultiPoint
from shapely.ops import nearest_points
from config import MAP_CENTER, MAP_DIST, GEOJSON_FILE

def load_campus_map():
    G_all_raw = ox.graph_from_point(MAP_CENTER, dist = MAP_DIST, network_type='all')
    G_all = ox.project_graph(G_all_raw, to_crs="EPSG:3857")
    G_all = G_all.to_undirected()

    G_drive_raw = ox.graph_from_point(MAP_CENTER, dist = MAP_DIST, network_type='drive')
    G_drive = ox.project_graph(G_drive_raw, to_crs="EPSG:3857")
    G_drive = G_drive.to_undirected()

    nodes_proj, edges_proj = ox.graph_to_gdfs(G_all)

    print("Street map load successfully\n")

    buildings_gdf = gpd.read_file(GEOJSON_FILE)
    buildings_gdf = buildings_gdf.to_crs("EPSG:3857")
    print(f"Buildings map load successfully ({len(buildings_gdf)} buildings)\n")

    safe_nodes = [
        nodes_proj.geometry.y.idxmax(),
        nodes_proj.geometry.y.idxmin(),
        nodes_proj.geometry.x.idxmin(),
        nodes_proj.geometry.x.idxmax(),
    ]

    building_doors = {}
    if not buildings_gdf.empty:
        street_points_cloud = MultiPoint(nodes_proj.geometry.tolist())
        for idx, row in buildings_gdf.iterrows():
            poly = row.geometry
            temp_nearest_street_node = nearest_points(poly, street_points_cloud)[1]
            door_point = nearest_points(poly.boundary, temp_nearest_street_node)[0]
            building_doors[idx] = (door_point.x, door_point.y)

    return G_all, G_drive, nodes_proj, edges_proj, buildings_gdf, building_doors, safe_nodes