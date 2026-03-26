import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, MultiPoint, LineString
from shapely.ops import nearest_points, unary_union
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

    dorms_gdf = gpd.read_file(GEOJSON_FILE)
    dorms_gdf = dorms_gdf.to_crs("EPSG:3857")
    dorms_gdf['is_dorm'] = True

    buildings_gdf = dorms_gdf.copy()
    print(f"Buildings map load successfully, {len(buildings_gdf)} dorms loaded\n")

    all_buildings_shape = unary_union(buildings_gdf.geometry)
    edges_to_remove = []


    for u,v,k,data in G_all.edges(data=True, keys=True):
        if 'geometry' in data:
            edge_geom = data['geometry']
        else:
            node_u = nodes_proj.loc[u]
            node_v = nodes_proj.loc[v]
            edge_geom = LineString([(node_u.geometry.x, node_u.geometry.y), (node_v.geometry.x, node_v.geometry.y)])

        if edge_geom.intersects(all_buildings_shape):
            overleap = edge_geom.intersection(all_buildings_shape)
            if hasattr(overleap, 'length') and overleap.length > 3.0:
                edges_to_remove.append((u, v, k))

    for u,v,k in edges_to_remove:
        G_all.remove_edge(u, v, key=k)

    largest_cc = max(nx.connected_components(G_all), key=len)
    G_all = G_all.subgraph(largest_cc).copy()

    single_nodes = [n for n, d in G_all.degree() if d==0]
    G_all.remove_nodes_from(single_nodes)

    nodes_proj, edges_proj = ox.graph_to_gdfs(G_all)

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