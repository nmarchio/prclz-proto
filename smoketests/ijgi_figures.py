from itertools import cycle, product

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import shapely
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.ops import polygonize, polygonize_full

from prclz.topology import Edge, Node, PlanarGraph


#############################################################################
# region utility functions
def clean(show = True):
    plt.gca().axis("off")
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    if show: plt.show()

def non_empty(gdf):
    return gdf[~gdf.is_empty]

def clip(geo, mask):
    return non_empty(gpd.clip(geo, mask))

def const(c):
    return lambda *args: c

def lattice(size = 4, dx = None, dy = None):
    if not dx: dx = const(0)
    if not dy: dy = const(0)
    t0 = PlanarGraph()
    size = 4
    pts = dict()
    for x in range(size):
        for y in range(size):
            pts[(x, y)] = Node((x + dx(x, y), y + dy(x, y)))

    for ((ix, iy), n) in pts.items(): 
        if (ix + 1, iy) in pts: 
            t0.add_edge(Edge((n, pts[(ix + 1, iy)])))
        if (ix, iy + 1) in pts: 
            t0.add_edge(Edge((n, pts[(ix, iy + 1)])))
    
    return t0 

def plot_planar_graph(g, ax, color, highlight = "white", z = 1, size = 100, width = 4, **kwargs):
    nlocs = {node: (node.x, node.y) for node in g.nodes}
    
    edge_kwargs = kwargs.copy()
    edge_kwargs["edge_color"] = highlight
    edge_kwargs["label"]      = "_nolegend"
    edge_kwargs["pos"]        = nlocs
    edge_kwargs["width"]      = width + 2
    edges = nx.draw_networkx_edges(g, ax = ax, **edge_kwargs)
    if edges: 
        edges.set_zorder(z-1)

    node_kwargs = kwargs.copy()
    node_kwargs["node_color"] = highlight
    node_kwargs["node_size"]  = 1.5 * size
    node_kwargs["label"]      = g.name
    node_kwargs["pos"]        = nlocs
    nodes = nx.draw_networkx_nodes(g, ax = ax, **node_kwargs)
    if nodes: 
        nodes.set_edgecolor("None")
        nodes.set_zorder(z-1)

    edge_kwargs["edge_color"] = color
    edge_kwargs["width"]      = width
    edges = nx.draw_networkx_edges(g, ax = ax, **edge_kwargs)
    if edges: 
        edges.set_zorder(z)

    node_kwargs["node_color"] = color 
    node_kwargs["node_size"]  = size
    nodes = nx.draw_networkx_nodes(g, ax = ax, **node_kwargs)
    if nodes:
        nodes.set_edgecolor("None")
        nodes.set_zorder(z)

# endregion utilityfunctions

############################################################################
# region aesthetics

## general 
sns.despine()
sns.set(palette = "muted", style = "white")
mpl.rcParams['savefig.dpi'] = 300

## ontology 
## https://coolors.co/b6db6b-56638a-000000-ffffff-808080
## https://coolors.co/292f36-5e4c73-a84c4c-c3693c-f0e100-dedad6
## aeb7c2-b7a9c6-dfb9b9-e7c3b1-ccbe00-71665b
freetown_color  = "#B6DB6B"
roads_color     = "#56638A"
buildings_color = "dimgray"
blocks_color    = "#974564"
blocks_palette  = ["#CC8FA6", "#C5819B", "#BF7390", "#B86585", "#B2577A", "#A84D70", "#974564", "#8C405D", "#7E3A54", "#70334B", "#622D41", "#542738"]
wd_palette      = ["#292f36", "#5e4c73", "#a84c4c", "#c3693c", "#f0e100", "#dedad6"]
wd_highlights   = ["#aeb7c2", "#b7a9c6", "#dfb9b9", "#e7c3b1", "#ccbe00", "#71665b"]
# wd_highlights   = ["white",    "white",  "white",   "white",   "black",   "black"]

# endregion endaesthetics 

#############################################################################
# region read in datasets
gadm          = gpd.read_file("scratch/gadm36_SLE.gpkg")
roads         = gpd.read_file("scratch/sl_test_lines.geojson")
buildings     = gpd.read_file("scratch/sle_building_linestrings.geojson")
buildings_421 = gpd.read_file("scratch/buildings_SLE.4.2.1_1.geojson")
parcels       = gpd.read_file("scratch/parcels_SLE.4.2.1_1.geojson")

buildings["geometry"] = buildings.geometry.apply(Polygon)

blocks = pd.read_csv("scratch/blocks_SLE.4.2.1_1.csv").drop("Unnamed: 0", axis = 1)
blocks = gpd.GeoDataFrame(blocks.assign(geometry = blocks["geometry"].apply(shapely.wkt.loads)), geometry="geometry")

complexity = pd.read_csv("scratch/complexity_SLE.4.2.1_1.csv").drop(columns = ["centroids_multipoint"])
complexity = gpd.GeoDataFrame(complexity.assign(geometry = complexity["geometry"].apply(shapely.wkt.loads)), geometry = "geometry")

reblock = pd.read_csv("scratch/steiner_lines_SLE.4.2.1_1.csv").drop(columns = ["Unnamed: 0", "block_w_type"]).dropna() 
reblock = gpd.GeoDataFrame(reblock.assign(geometry = reblock["geometry"].apply(shapely.wkt.loads)), geometry = "geometry")

#endregion

#############################################################################
# region figure 1: GADM in SLE boundary
natl_polygon    = gadm.dissolve(by="GID_0")
gadm_boundaries = gadm.boundary.simplify(0.00001)
natl_boundary   = natl_polygon.boundary
freetown_gadm   = gadm[gadm.GID_3 == "SLE.4.2.1_1"]
freetown_poly   = freetown_gadm.geometry.iloc[0]
minx, miny, maxx, maxy = freetown_gadm.buffer(0.12).bounds.iloc[0]
height = maxy - miny
width  = maxx - minx

# 1a main: GADMs in SLE
fig, ax = plt.subplots()
gadm_boundaries.plot(ax = ax, zorder = 0, linewidth = 2, edgecolor = "gray")
freetown_gadm  .plot(ax = ax, zorder = 1, linewidth = 0, facecolor = freetown_color)
natl_boundary  .plot(ax = ax, zorder = 2, linewidth = 3, edgecolor = "black")
patch = mpl.patches.Rectangle(xy = (minx, miny), width = width, height = height, fill = False, linewidth = 1, edgecolor = "black")
ax.add_patch(patch)
clean()

# 1a inset: Freetown zoom 
fig, ax = plt.subplots()
gadm_boundaries.plot(ax = ax, zorder = 0, linewidth = 2, edgecolor = "gray")
freetown_gadm  .plot(ax = ax, zorder = 1, linewidth = 0, facecolor = freetown_color)
natl_boundary  .plot(ax = ax, zorder = 2, linewidth = 3, edgecolor = "black")
plt.xlim(left   = minx, right = maxx)
plt.ylim(bottom = miny, top   = maxy)
plt.subplots_adjust(left=0, bottom=0.01, right=1, top=0.99)
plt.xticks([])
plt.yticks([])
plt.show()

# 1b: roads and buildings in SLE 
# define zoom box 
zoombox = box(minx = minx, maxx = maxx + width, miny = miny - height, maxy = maxy + height)

# get top n% of roads by length, buildings by area
roads_zoomed  = clip(roads, zoombox)
roads_zoomed["length"] = roads_zoomed.geometry.length
roads_sample = roads_zoomed.sort_values(by="length", ascending=False).head(int(len(roads_zoomed) * 0.2))

fig, ax = plt.subplots()
roads_sample .plot(ax = ax, zorder = 1, linewidth = 1, edgecolor = roads_color)
natl_boundary.plot(ax = ax, zorder = 2, linewidth = 2, edgecolor = "black")
plt.xlim(left   = minx, right = maxx + width)
plt.ylim(bottom = miny - height, top = maxy + height)
plt.subplots_adjust(left=0, bottom=0.01, right=1, top=0.99)
plt.xticks([])
plt.yticks([])
plt.show()

buildings_zoomed = clip(buildings[buildings.is_valid], zoombox)
buildings_zoomed["area"] = buildings_zoomed.geometry.buffer(0).area
buildings_sample = buildings_zoomed.sort_values(by = "area", ascending = False).head(int(len(buildings_zoomed) * 0.20))

fig, ax = plt.subplots()
buildings_sample.plot(ax = ax, zorder= 1, linewidth = 1, edgecolor = "gray", facecolor = "dimgray")
natl_boundary.   plot(ax = ax, zorder= 2, linewidth = 2, edgecolor = "black")
plt.xlim(left   = minx, right = maxx + width)
plt.ylim(bottom = miny - height, top = maxy + height)
plt.subplots_adjust(left=0, bottom=0.01, right=1, top=0.99)
plt.xticks([])
plt.yticks([])
plt.show()

# 1c: Freetown intersected
fig, ax = plt.subplots()
clip(roads_sample, freetown_poly)\
             .plot(ax = ax, zorder = 1, linewidth = 2, edgecolor = roads_color)
freetown_gadm.plot(ax = ax, zorder = 0, linewidth = 1, facecolor = freetown_color)
clean()

fig, ax = plt.subplots()
clip(buildings_sample, freetown_poly)\
             .plot(ax = ax, zorder = 1, linewidth = 2, edgecolor = "gray", facecolor = "dimgray")
freetown_gadm.plot(ax = ax, zorder = 0, linewidth = 1, facecolor = freetown_color)
clean()

# endregion

#############################################################################
# region figure 2: road union 

roads = non_empty(roads.intersection(freetown_gadm.iloc[0].geometry))
np.random.seed(0)
roads.sample(frac=1).plot(cmap = "viridis", linewidth=2)
clean()

roads.plot(edgecolor = roads_color, linewidth=3.5)
clean()

blocks.plot(color = [blocks_palette[_ % len(blocks_palette)] for _ in range(len(blocks))], edgecolor="white", linewidth=2)
clean()
# endregion

#############################################################################
# region figure 3: cadastal parcels
ibox_coords = [(-13.249354, 8.490069), (-13.249112, 8.475772), (-13.227738, 8.475925), (-13.227764, 8.490419)]
ibox = Polygon(ibox_coords).buffer(0)

buildings_421 = clip(buildings_421, ibox)
zoomed        = clip(parcels, ibox)
merged        = gpd.GeoDataFrame(zoomed.merge(blocks, how="left", on="block_id"), geometry="geometry_x")
diff_buffer   = merged.geometry_y.boundary.buffer(0.0001)


clipped_parcels = non_empty(merged.difference(diff_buffer))
clipped_blocks  = non_empty(merged.geometry_y.difference(diff_buffer))
buildings_diff  = clip(buildings_421, clipped_blocks)
buildings_diff  = non_empty(buildings_diff.difference(buildings_diff.boundary.buffer(0.00001)))

fig, ax = plt.subplots()
clipped_blocks.plot(ax = ax, facecolor = blocks_color, alpha = 0.2)
clipped_blocks.boundary.plot(ax = ax, edgecolor = blocks_color, linewidths = 2)
clipped_parcels.plot(ax = ax, edgecolor = blocks_color, linewidth = 1)
buildings_diff.plot(ax = ax, facecolor="gray", linewidth=0)
clean()
# endregion

#############################################################################
# region figure 4: weak dual sequences 

# 4a: schematic 
t0 = lattice()
t1 = t0.weak_dual()
t2 = t1.weak_dual()
t3 = t2.weak_dual()

graphs = [t1, t2, t3]

p0 = gpd.GeoDataFrame([Polygon([(n.x, n.y) for n in _.nodes]).convex_hull for _ in t0.trace_faces()], geometry = 0)
p0["dissolve_key"] = 0

# figure out central parcel index 
fig, ax = plt.subplots()
p0.plot(cmap="viridis", ax = ax) 
for (i, (x, y)) in p0.centroid.apply(lambda p: list(p.coords)[0]).items():
    plt.text(x, y, str(i))
clean()

# schematic parcels in block 
fig, ax = plt.subplots()
ax.set_aspect(aspect=1)
p0.plot(
    ax = ax, edgecolor = "white",      linewidths = 1, facecolor = blocks_color, alpha = 0.2, zorder = -1)
p0.iloc[4:5].drop("dissolve_key", axis=1).plot(
    ax = ax, edgecolor = "white",      linewidths = 1, facecolor = blocks_color, alpha = 0.2, zorder = -1)
p0.boundary.plot(
    ax = ax, edgecolor = blocks_color, linewidths = 1, facecolor = "none", alpha = 1, zorder = -1)
p0.dissolve(by = "dissolve_key").boundary.plot(
    ax = ax, edgecolor = blocks_color, linewidths = 2, zorder = -1, alpha = 1)
clean()

# successive weak duals with previous graphs faded
for i in range(1, len(graphs)+1):
    gs = graphs[:i]

    fig, ax = plt.subplots()
    ax.set_aspect(aspect=1)
    p0.plot(
        ax = ax, edgecolor = "white",      linewidths = 1, facecolor = blocks_color, alpha = 0.01, zorder = -1)
    p0.iloc[4:5].drop("dissolve_key", axis=1).plot(
        ax = ax, edgecolor = "white",      linewidths = 1, facecolor = blocks_color, alpha = 0.02, zorder = -1)
    p0.boundary.plot(
        ax = ax, edgecolor = blocks_color, linewidths = 1, facecolor = "none", alpha = 0.2, zorder = -1)
    p0.dissolve(by = "dissolve_key").boundary.plot(
        ax = ax, edgecolor = blocks_color, linewidths = 2, zorder = -1, alpha = 0.2)

    for (i, g) in enumerate(gs):
        plot_planar_graph(g, ax, z = 2*i, color = wd_palette[i], highlight = wd_highlights[i], alpha = 0.2 if i < len(gs)-1 else 1, width = 12, size = 400 if i < len(gs)-1 else 700)
    clean()

# superposition 
fig, ax = plt.subplots()
ax.set_aspect(aspect=1)
p0.plot(
    ax = ax, edgecolor = "white",      linewidths = 1, facecolor = blocks_color, alpha = 0.2, zorder = -1)
p0.iloc[4:5].drop("dissolve_key", axis=1).plot(
    ax = ax, edgecolor = "white",      linewidths = 1, facecolor = blocks_color, alpha = 0.2, zorder = -1)
p0.boundary.plot(
    ax = ax, edgecolor = blocks_color, linewidths = 1, facecolor = "none", alpha = 1, zorder = -1)
p0.dissolve(by = "dissolve_key").boundary.plot(
    ax = ax, edgecolor = blocks_color, linewidths = 2, zorder = -1, alpha = 1)

# set r to non-zero to shift duals down and to the right
r = 0
for i in (1, 2, 3):
    g = lattice(dx = const(i * 0), dy = const(-i * 0)) 
    for k in range(i):
        g = g.weak_dual()
    plot_planar_graph(g, ax, z = 2*i, color = wd_palette[i-1], width = 12, size = 400 if i < 3 else 500)
clean()

# endregion


# region figure 5: example graphs

## look at low k vs high k blocks
complexity.query("complexity == 3")\
          .drop(["complexity", "geometry"], axis = 1)\
          .merge(parcels, on="block_id")\
          .plot()
clean()

complexity.query("complexity == 6")\
          .drop(["complexity", "geometry"], axis = 1)\
          .merge(parcels, on="block_id")\
          .plot()
clean()

## select blocks by point
# k3_pt      = Point(-13.210037, 8.475965)
k3_pt      = Point(-13.2217172, 8.4886156)
k3_block   = complexity[complexity.contains(k3_pt)].iloc[0]
k3_parcels = parcels.query(f"block_id == '{k3_block.block_id}'")

# fix weird floating polygon 
k3_polygons = gpd.GeoDataFrame(polygonize(k3_parcels.geometry.iloc[0]), geometry = 0)
k3_polygons["dissolve_key"] = k3_polygons.index.where(~k3_polygons.index.isin((6, 22)), -1)
k3_polygons = k3_polygons.dissolve(by = "dissolve_key")

k3_polygons["boundary_key"] = 0
boundary = k3_polygons.dissolve(by = "boundary_key").boundary.iloc[0]
k3_inner = k3_polygons[~k3_polygons.touches(boundary)]

S1 = PlanarGraph.from_polygons(k3_polygons[0]).weak_dual()
S2 = S1.weak_dual()
S3 = S2.weak_dual()

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.axis("off"); ax1.set_aspect(1)
ax2.axis("off"); ax2.set_aspect(1)

k3_polygons.plot(
    ax = ax1, edgecolor = "white",      linewidths = 1, facecolor = blocks_color, alpha = 0.2, zorder = -1)
k3_inner.plot(
    ax = ax1, edgecolor = "white",      linewidths = 1, facecolor = blocks_color, alpha = 0.4, zorder = -1)
k3_polygons.boundary.plot(
    ax = ax1, edgecolor = blocks_color, linewidths = 1, facecolor = "none", alpha = 1, zorder = -1)
k3_polygons.dissolve(by = "boundary_key").boundary.plot(
    ax = ax1, edgecolor = blocks_color, linewidths = 2, zorder = -1, alpha = 1)

k3_polygons.plot(
    ax = ax2, edgecolor = "white",      linewidths = 1, facecolor = blocks_color, alpha = 0.01, zorder = -1)
k3_polygons.boundary.plot(
    ax = ax2, edgecolor = blocks_color, linewidths = 1, facecolor = "none", alpha = 0.2, zorder = -1)
k3_polygons.dissolve(by = "boundary_key").boundary.plot(
    ax = ax2, edgecolor = blocks_color, linewidths = 2, zorder = -1, alpha = 0.2)

plot_planar_graph(S1, ax2, wd_palette[0], highlight=wd_highlights[0])
plot_planar_graph(S2, ax2, wd_palette[1], highlight=wd_highlights[1])
plot_planar_graph(S3, ax2, wd_palette[2], highlight=wd_highlights[2], size = 200)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
plt.show()


# k6_pt      = Point(-13.218857, 8.459205)
k6_pt      = Point(-13.2280364, 8.4891011)
k6_block   = complexity[complexity.contains(k6_pt)].iloc[0]
k6_parcels = parcels.query(f"block_id == '{k6_block.block_id}'")
k6_polygons = gpd.GeoDataFrame(polygonize(k6_parcels.geometry.iloc[0]), geometry = 0)
k6_polygons["boundary_key"] = 0
boundary0 = k6_polygons.dissolve(by = "boundary_key").boundary.iloc[0]
k6_inner1 = k6_polygons[~k6_polygons.touches(boundary0)]
k6_inner2 = k6_inner1[~k6_inner1.touches(k6_inner1.dissolve(by = "boundary_key").boundary.iloc[0])]

T1 = PlanarGraph.from_polygons(k6_polygons[0]).weak_dual()
T2 = T1.weak_dual()
T3 = T2.weak_dual()
T4 = T3.weak_dual()
T5 = T4.weak_dual()
T6 = T5.weak_dual()

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.axis("off"); ax1.set_aspect(1)
ax2.axis("off"); ax2.set_aspect(1)

k6_polygons.plot(
    ax = ax1, edgecolor = "white",      linewidths = 1, facecolor = blocks_color, alpha = 0.2, zorder = -1)
k6_inner1.plot(
    ax = ax1, edgecolor = "white",      linewidths = 1, facecolor = blocks_color, alpha = 0.4, zorder = -1)
k6_inner2.plot(
    ax = ax1, edgecolor = "white",      linewidths = 1, facecolor = blocks_color, alpha = 0.8, zorder = -1)
k6_polygons.boundary.plot(
    ax = ax1, edgecolor = blocks_color, linewidths = 1, facecolor = "none", alpha = 1, zorder = -1)
k6_polygons.dissolve(by = "boundary_key").boundary.plot(
    ax = ax1, edgecolor = blocks_color, linewidths = 2, zorder = -1, alpha = 1)

k6_polygons.plot(
    ax = ax2, edgecolor = "white",      linewidths = 1, facecolor = blocks_color, alpha = 0.03, zorder = -1)
k6_polygons.boundary.plot(
    ax = ax2, edgecolor = blocks_color, linewidths = 1, facecolor = "none", alpha = 0.2, zorder = -1)
k6_polygons.dissolve(by = "boundary_key").boundary.plot(
    ax = ax2, edgecolor = blocks_color, linewidths = 2, zorder = -1, alpha = 0.2)

plot_planar_graph(T1, ax2, wd_palette[0], highlight=wd_highlights[0])
plot_planar_graph(T2, ax2, wd_palette[1], highlight=wd_highlights[1])
plot_planar_graph(T3, ax2, wd_palette[2], highlight=wd_highlights[2])
plot_planar_graph(T4, ax2, wd_palette[3], highlight=wd_highlights[3])
plot_planar_graph(T5, ax2, wd_palette[4], highlight=wd_highlights[4])
plot_planar_graph(T6, ax2, wd_palette[5], highlight=wd_highlights[5], size = 200)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
plt.show()

# endregion

# region figure 6: reblock

k3_roads = clip(roads, k3_block.geometry.buffer(0.0006))

fig, ax = plt.subplots()
ax.axis("off"); ax.set_aspect(1)

k3_roads.plot(ax = ax, color = roads_color, linewidth = 6)
reblock[(reblock.block == "SLE.4.2.1_1_1765") & (reblock.line_type == "new_steiner")]\
    .plot(ax = ax, color = wd_palette[-1], linewidth = 6)
reblock[(reblock.block == "SLE.4.2.1_1_1765") & (reblock.line_type == "new_steiner")]\
    .plot(ax = ax, color = roads_color, linewidth = 4)

k3_polygons.plot(
    ax = ax, edgecolor = "white",      linewidths = 1, facecolor = blocks_color, alpha = 0.2, zorder = -1)
k3_inner.plot(
    ax = ax, edgecolor = "white",      linewidths = 1, facecolor = blocks_color, alpha = 0.4, zorder = -1)
k3_polygons.boundary.plot(
    ax = ax, edgecolor = blocks_color, linewidths = 1, facecolor = "none", alpha = 1, zorder = -1)
k3_polygons.dissolve(by = "boundary_key").boundary.plot(
    ax = ax, edgecolor = blocks_color, linewidths = 2, zorder = -1, alpha = 1)

clean()

# k6_roads     = clip(roads,         k6_block.geometry.buffer(0.00025))
k6_buildings = clip(buildings_421, k6_block.geometry)
k6_raw_steiner = reblock[(reblock.block == k6_block.block_id) & (reblock.line_type == "new_steiner")].explode()

# k6_buildings0 = k6_buildings.copy()
k6_buildings = k6_buildings0.copy()

k6_buildings["area"] = k6_buildings.area
k6_buildings.sort_values("area", inplace = True, ascending=False)
k6_buildings.reset_index(inplace = True)
k6_buildings.drop("index", axis = 1, inplace=True)
k6_buffered = k6_buildings.boundary.buffer(0.00002)
k6_buildings_shrunk = non_empty(k6_buildings.difference(k6_buffered))
k6_buildings["geometry"] = k6_buildings["geometry"].where(k6_buildings.index > 40, k6_buildings_shrunk.geometry)
k6_buildings["geometry"] = k6_buildings["geometry"].where(k6_buildings.index > 20, k6_buffered.geometry)

fig, ax = plt.subplots()
ax.axis("off"); ax.set_aspect(1)
k6_buildings_buffered.plot(ax = ax, color = "blue")
k6_buildings.plot(ax = ax, color = "blue")
clean()

fig, ax = plt.subplots()
ax.axis("off"); ax.set_aspect(1)
# k6_roads.plot(ax = ax, color = roads_color, linewidth = 3, zorder = -2, alpha = 0.5)
# reblock[(reblock.block == k6_block.block_id) & (reblock.line_type == "new_steiner")]\
#     .plot(ax = ax, color = "white", linewidth = 4)
reblock[(reblock.block == k6_block.block_id) & (reblock.line_type == "new_steiner")]\
    .plot(ax = ax, color = roads_color, linewidth = 3, capstyle="round")
reblock[(reblock.block == k6_block.block_id) & (reblock.line_type == "new_steiner")]\
    .plot(ax = ax, color = "white"   , linewidth = 2.5, capstyle="round")

k6_polygons.plot(
    ax = ax, edgecolor = "white",      linewidths = 1, facecolor = blocks_color, alpha = 0.2, zorder = -1)
k6_inner1.plot(
    ax = ax, edgecolor = "white",      linewidths = 1, facecolor = blocks_color, alpha = 0.4, zorder = -1)
k6_inner2.plot(
    ax = ax, edgecolor = "white",      linewidths = 1, facecolor = blocks_color, alpha = 0.8, zorder = -1)
k6_polygons.boundary.plot(
    ax = ax, edgecolor = blocks_color, linewidths = 1, facecolor = "none", alpha = 1, zorder = -1)
k6_polygons.dissolve(by = "boundary_key").boundary.plot(
    ax = ax, edgecolor = roads_color, linewidths = 3, zorder = 10, alpha = 1)

# k6_buildings.plot(ax = ax, facecolor="gray", linewidth=0)
k6_buildings_shrunk.iloc[:32].plot(ax = ax, facecolor=buildings_color, linewidth=0)
k6_buildings.iloc[32:].plot(ax = ax, facecolor=buildings_color, linewidth=0)
# k6_buildings_shrunk.iloc[:21].plot(ax = ax, facecolor="dimgray", linewidth=0)
# k6_buildings.iloc[21:41].plot(ax = ax, facecolor="dimgray", linewidth=0)
# k6_buffered.iloc[41:].plot(ax = ax, facecolor="dimgray", linewidth=0)
clean()

# endregion