# SHOULD BE WITH prclz-proto/prclz/ folder

from numpy import linspace
from scipy.stats.kde import gaussian_kde
import pandas as pd 
import geopandas as gpd 
from pathlib import Path 
from shapely.wkt import loads

import numpy as np 
from view import load_aoi, make_bokeh, read_steiner 

from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, FixedTicker, PrintfTickFormatter, NumeralTickFormatter
from bokeh.plotting import figure, save
from bokeh.layouts import layout, column, gridplot, row 
from bokeh.models.annotations import Title
from bokeh.models.tools import WheelZoomTool
from bokeh.io import export_png, export_svgs

import statsmodels.api as sm

def add_title(fig, main_title=None, sub_title=None):

    if sub_title is not None:
        fig.add_layout(Title(text=sub_title, text_color='black', text_font_size='8pt', text_font_style='italic'), 'above')

    if main_title is not None:
        fig.add_layout(Title(text=main_title, text_color='black',  text_font_size='16pt'), 'above')

def ridge(category, data, scale=20):
    l = list(zip([category]*len(data), scale*data))
    l.insert(0, (category, 0))
    l.insert(1, (category, 0))
    return l

MNP = ["#0571b0", "#f7f7f7", "#f4a582", "#cc0022"]
#MNP = ["#cc0022", "#f4a582", "#f7f7f7", "#0571b0"]
PALETTE = MNP
#RANGE = [(0,1), (2,3), (4,7), (8, np.inf)]
RANGE = [1, 3, 7, np.inf]

def get_color(complexity):
    for upper, color in zip(RANGE, PALETTE):
        if complexity <= upper:
            return color 

HEIGHT = 900
WIDTH = 900

def freetown_df():
    '''
    Hacky function to merge complexity and reblocking
    for Freetown
    '''
    compl_path = Path("../data/complexity/Africa/SLE/complexity_SLE.4.2.1_1.csv") 
    complexity_df = gpd.GeoDataFrame(pd.read_csv(compl_path))
    complexity_df['geometry'] = complexity_df['geometry'].apply(loads)
    complexity_df.geometry.crs = {'init': 'epsg:4326'}  
    complexity_df.crs = {'init': 'epsg:4326'}
    complexity_df['block_ext_len'] = complexity_df['geometry'].boundary.to_crs({'init': 'epsg:3395'}).length

    reblock_path = Path("../data/reblock/Africa/SLE/steiner_lines_SLE.4.2.1_1.csv")
    reblock_df = read_steiner(reblock_path, True) 
    reblock_df.rename(columns={'block': 'block_id'}, inplace=True)

    fn = lambda x: "new" if "new" in x else "existing"
    reblock_df['type'] = reblock_df['line_type'].apply(fn)
    long_df = reblock_df[['type', 'road_len_m', 'block_id']].pivot(index='block_id', columns='type', values='road_len_m') 
    long_df['existing'] = long_df['existing'].fillna(0) / 1000
    long_df['new'] = long_df['new'].fillna(0) / 1000


    reblock_comp = long_df.merge(complexity_df[['block_id', 'complexity']], how='left', on='block_id')
    return reblock_comp 

def make_filler_fig():
    filler = figure(toolbar_location=None, background_fill_color='blue', background_fill_alpha=0.25,  border_fill_color='blue', border_fill_alpha=0.25, plot_width=100, plot_height=900)    
    filler.yaxis.visible = False
    filler.xaxis.visible = False
    filler.text(x=[0], y=[1], text=["."]) 
    return filler 

def make_freetown_summary():
    #p = Path("../data/LandScan_Global_2018/freetown_w_reblock.csv")
    #df = pd.read_csv(str(p))

    df = freetown_df()
    df['existing'] = df['existing'].where(df['complexity']>2, other=0)
    df['new'] = df['new'].where(df['complexity']>2, other=0)


    cur_df = df[['complexity', 'existing', 'new']].groupby('complexity').sum()
    cur_df['total'] = cur_df['existing'] + cur_df['new']   
    cur_df.reset_index(inplace=True)
    cur_df['color'] = cur_df['complexity'].apply(get_color)

    size = 900
    total_new = int(cur_df['new'].sum())
    total_exist = int(cur_df['existing'].sum())
    grand_total = total_new + total_exist
    main_title = "Estimated road length for universal access"
    sub_title = "Total existing={:,}, Total new={:,}, Grand total={:,}".format(total_exist, total_new, grand_total)
    fig = figure(x_range=(0, cur_df['complexity'].max()+1), toolbar_location='above', border_fill_color='blue', border_fill_alpha=0.25, 
                plot_height=size, plot_width=size)
    add_title(fig, main_title=main_title, sub_title=sub_title)

    cur_source = ColumnDataSource(cur_df)

    line_types = ['existing', 'new']
    fig.vbar_stack(line_types, hatch_pattern=[' ', 'spiral'], x='complexity', width=1.0, 
                   color='color', source=cur_source, line_color='black', legend_label=line_types)
    fig.y_range.start = 0
    fig.axis.minor_tick_line_color = None

    fig.yaxis.axis_label = 'Total road length (km)'
    fig.yaxis.axis_label_text_font_style = 'bold'
    fig.yaxis.axis_label_text_font_size = '14pt'
    fig.yaxis.axis_label_text_color = 'black'
    fig.yaxis.major_label_text_font_size = '12pt'
    fig.yaxis.major_label_text_font_style = 'bold'
    fig.yaxis.major_label_text_color = 'black'
 
    fig.xaxis.axis_label = 'Block complexity'
    fig.xaxis.axis_label_text_font_style = 'bold'
    fig.xaxis.axis_label_text_font_size = '14pt'
    fig.xaxis.axis_label_text_color = 'black'
    fig.xaxis.major_label_text_font_size = '12pt'
    fig.xaxis.major_label_text_font_style = 'bold'
    fig.xaxis.major_label_text_color = 'black'
    fig.xaxis[0].ticker.desired_num_ticks = cur_df['complexity'].max()+2

    return fig 

# Load main dataframe
def make_ridge_plot_w_examples(aoi_name, file_path, output_filename, add_observations=True, bandwidth=.05, block_list=[]):

    output_file(output_filename)

    max_density = 1
    probly_df = load_aoi(file_path)

    missing_compl = probly_df['complexity'].isna()
    probly_df = probly_df.loc[~missing_compl]
    probly_df['count'] = 1
    probly_df_gb = probly_df.groupby('complexity')

    cats_int = np.arange(probly_df['complexity'].max()+1).astype('uint8')
    cats_str = ["Complexity {}".format(i) for i in cats_int]
    int_to_str = {i:s for i, s in zip(cats_int, cats_str)}
    str_to_int = {s:i for i, s in zip(cats_int, cats_str)}


    #x = linspace(-20,110, 500)
    target_col = 'bldg_density'
    #x = np.linspace(-10, 10, 500)
    x = np.linspace(0,max_density,500)
    x_prime = np.concatenate([np.array([0]), x, np.array([1])])
    SCALE = .35 * max_density

    #source = ColumnDataSource(data=dict(x=x))
    source = ColumnDataSource(data=dict(x=x_prime))

    title = "Block building density and\nblock complexity: {}".format(aoi_name)
    size = 900

    # Make the main figure
    p = figure(toolbar_location='above', border_fill_color='blue', border_fill_alpha=0.25, y_range=cats_str, plot_height=size, plot_width=size, x_range=(0, 1.0))
    add_title(p, main_title=title, sub_title='Distribution of block density by complexity level')
    # p.title.text_font_size = '20pt'
    # p.title.text_font_style = 'bold'
    # p.title.text_color = 'black'

    # Now make the histogram count figure
    obs_count = probly_df_gb.sum()[['count']].reset_index()
    obs_count['complexity_str'] = obs_count['complexity'].apply(lambda x: int_to_str[x] )
    print(obs_count)
    hist = figure(toolbar_location=None, border_fill_color='blue', border_fill_alpha=0.25, plot_width=100, plot_height=p.plot_height, y_range=p.y_range,
               x_range=(0, obs_count['count'].max()))
    hist.hbar(y='complexity_str', right='count', source=obs_count, height=1, line_color=None, fill_color='black', fill_alpha=.5)
    add_title(hist, sub_title='Complexity hist.')
    hist.ygrid.grid_line_color = None
    hist.yaxis.visible = False
    hist.xaxis[0].ticker.desired_num_ticks = 5
    hist.xaxis.major_label_orientation = np.pi/4
    hist.xaxis.minor_tick_line_color = None 
    hist.xaxis.major_label_text_color = 'black'

    for i, cat_s in enumerate(reversed(cats_str)):

        cat_i = str_to_int[cat_s]
        if cat_i not in probly_df_gb.groups.keys():
            p.line([0, 1], [cat_s, cat_s], line_color='black')
            continue 

        #if cat_i not in probly_df.groups
        cat_data = probly_df_gb.get_group(cat_i)['bldg_density'].values
        cat_x = [cat_s]*len(cat_data)
        # Add circles for observations
        if add_observations:
            p.circle(cat_data, cat_x, fill_alpha=0.5, size=5, fill_color='black')

        print("Processing cat = {}".format(cat_i))
        print("shape = {}".format(cat_data.shape))
        if cat_data.shape[0] == 1:
            p.line([0, 1], [cat_s, cat_s], line_color='black')
            continue 

        #pdf = gaussian_kde(cat_data)
        kernel_density = sm.nonparametric.KDEMultivariate(data=cat_data, var_type='c', bw=[bandwidth])
        y = ridge(cat_s, kernel_density.pdf(x), SCALE)
        source.add(y, cat_s)
        p.patch('x', cat_s, color=get_color(cat_i), alpha=0.6, line_color="black", source=source)

        # Get count for cat_s
        #print(obs_count)
        #cat_bool = obs_count['complexity']==cat_i  
        #cur_count = obs_count['count'].loc[cat_bool].item()
        #print("x = {} y = {}".format(cur_count, cat_s))
        #hist.circle(x=[cur_count], y=[cat_s], size=100, fill_color='black')   

    p.outline_line_color = None

    p.xaxis.ticker = FixedTicker(ticks=list(np.linspace(0, max_density, 11)))

    p.ygrid.grid_line_color = None
    p.xgrid.grid_line_color = "#dddddd"
    #p.xgrid.ticker = p.xaxis.ticker

    p.axis.minor_tick_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.axis_line_color = None
    p.xaxis[0].formatter = NumeralTickFormatter(format="0%")

    # p.yaxis.axis_label = 'Block density'
    # p.yaxis.axis_label_text_font_style = 'bold'
    # p.yaxis.axis_label_text_font_size = '14pt'
    p.yaxis.major_label_text_font_size = '12pt'
    p.yaxis.major_label_text_font_style = 'bold'
    p.yaxis.major_label_text_color = 'black'
 
    p.xaxis.axis_label = 'Block density'
    p.xaxis.axis_label_text_font_style = 'bold'
    p.xaxis.axis_label_text_font_size = '14pt'
    p.xaxis.axis_label_text_color = 'black'
    p.xaxis.major_label_text_font_size = '12pt'
    p.xaxis.major_label_text_font_style = 'bold'
    p.xaxis.major_label_text_color = 'black'
    #p.x_range.range_padding = 0.1

    # Plot a 1,2,3,4 int on the main plot signalling where the example block is
    text_x = []
    text_y = []
    text = []
    for i, block_id in enumerate(block_list):
        block_obs = probly_df[probly_df['block_id']==block_id].iloc[0]
        text_x.append( block_obs['bldg_density'] )
        text_y.append( "Complexity {}".format(block_obs['complexity']) )
        text.append(str(i+1))
    p.text(x=text_x, y=text_y, text=text, angle=0)

    # Add subplots
    sub_layout_list = make_block_example_grid(probly_df, HEIGHT, WIDTH, block_list)
    columns = 1
    toolbar_location = None 
    grid = gridplot(children=sub_layout_list, ncols=columns, toolbar_location=toolbar_location)

    # Add subplots -- reblocked
    sub_layout_list_reblocked = make_block_example_grid(probly_df, HEIGHT, WIDTH, block_list, add_reblock=True, region='Africa')
    grid_reblocked = gridplot(children=sub_layout_list_reblocked, ncols=columns, toolbar_location=toolbar_location)    


    fig_list = [p, hist, grid]
    counts_df = obs_count
    #return fig_list, counts_df

    #final_layout = row([p, hist, grid])
    fig10 = figure(plot_height=p.height, plot_width=p.width)
    fig11 = figure(plot_height=hist.height, plot_width=hist.width)

    #right_plot = column([grid, grid_reblocked]) if columns==2 else row([grid, grid_reblocked])
    upper_left = row([p, hist])

    if aoi_name == "Freetown":
        left_plot = column([upper_left, row([make_freetown_summary(), make_filler_fig()])])
    elif aoi_name == "Monrovia":
        probly_df.sort_values('bldg_density', inplace=True)  
        blocks15 = list(probly_df[probly_df['complexity']==13]['block_id'].values)
        blocks8 = list(probly_df[probly_df['complexity']==8]['block_id'].values)
        block_list = [blocks8[0], blocks8[-1], blocks15[0], blocks15[-1]]

        plot_list = make_block_example_grid(probly_df, HEIGHT, WIDTH, block_list, add_reblock=False)
        mon_grid = gridplot(children=plot_list, ncols=2, toolbar_location=None)
        left_plot = column([upper_left, row([mon_grid, make_filler_fig()])])
    else:
        left_plot = column([ upper_left])

    #final_layout = row([left_plot, right_plot])
    final_layout = row([left_plot, grid_reblocked])

    #final_layout = gridplot([[p, hist, grid], 
    #                       [bottom_fig, grid_reblocked]])
    #show(final_layout)
    #return fig_list, counts_df, probly_df

    save(final_layout, output_filename)
    #return final_layout
    final_layout.background = 'white'
    export_png(final_layout, str(output_filename).replace("html", "png"))

    #export_svgs(final_layout, str(output_filename).replace("html", "svg"))
    return final_layout


def make_block_example_grid(aoi_gdf, total_height, total_width, block_list, 
                            force_title=None, add_cluster=False, add_reblock=False, region=None ):
    '''
    Start by assuming a 2x2 grid of examples
    '''

    if add_reblock:
        assert region is not None, "If adding reblocking must also provide region, i.e. 'Africa' "

    # Add other figs
    r = 2
    c = 2
    height = total_height // r 
    width = total_width // c 

    flat_l = []

    for i, block in enumerate(block_list):
        cur_r = i // r 
        cur_c = i % c 
        print("Plotting block {} index {} at ({},{})".format(block, i, cur_r, cur_c))
        is_block = aoi_gdf['block_id']==block 
        cur_gdf = aoi_gdf.loc[is_block]
        cur_density = cur_gdf['bldg_density'].iloc[0]
        cur_complexity = cur_gdf['complexity'].iloc[0]
        cur_count = cur_gdf['bldg_count'].iloc[0]
        if add_cluster:
            cur_cluster = cur_gdf['cluster'].iloc[0]

        #fig = make_bokeh(cur_gdf, plot_height=height, plot_width=width, bldg_alpha=1.0)
        fig, new_road_length = make_bokeh(cur_gdf, plot_height=height, plot_width=width, bldg_alpha=1.0, add_reblock=add_reblock, region=region)
        fig.toolbar.active_scroll = WheelZoomTool()
        #t = Title()

        if cur_complexity <= 2:
            new_road_length = 0

        if add_reblock:
            if new_road_length is None:
                new_road_length = "???"
                sub_title_text = "[{}] Compl. = {}; Density = {}%; Bldg. count = {:,}; New road len = {}m".format(i+1, cur_complexity, np.round(cur_density*100), int(cur_count), new_road_length)
            else:
                #sub_title_text = "[{}] Complexity = {}; Bldg. density = {}%; Bldg. count = {:,}; New road len = {:,}m".format(i+1, cur_complexity, np.round(cur_density*100), int(cur_count), new_road_length)
                sub_title_text = "[{}] Compl. = {}; Density = {}%; Bldg. count = {:,}; New road len = {:,}m".format(i+1, cur_complexity, np.round(cur_density*100), int(cur_count), new_road_length)

        else:
            if add_cluster:
                sub_title_text = "[{}] Complexity = {}; Bldg. density = {}%; Cluster = {:,}".format(i+1, cur_complexity, np.round(cur_density*100), int(cur_cluster))
            else:
                sub_title_text = "[{}] Complexity = {}; Bldg. density = {}%; Bldg. count = {:,}".format(i+1, cur_complexity, np.round(cur_density*100), int(cur_count))
        #sub_title_text = "[{}] Complexity = {}; Bldg. density = {}%; Bldg. count = {:,}; New road len = {:,}m".format(i+1, cur_complexity, np.round(cur_density*100), int(cur_count), new_road_length)

        fig.xaxis.visible = False
        fig.yaxis.visible = False 
        fig.title.text_color = 'black' 

        # Main title for first observation only
        t = "after" if add_reblock else "before"

        if force_title is None:
            main_title_text = "Example blocks {} reblocking".format(t) if i==0 else None 
        else:
            main_title_text = force_title
        add_title(fig, main_title=main_title_text, sub_title=sub_title_text)

        # Add the plot to the below
        flat_l.append(fig)

    return flat_l
    #return sub_layout

# Load main dataframe
def make_ridge_plot(aoi_name, file_path, output_filename, bandwidth=.05, add_observations=True):
    #aoi_name = "Nairoi"
    #file_path = "mnp/prclz-proto/data/LandScan_Global_2018/aoi_datasets/analysis_nairobi.csv"
    #bandwidth = 0.05
    #add_observations = True
    #output_filename = 
    output_file(output_filename)

    max_density = 1
    probly_df = pd.read_csv(file_path) 
    probly_df['bldg_density'] = max_density*probly_df['total_bldg_area_sq_km']/probly_df['block_area_km2']
    missing_compl = probly_df['complexity'].isna()
    probly_df = probly_df.loc[~missing_compl]
    probly_df_gb = probly_df.groupby('complexity')
    #print(probly_df_gb.groups.keys())

    #cats = list(reversed(probly.keys()))
    cats_int = np.arange(probly_df['complexity'].max()+1).astype('uint8')
    cats_str = ["Complexity {}".format(i) for i in cats_int]
    int_to_str = {i:s for i, s in zip(cats_int, cats_str)}
    str_to_int = {s:i for i, s in zip(cats_int, cats_str)}

    #palette = [cc.rainbow[i*15] for i in range(17)]

    #x = linspace(-20,110, 500)
    target_col = 'bldg_density'
    #x = np.linspace(-10, 10, 500)
    x = np.linspace(0,max_density,500)
    x_prime = np.concatenate([np.array([0]), x, np.array([1])])
    SCALE = .35 * max_density

    #source = ColumnDataSource(data=dict(x=x))
    source = ColumnDataSource(data=dict(x=x_prime))

    title = "Building density - {}".format(aoi_name)
    p = figure(y_range=cats_str, plot_height=900, plot_width=900, x_range=(0, 1.0), title=title)

    for i, cat_s in enumerate(reversed(cats_str)):

        cat_i = str_to_int[cat_s]
        if cat_i not in probly_df_gb.groups.keys():
            continue 

        #if cat_i not in probly_df.groups
        cat_data = probly_df_gb.get_group(cat_i)['bldg_density'].values
        cat_x = [cat_s]*len(cat_data)
        # Add circles for observations
        if add_observations:
            p.circle(cat_data, cat_x, fill_alpha=0.5, size=5, fill_color='black')

        print("Processing cat = {}".format(cat_i))
        print("shape = {}".format(cat_data.shape))
        if cat_data.shape[0] == 1:
            continue 
        #pdf = gaussian_kde(cat_data)
        kernel_density = sm.nonparametric.KDEMultivariate(data=cat_data, var_type='c', bw=[bandwidth])
        y = ridge(cat_s, kernel_density.pdf(x), SCALE)
        source.add(y, cat_s)
        #p.patch('x', cat_s, color=palette[i], alpha=0.6, line_color="black", source=source)
        p.patch('x', cat_s, color=get_color(cat_i), alpha=0.6, line_color="black", source=source)
     

    #p00.circle('complexity', 'bldg_density', fill_alpha=0.5, size=10, hover_color="firebrick")

    p.outline_line_color = None
    p.background_fill_color = "#efefef"

    p.xaxis.ticker = FixedTicker(ticks=list(np.linspace(0, max_density, 11)))
    #p.xaxis.formatter = PrintfTickFormatter(format="%d%%")

    p.ygrid.grid_line_color = None
    p.xgrid.grid_line_color = "#dddddd"
    #p.xgrid.ticker = p.xaxis.ticker

    p.axis.minor_tick_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.axis_line_color = None    

    #p.y_range.range_padding = 0.12
    save(p, output_filename)





if __name__ == "__main__":
    aoi_dir = Path("../data/LandScan_Global_2018/aoi_datasets/")
    # make_ridge_plot(aoi_name, file_path, output_filename)

    aoi_names = ['Freetown', 'Monrovia', 'Kathmandu', 'Nairobi', 'Port au Prince']
    stubs = ['freetown', 'greater_monrovia', 'kathmandu', 'nairobi', 'port_au_prince']
    file_names = ['analysis_{}.csv'.format(n) for n in stubs]
    file_paths = [aoi_dir / f for f in file_names]
    output_dir = Path("./ridge_plots_examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filenames = [output_dir / (n.replace(".csv", "_ridge.html")) for n in file_names]

    # block_map = {
    #      'Nairobi': ['KEN.30.3.3_1_59', 'KEN.30.17.4_1_63', 'KEN.30.10.2_1_27', 'KEN.30.16.1_1_3'],
    #     'Kathmandu': ['NPL.1.1.3.31_1_343', 'NPL.1.1.3.14_1_68', 'NPL.1.1.3.31_1_3938', 'NPL.1.1.3.31_1_1253'],
    #     'Monrovia': ['LBR.11.2.1_1_2563', 'LBR.11.2.1_1_282', 'LBR.11.2.1_1_1360', 'LBR.11.2.1_1_271'],
    #     'Freetown': ['SLE.4.2.1_1_1060', 'SLE.4.2.1_1_1280', 'SLE.4.2.1_1_693', 'SLE.4.2.1_1_1870']
    # }
    block_map = {
        'Monrovia': ['LBR.11.2.1_1_2563', 'LBR.11.2.1_1_282', 'LBR.11.2.1_1_1360', 'LBR.11.2.1_1_271'],
        'Freetown': ['SLE.4.2.1_1_1060', 'SLE.4.2.1_1_1280', 'SLE.4.2.1_1_693', 'SLE.4.2.1_1_1870']
    }
    # block_map = {
    #     'Freetown': ['SLE.4.2.1_1_1060', 'SLE.4.2.1_1_1280', 'SLE.4.2.1_1_693', 'SLE.4.2.1_1_1870']
    # }
    for aoi_name, file_path, output_filename in zip(aoi_names, file_paths, output_filenames):
        print("aoi_name = {}".format(aoi_name))
        print("file_path = {}".format(file_path))
        print("output_filename = {}".format(output_filename))

        #make_ridge_plot(aoi_name, file_path, output_filename)

        if aoi_name not in block_map.keys():
            continue
        else:
            block_list = block_map[aoi_name]
            make_ridge_plot_w_examples(aoi_name, file_path, output_filename, block_list=block_list)
        # if "Freetown" in aoi_name:
        #     t = make_ridge_plot_w_examples(aoi_name, file_path, output_filename, block_list=block_list)

        if "Monrovia" in aoi_name:
            block_list = block_map[aoi_name]
            t = make_ridge_plot_w_examples(aoi_name, file_path, output_filename, block_list=block_list)
            

