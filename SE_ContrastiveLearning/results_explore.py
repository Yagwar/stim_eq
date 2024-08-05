import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
plt.set_cmap("Greys")
from time import localtime, strftime
import networkx as nx
import math
import itertools
import pickle
import os
import seaborn as sns
from matplotlib.colors import BoundaryNorm, ListedColormap
pd.options.display.float_format = "{:,.3f}".format

from itertools import product

###### Colors Plot
colors_dict={"color_pass":"#1E88E5",#colorblind_blue#"#313695",#blue#
             "color_soft_pass":"#744C94FF",#colorblind_orange_blue#"#FFC107",#colorblind_yellow#
             "color_random_fail":"#EC6E34",#colorblind_yellow_red#
             "color_fail":"#D81B60",#colorblind_red#"#d73027",#red#
            }

def get_agent_graph_dat(agent_info_df, metric_column):
    agent_accuracy_member_df=pd.pivot_table(
        agent_info_df, 
        index=['sample_subset',
               'sample_member',
               'comparison_member'
              ], 
        values=metric_column,
        aggfunc="mean"
    ).reset_index()
    return agent_accuracy_member_df

def plot_heatmap_dat(
    graph_dat_plot, 
    metric_column,
    minimal_cut_value,
    random_level,
    ax_plot,
    soft_cut_value=.7    
):
    my_colors = [colors_dict["color_fail"], 
                 colors_dict["color_random_fail"], 
                 colors_dict["color_soft_pass"], 
                 colors_dict["color_pass"]]
    my_cmap = ListedColormap(my_colors)
    my_bounds = [.0, 
                 random_level,
                 soft_cut_value,
                 minimal_cut_value, 
                 1.]
    my_norm = BoundaryNorm(my_bounds, my_cmap.N)
    graph_dat_heatmap=graph_dat_plot.pivot(
        index='sample_member', 
        columns='comparison_member',
        values=metric_column)
    plt.figure(figsize=(10,8))#(8,6))#(14,6))#
    sns.heatmap(graph_dat_heatmap,#activations_pivot,
                cmap=my_cmap,
                norm=my_norm,
                vmin=0, vmax=1, linewidth=1, linecolor='#bababa',
                annot=True, fmt=".3f",
                ax = ax_plot
               )
    # plt.show()
    # return graph_dat_heatmap

def plot_graph_results(
    graph_dat_plot,
    metric_column,
    node_origin,
    node_destiny,
    trial_group,
    minimal_cut_value,
    random_level,
    ax_plot,
    soft_cut_value=.7
):

    ### graph
    edges_info=[node_origin,node_destiny,metric_column]
    full_pairs=np.array(graph_dat_plot[edges_info])
    #### results graph
    dat_plot_pass=graph_dat_plot[graph_dat_plot[metric_column]>minimal_cut_value]
    dat_plot_train_fail=graph_dat_plot[graph_dat_plot[metric_column]<=minimal_cut_value]
    dat_plot_soft=graph_dat_plot[(graph_dat_plot[metric_column]<=minimal_cut_value)&(
        graph_dat_plot[metric_column]>soft_cut_value)]
    dat_plot_random=graph_dat_plot[(graph_dat_plot[metric_column]<=soft_cut_value)&(
        graph_dat_plot[metric_column]>random_level)]
    dat_plot_fail=graph_dat_plot[graph_dat_plot[metric_column]<=random_level]

    train_pairs_pass = np.array(dat_plot_pass.loc[dat_plot_pass[trial_group]=="train", edges_info])
    reflexivity_pairs_pass = np.array(dat_plot_pass.loc[dat_plot_pass[trial_group]=="reflexivity", edges_info])
    transitivity_pairs_pass = np.array(dat_plot_pass.loc[dat_plot_pass[trial_group]=="transitivity", edges_info])
    symmetry_pairs_pass = np.array(dat_plot_pass.loc[dat_plot_pass[trial_group]=="symmetry", edges_info])

    reflexivity_pairs_soft = np.array(dat_plot_soft.loc[dat_plot_soft[trial_group]=="reflexivity", edges_info])
    transitivity_pairs_soft = np.array(dat_plot_soft.loc[dat_plot_soft[trial_group]=="transitivity", edges_info])
    symmetry_pairs_soft = np.array(dat_plot_soft.loc[dat_plot_soft[trial_group]=="symmetry", edges_info])

    reflexivity_pairs_random = np.array(dat_plot_random.loc[dat_plot_random[trial_group]=="reflexivity", edges_info])
    transitivity_pairs_random = np.array(dat_plot_random.loc[dat_plot_random[trial_group]=="transitivity", edges_info])
    symmetry_pairs_random = np.array(dat_plot_random.loc[dat_plot_random[trial_group]=="symmetry", edges_info])

    train_pairs_fail = np.array(dat_plot_train_fail.loc[dat_plot_train_fail[trial_group]=="train", edges_info])
    reflexivity_pairs_fail = np.array(dat_plot_fail.loc[dat_plot_fail[trial_group]=="reflexivity", edges_info])
    transitivity_pairs_fail = np.array(dat_plot_fail.loc[dat_plot_fail[trial_group]=="transitivity", edges_info])
    symmetry_pairs_fail = np.array(dat_plot_fail.loc[dat_plot_fail[trial_group]=="symmetry", edges_info])

    ### plot graph
    options = {
        "font_size": 24,
        "node_size": 1500,
        "node_color": "none",#"white",
        "edgecolors": "black",
        "linewidths": 4,
        "width": 5,
        "connectionstyle":"arc3, rad = 0.05",
    }

    ###  plot graphs
    DG = nx.DiGraph()
    DG.add_edges_from([[pair[0],pair[1]] for pair in full_pairs])
    pos = nx.circular_layout(DG)

    # ###  plot results graph

    plt.figure(figsize=(10,10)) 
    nx.draw_networkx_nodes(DG, 
                           pos,  
                           node_size= options["node_size"],
                           node_color= options["node_color"],
                           edgecolors= options["edgecolors"],
                           linewidths= options["linewidths"],
                           ax = ax_plot
                          )

    nx.draw_networkx_labels(DG, 
                            pos, 
                            font_size=options["font_size"], 
                            font_family="sans-serif",
                            ax = ax_plot)

    #train
    nx.draw_networkx_edges(DG, 
                           pos, 
                           edgelist=train_pairs_fail, 
                           node_size= options["node_size"], 
                           width= options["width"],  
                           connectionstyle=options["connectionstyle"], 
                           edge_color=colors_dict["color_fail"],
                           ax = ax_plot)
    nx.draw_networkx_edges(DG, 
                           pos, 
                           edgelist=train_pairs_pass, 
                           node_size= options["node_size"], 
                           width= options["width"],  
                           connectionstyle=options["connectionstyle"], 
                           edge_color=colors_dict["color_pass"],
                           ax = ax_plot)


    # reflexivity
    nx.draw_networkx_edges(DG, 
                           pos, 
                           edgelist=reflexivity_pairs_fail, 
                           # node_size= options["node_size"], 
                           arrowstyle="<|-", 
                           width= options["linewidths"]-1, 
                           style="dashed", 
                           # connectionstyle=options["connectionstyle"], 
                           edge_color= colors_dict["color_fail"],
                           ax = ax_plot)
    nx.draw_networkx_edges(DG, 
                           pos, 
                           edgelist=reflexivity_pairs_random, 
                           # node_size= options["node_size"], 
                           arrowstyle="<|-", 
                           width= options["linewidths"]-1, 
                           style="dashed", 
                           # connectionstyle=options["connectionstyle"], 
                           edge_color= colors_dict["color_random_fail"],
                           ax = ax_plot)
    nx.draw_networkx_edges(DG, 
                           pos, 
                           edgelist=reflexivity_pairs_soft, 
                           # node_size= options["node_size"], 
                           arrowstyle="<|-", 
                           width= options["linewidths"]-1, 
                           style="dashed", 
                           # connectionstyle=options["connectionstyle"], 
                           edge_color= colors_dict["color_soft_pass"],
                           ax = ax_plot)
    nx.draw_networkx_edges(DG, 
                           pos, 
                           edgelist=reflexivity_pairs_pass, 
                           # node_size= options["node_size"], 
                           arrowstyle="<|-", 
                           width= options["linewidths"]-1, 
                           style="dashed", 
                           # connectionstyle=options["connectionstyle"], 
                           edge_color=colors_dict["color_pass"],
                           ax = ax_plot)
    
    # symmetry
    nx.draw_networkx_edges(DG, 
                           pos, 
                           edgelist=symmetry_pairs_fail, 
                           node_size= 5000, 
                           width= options["linewidths"], 
                           style="dashed", 
                           connectionstyle=options["connectionstyle"],
                           edge_color=colors_dict["color_fail"],
                           ax = ax_plot)
    nx.draw_networkx_edges(DG, 
                           pos, 
                           edgelist=symmetry_pairs_random, 
                           node_size= 5000, 
                           width= options["linewidths"], 
                           style="dashed", 
                           connectionstyle=options["connectionstyle"],
                           edge_color=colors_dict["color_random_fail"],
                           ax = ax_plot)
    nx.draw_networkx_edges(DG, 
                           pos, 
                           edgelist=symmetry_pairs_soft, 
                           node_size= 5000, 
                           width= options["linewidths"], 
                           style="dashed", 
                           connectionstyle=options["connectionstyle"],
                           edge_color=colors_dict["color_soft_pass"],
                           ax = ax_plot
                          )
    nx.draw_networkx_edges(DG, 
                           pos, 
                           edgelist=symmetry_pairs_pass, 
                           node_size= 5000, 
                           width= options["linewidths"], 
                           style="dashed", 
                           connectionstyle=options["connectionstyle"],
                           edge_color=colors_dict["color_pass"],
                           ax = ax_plot
                          )
    
    # transitivity
    nx.draw_networkx_edges(DG, 
                           pos, 
                           edgelist=transitivity_pairs_fail, 
                           node_size= 5000, 
                           width= options["linewidths"]-1, 
                           style="dotted", 
                           connectionstyle=options["connectionstyle"],
                           edge_color=colors_dict["color_fail"],
                           ax = ax_plot)
    nx.draw_networkx_edges(DG, 
                           pos, 
                           edgelist=transitivity_pairs_random, 
                           node_size= 5000, 
                           width= options["linewidths"]-1, 
                           style="dotted", 
                           connectionstyle=options["connectionstyle"],
                           edge_color=colors_dict["color_random_fail"],
                           ax = ax_plot)
    nx.draw_networkx_edges(DG, 
                           pos, 
                           edgelist=transitivity_pairs_soft, 
                           node_size= 5000, 
                           width= options["linewidths"]-1, 
                           style="dotted", 
                           connectionstyle=options["connectionstyle"],
                           edge_color=colors_dict["color_soft_pass"],
                           ax = ax_plot)
    nx.draw_networkx_edges(DG, 
                           pos, 
                           edgelist=transitivity_pairs_pass, 
                           node_size= 5000, 
                           width= options["linewidths"]-1, 
                           style="dotted", 
                           connectionstyle=options["connectionstyle"],
                           edge_color=colors_dict["color_pass"],
                           ax = ax_plot)