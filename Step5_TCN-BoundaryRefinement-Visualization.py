import re
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import sci_palettes
import os
import shutil
import datetime
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42  # make text in plot editable in AI.
sci_palettes.register_cmap("d3_category20")  # register a specific palette for TCN coloring.


# Hyperparameters
KNN_K = 50
Num_TCN = 4
Smoothing_range = 50  # How large (um) for boundary refinement.
InputFolderName = "./Step0_Output/"  # Change it to the original input folder name for multi-condition datasets.


# Output folder
LastStep_OutputFolderName = "./Step4_Output/"
ThisStep_OutputFolderName = "./Step5_Output/"
PatchFolderName = "./Step1_Output/"
if os.path.exists(ThisStep_OutputFolderName):
    shutil.rmtree(ThisStep_OutputFolderName)
os.makedirs(ThisStep_OutputFolderName)

OutputFolderName_1 = ThisStep_OutputFolderName + "TCN_Plot/"
os.mkdir(OutputFolderName_1)
OutputFolderName_2 = ThisStep_OutputFolderName + "CellRefinement_Plot/"
os.mkdir(OutputFolderName_2)
OutputFolderName_3 = ThisStep_OutputFolderName + "ResultTable_File/"
os.mkdir(OutputFolderName_3)
OutputFolderName_4 = ThisStep_OutputFolderName + "CellType_Plot/"
os.mkdir(OutputFolderName_4)

# Import image name list.
Region_filename = InputFolderName + "ImageNameList.txt"
region_name_list = pd.read_csv(
    Region_filename,
    sep="\t",  # tab-separated
    header=None,  # no heading row
    names=["Image"],  # set our own names for the columns
)

unique_cell_type_df = pd.read_csv(
    "./Step2_Output/UniqueCellTypeList.txt",
    sep="\t",  # tab-separated
    header=None,  # no heading row
    names=["UniqueCellType"],  # set our own names for the columns
)
UniqueCellType_vec = unique_cell_type_df['UniqueCellType'].values.tolist()

## Initialize a TCN code list used for matching color palettes across different TCN plots.
UniqueTCN_vec = list(range(1, Num_TCN + 1))
UniqueTCN_vec = [str(element) for element in UniqueTCN_vec]

def load_boundary_ranges(region_name):
    cut_lines = []
    boundary_file =PatchFolderName + "All_Boundary.txt"  

    with open(boundary_file, 'r') as bf:
        for line in bf:
            parts = line.strip().split('\t')
            if parts[0] == region_name: 
                cut_lines.append(tuple(map(float, parts[1:])))

    print(f"Loaded {len(cut_lines)} cut lines for {region_name}")
    return cut_lines


for graph_index in range(len(region_name_list)):
    region_name = region_name_list.Image[graph_index]

    # Import patch name list for the current region
    PatchNameList_filename = PatchFolderName + "ImagePatchNameList.txt"
    patch_name_list = pd.read_csv(
        PatchNameList_filename,
        sep="\t",  # tab-separated
        header=None,  # no heading row
        names=["PatchName"],  # set our own names for the columns
    )

    target_graph_map_Merged = pd.DataFrame()
    patch_boundary_dict={}
    
    print(f"\nStart to match sample/image_name: {region_name}")
    matched_count = 0  
    for patch_name in patch_name_list.PatchName:
        # Precise matching: Patch names must end with "-" + region_name
        if patch_name.endswith(f"-{region_name}"):
            matched_count += 1
            print(f"Matched [{matched_count}]: {patch_name}")

            PatchCoord_filename = PatchFolderName + patch_name + "_Coordinates.txt"
            patch_coords = pd.read_csv(PatchCoord_filename, sep="\t", header=None, names=["x_coordinate", "y_coordinate"])
            patch_x_min, patch_x_max = patch_coords["x_coordinate"].min(), patch_coords["x_coordinate"].max()
            patch_y_min, patch_y_max = patch_coords["y_coordinate"].min(), patch_coords["y_coordinate"].max()
            patch_boundary_dict[patch_name] = (patch_x_min, patch_x_max, patch_y_min, patch_y_max)

            target_graph_map = patch_coords

            CellType_filename = PatchFolderName + patch_name + "_CellTypeLabel.txt"
            cell_type_label = pd.read_csv(CellType_filename, sep="\t", header=None, names=["cell_type"])
            target_graph_map["Cell_Type"] = cell_type_label.cell_type

            MajorityVoting_FileName = LastStep_OutputFolderName + "ImageCollection/" + patch_name + "/TCNLabel_MajorityVoting.csv"
            tcn_labels = np.loadtxt(MajorityVoting_FileName, dtype='int', delimiter=",")
            target_graph_map["TCN_Label"] = tcn_labels

            target_graph_map_Merged = pd.concat([target_graph_map_Merged, target_graph_map], ignore_index=True)

    # Final summary
    print(f"\nTotally matched {matched_count} patches (region_name={region_name})")
    # Continue dealing with target_graph_map_Merged
    target_graph_map_Merged.TCN_Label = target_graph_map_Merged.TCN_Label.astype(str)
    ## Converting integer list to string list for making color scheme discrete.
    target_graph_map_Merged.TCN_Label = target_graph_map_Merged.TCN_Label.astype(str)
    ## Below is for matching color palettes across different TCN plots, which is quite useful for supervised tasks.
    target_graph_map_Merged["TCN_Label"] = pd.Categorical(target_graph_map_Merged["TCN_Label"], UniqueTCN_vec)
    ## Below is for matching color palettes across different cell type plots, which is quite useful for supervised tasks.
    target_graph_map_Merged["Cell_Type"] = pd.Categorical(target_graph_map_Merged["Cell_Type"], UniqueCellType_vec)


    # -----------------------------------------Get smooth_cells in boundaries------------------------------------------------- #
    # Create "target_graph_map_Merged_Smooth" to store results after smoothing (refinement)
    target_graph_map_Merged_Smooth = target_graph_map_Merged.copy()
    target_graph_map_Merged_Smooth["Smooth_Label"] = "0"  # Initialize smooth_cells as "0" for visualization to ensure correctness.

    x_coords = target_graph_map_Merged['x_coordinate'].values
    y_coords = target_graph_map_Merged['y_coordinate'].values
    coordinates_array = np.column_stack((x_coords, y_coords))  # Coordinate Order Mismatch Alert​: **target_graph_map_Merged vs input**​

    # Read splitting boundaries.
    boundary_ranges = load_boundary_ranges(region_name)

    # Determine the smooth cells
    smooth_cells = []
    for index, (x, y) in tqdm(enumerate(coordinates_array), total=len(coordinates_array), desc="Smoothing progression"):
        # ​Ensure Cells Are Within Patch Boundaries
        for patch_name, (patch_x_min, patch_x_max, patch_y_min, patch_y_max) in patch_boundary_dict.items():
            if patch_x_min <= x <= patch_x_max and patch_y_min <= y <= patch_y_max:
                for boundary_x_min, boundary_x_max, boundary_y_min, boundary_y_max in boundary_ranges:
                    if boundary_x_min == boundary_x_max:  # Vertical _CUT Lines
                        if abs(x - boundary_x_min) <= smoothing_range and (boundary_y_min - smoothing_range) <= y <= (boundary_y_max + smoothing_range):
                            target_graph_map_Merged_Smooth.loc[index, "Smooth_Label"] = "1"
                    
                    elif boundary_y_min == boundary_y_max:  # Horizontal _CUT Lines
                        if abs(y - boundary_y_min) <= smoothing_range and (boundary_x_min - smoothing_range) <= x <= (boundary_x_max + smoothing_range):
                            target_graph_map_Merged_Smooth.loc[index, "Smooth_Label"] = "1"

    smooth_cells_indices = target_graph_map_Merged_Smooth[target_graph_map_Merged_Smooth["Smooth_Label"] == "1"].index

    # Only perform KNN when smooth cells are detected
    if len(smooth_cells_indices) > 0:
        K = KNN_K
        nbrs = NearestNeighbors(n_neighbors=K + 1)  # Add 1 to Include Itself 
        nbrs.fit(coordinates_array)

        for index in smooth_cells_indices:
            distances, indices = nbrs.kneighbors(coordinates_array[index].reshape(1, -1))
            KNN_neighbors = indices[0][1:]  # Select Top K Neighbors (Excluding Itself)

            # Get TCN_Label for K Nearest Neighbors
            neighbor_labels = target_graph_map_Merged.loc[KNN_neighbors, 'TCN_Label'].values
            new_labels = Counter(neighbor_labels).most_common(1)  # Select the Most Frequent TCN_Label Among Neighbors

            # Update TCN_Label
            target_graph_map_Merged_Smooth.loc[index, "TCN_Label"] = new_labels[0][0]


    # -----------------------------------------Visualization: Highlighting Smooth Cells at Boundaries---------------------------------------------- #
    smooth_plot_dict = {"0": "#beaed4", "1": "#7fc97f"}  # purple for non-boundary cells and green for boundary cells
    smooth_plot = sns.scatterplot(x="x_coordinate", y="y_coordinate", data=target_graph_map_Merged_Smooth, hue="Smooth_Label",
                                 palette=smooth_plot_dict, alpha=1.0, s=0.5, legend="full")
    smooth_plot.spines.right.set_visible(False)
    smooth_plot.spines.left.set_visible(False)
    smooth_plot.spines.top.set_visible(False)
    smooth_plot.spines.bottom.set_visible(False)
    smooth_plot.set(xticklabels=[])  # remove the tick label.
    smooth_plot.set(yticklabels=[])
    smooth_plot.set(xlabel=None)  # remove the axis label.
    smooth_plot.set(ylabel=None)
    smooth_plot.tick_params(bottom=False, left=False)  # remove the ticks.

    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0)
    # Save the CURRENT figure.
    smooth_fig_filename1 = OutputFolderName_2 + "SmoothPlot_" + region_name + ".png"
    plt.savefig(smooth_fig_filename1)
    plt.close()


    # -----------------------------------------Generate TCN plots------------------------------------------------- #
    ## Plot x/y map with "TCN_Label" coloring.
    TCN_plot = sns.scatterplot(x="x_coordinate", y="y_coordinate", data=target_graph_map_Merged_Smooth, hue="TCN_Label",
                               palette="d3_category20", alpha=1.0, s=0.5, legend="full")  # "d3_category20" for 20 colors.
    # Hide all four spines
    TCN_plot.spines.right.set_visible(False)
    TCN_plot.spines.left.set_visible(False)
    TCN_plot.spines.top.set_visible(False)
    TCN_plot.spines.bottom.set_visible(False)
    TCN_plot.set(xticklabels=[])  # remove the tick label.
    TCN_plot.set(yticklabels=[])
    TCN_plot.set(xlabel=None)  # remove the axis label.
    TCN_plot.set(ylabel=None)
    TCN_plot.tick_params(bottom=False, left=False)  # remove the ticks.
    # Place legend outside top right corner of the CURRENT plot

    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0)
    # Save the CURRENT figure.
    TCN_fig_filename1 = OutputFolderName_1 + "TCN_" + region_name + ".pdf"
    plt.savefig(TCN_fig_filename1)
    TCN_fig_filename2 = OutputFolderName_1 + "TCN_" + region_name + ".png"
    plt.savefig(TCN_fig_filename2)
    plt.close()


    #-----------------------------------------Generate CT plots-------------------------------------------------#
    ## Plot x/y map with "Cell_Type" coloring.
    CellType_plot = sns.scatterplot(x="x_coordinate", y="y_coordinate", data=target_graph_map_Merged_Smooth, hue="Cell_Type", 
                                    palette=sns.color_palette("husl", 30), alpha=1.0, s=0.5, legend="full")  # 30 colors at maximum.

    # Hide all four spines
    CellType_plot.spines.right.set_visible(False)
    CellType_plot.spines.left.set_visible(False)
    CellType_plot.spines.top.set_visible(False)
    CellType_plot.spines.bottom.set_visible(False)
    CellType_plot.set(xticklabels=[])  # remove the tick label.
    CellType_plot.set(yticklabels=[])
    CellType_plot.set(xlabel=None)  # remove the axis label.
    CellType_plot.set(ylabel=None)
    CellType_plot.tick_params(bottom=False, left=False)  # remove the ticks.
    # Place legend outside top right corner of the CURRENT plot

    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0)
    # Save the CURRENT figure.
    CellType_fig_filename1 = OutputFolderName_4 + "CellType_" + region_name + ".pdf"
    plt.savefig(CellType_fig_filename1)
    CellType_fig_filename2 = OutputFolderName_4 + "CellType_" + region_name + ".png"
    plt.savefig(CellType_fig_filename2)
    plt.close()


    #---------------------------------Export result dataframe: "target_graph_map_Merged_Smooth"--------------------------#
    TargetGraph_dataframe_filename = OutputFolderName_3 + "ResultTable_" + region_name + ".csv"
    target_graph_map_Merged_Smooth = target_graph_map_Merged_Smooth.drop('Smooth_Label', axis=1)
    target_graph_map_Merged_Smooth.to_csv(TargetGraph_dataframe_filename, na_rep="NULL", index=False)  # remove row index.


print("Step5 done!")
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


