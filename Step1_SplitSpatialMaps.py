import numpy as np
import os
import shutil
import pandas as pd
import datetime


# Hyperparameters
CellPatchNum = 50000  # Cell count threshold for >2nd splitting.
MinCellCount_Patch = 20  # Cell count threshold for keeping a patch.
InputFolderName = "./Step0_Output/"  # Change it to the original input folder name for multi-condition datasets.


# Output folder
ThisStep_OutputFolderName = "./Step1_Output/"
if os.path.exists(ThisStep_OutputFolderName):
    shutil.rmtree(ThisStep_OutputFolderName)
os.makedirs(ThisStep_OutputFolderName)

# Import image name list.
Region_filename = InputFolderName + "ImageNameList.txt"
region_name_list = pd.read_csv(
    Region_filename,
    sep="\t",  # tab-separated
    header=None,  # no heading row
    names=["Image"],  # set our own names for the columns
)

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print("Splitting each image/sample into small patches...")

global_boundary_file = f"{ThisStep_OutputFolderName}All_Boundary.txt"

# Clear this file every time it runs
if os.path.exists(global_boundary_file):
    os.remove(global_boundary_file)


# During recursive splitting, the boundaries of the current segmentation region are recorded at each step
def split_region(region_name, coordinates, cell_types, graph_labels, x_range, y_range, prefix="Patch"):
    x_min, x_max = x_range
    y_min, y_max = y_range

    # Compute the cutting lines (cross-cutting)
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2

    # Use a set to store the written _CUT lines and avoid duplication.
    cut_lines_set = set()

    with open(global_boundary_file, 'a') as gbf:
        cut_x_line = (region_name, x_mid, x_mid, y_min, y_max)
        cut_y_line = (region_name, x_min, x_max, y_mid, y_mid)

        if cut_x_line not in cut_lines_set:
            gbf.write(f"{region_name}\t{x_mid}\t{x_mid}\t{y_min}\t{y_max}\n")
            cut_lines_set.add(cut_x_line)

        if cut_y_line not in cut_lines_set:
            gbf.write(f"{region_name}\t{x_min}\t{x_max}\t{y_mid}\t{y_mid}\n")
            cut_lines_set.add(cut_y_line)


    # Partition into quadrants (four equal sub-regions)
    sub_ranges = [
        ([x_min, x_mid], [y_min, y_mid]),  # Top-left quadrant
        ([x_mid, x_max], [y_min, y_mid]),  # Top-right quadrant
        ([x_min, x_mid], [y_mid, y_max]),  # ​Bottom-left quadrant
        ([x_mid, x_max], [y_mid, y_max])   # Bottom-Right quadrant
    ]

    for i, (sub_x_range, sub_y_range) in enumerate(sub_ranges):
        sub_coordinates = []
        sub_cell_types = []
        for idx, (x, y) in enumerate(coordinates):
            if sub_x_range[0] <= x < sub_x_range[1] and sub_y_range[0] <= y < sub_y_range[1]:
                sub_coordinates.append((x, y))
                sub_cell_types.append(cell_types[idx])

        if len(sub_coordinates) == 0:
            continue  # Ignore Empty Regions

        # ​Continue Splitting Recursively
        if len(sub_coordinates) > CellPatchNum:
            new_prefix = f"{prefix}_{i}"
            split_region(region_name, sub_coordinates, sub_cell_types, graph_labels, sub_x_range, sub_y_range, new_prefix)
        else:
            # ​Record Final Patch Coordinates
            if len(sub_coordinates) >= MinCellCount_Patch:
                patch_name = f"{prefix}_{i}-{region_name}"
                coord_file = f"{ThisStep_OutputFolderName}{patch_name}_Coordinates.txt"
                type_file = f"{ThisStep_OutputFolderName}{patch_name}_CellTypeLabel.txt"
                label_file = f"{ThisStep_OutputFolderName}{patch_name}_GraphLabel.txt"

                with open(coord_file, 'w') as cf, open(type_file, 'w') as tf, open(label_file, 'w') as lf:
                    for coord in sub_coordinates:
                        cf.write(f"{coord[0]}\t{coord[1]}\n")
                    for cell_type in sub_cell_types:
                        tf.write(f"{cell_type}\n")
                    for label in graph_labels:
                        lf.write(f"{label}\n")

                # Add Patch Names
                with open(f"{ThisStep_OutputFolderName}ImagePatchNameList.txt", "a") as f0:
                    f0.write(f"{patch_name}\n")


# Main loop for processing each image
for graph_index in range(len(region_name_list)):
    print(f"This is image-{graph_index}")
    region_name = region_name_list.Image[graph_index]

    # Import target graph x/y coordinates
    GraphCoord_filename = InputFolderName + region_name + "_Coordinates.txt"
    coordinates = []
    with open(GraphCoord_filename, 'r') as file:
        for line in file:
            parts = line.split()
            coordinates.append((float(parts[0]), float(parts[1])))

    # Import cell type info in the target graph
    CellType_filename = InputFolderName + region_name + "_CellTypeLabel.txt"
    cell_types = []
    with open(CellType_filename, 'r') as file:
        for line in file:
            cell_types.append(line.strip())

    # Import target graph label
    GraphLabel_filename = InputFolderName + region_name + "_GraphLabel.txt"
    graph_labels = []
    with open(GraphLabel_filename, 'r') as file:
        for line in file:
            graph_labels.append(line.strip())

    # Get boundary of the entire region
    coordinates_array = np.array(coordinates)
    x_min, y_min = np.min(coordinates_array, axis=0)
    x_max, y_max = np.max(coordinates_array, axis=0)

    # Start recursive splitting
    split_region(region_name, coordinates, cell_types, graph_labels, [x_min, x_max], [y_min, y_max])


print("Step1 done!")
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


