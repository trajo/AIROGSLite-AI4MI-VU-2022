# Class for reading the annotation csv files
# Determines middle of optic disk based on two annotation points, then takes avg all annotators
# Also average of fovea

import json
import pandas as pd
import functools as ft

# Index needed to get first 1000 images
INDEX_FIRST_1000_IMAGES = 3000


def get_df_of_annotations_file(csv_annotations_file):

    """
    Use csv file with annotations one annotator to determine the optic disk
    Adds original filename to df based on shuffled filename
    :param csv_annotations_file: csv file with annotations one annotator
    :return: dataframe with mean optic disk, and other information
    """

    # Get unedited df from csv
    loaded_csv_df = get_unedited_df_from_csv(csv_annotations_file)

    # Check if each image has three annotations, otherwise give error
    error_checking_nr_of_annotations(loaded_csv_df)

    # Lines below needed to extract cx1, cx2, cx3, cy1, cy2 and cy3
    df_cx1_cy1 = get_df_for_one_annotation(0, loaded_csv_df)
    df_cx2_cy2 = get_df_for_one_annotation(1, loaded_csv_df)
    df_cx3_cy3 = get_df_for_one_annotation(2, loaded_csv_df)

    # Create dataframe with the cx and cy value for each of the three annotations, per image
    df_merged_three_annotations = pd.merge(pd.merge(df_cx1_cy1, df_cx2_cy2, on='filename'), df_cx3_cy3, on='filename')

    # Drop redundant columns from merged df_merged_three_annotations
    df_merged_three_annotations = drop_redundant_columns_merged_df(df_merged_three_annotations)

    # Get mean of annotation of cx1 and cx2, and of cy1 and cy2
    df_merged_three_annotations = get_mean_cx1_cx2_and_cy1_cy2(df_merged_three_annotations)

    # Read json file with mapping original and shuffled and add it to df_merged_three_annotations
    df_merged_three_annotations = get_df_with_mapping_column_filenames(df_merged_three_annotations)

    # Return df with all information one annotator
    return df_merged_three_annotations


def get_unedited_df_from_csv(filename):

    """
    Removes unnecessary columns from df, made from csv
    Extracts from column 'region_shape_attributes' cx1, cx2, cx3, cy1, cy2 and cy3
    :param filename: filename
    :return: df without unnecessary columns, contains cx1, cx2, cx3, cy1, cy2 and cy3
    """
    # Load CSV file with the annotations
    loaded_csv_df = pd.read_csv(filename)

    # Get information first 1000 pictures
    loaded_csv_df = loaded_csv_df.head(INDEX_FIRST_1000_IMAGES)

    # Drop useless columns
    loaded_csv_df = loaded_csv_df.drop('file_attributes', axis=1)
    loaded_csv_df = loaded_csv_df.drop('region_attributes', axis=1)

    # Transform column "region_shape_attributes" into dictionary containing cx and cy
    loaded_csv_df['region_shape_attributes'] = [json.loads(x) for x in loaded_csv_df['region_shape_attributes']]

    # Make from dictionary keys of column "region_shape_attributes" separate columns with cx1, cx2, cx3, cy1, cy2 and cy3
    loaded_csv_df = pd.concat([loaded_csv_df, loaded_csv_df["region_shape_attributes"].apply(pd.Series)], axis=1)

    # Drop useless columns created from the dictionary of column "region_shape_attributes"
    loaded_csv_df = loaded_csv_df.drop('region_shape_attributes', axis=1)
    loaded_csv_df = loaded_csv_df.drop('name', axis=1)
    loaded_csv_df = loaded_csv_df.drop('region_count', axis=1)

    return loaded_csv_df


def error_checking_nr_of_annotations(df_to_error_check):

    """
    Check if each images has three annotations
    :param df_to_error_check: df to check on three annotations per image
    :return: error println in case of missing annotations
    """
    # Check if each image has three annotations
    occurences_filename_series = df_to_error_check['filename'].value_counts()
    df_occurences_filename = pd.DataFrame(
        {'filename': occurences_filename_series.index, 'counts': occurences_filename_series.values})
    if not df_occurences_filename.loc[df_occurences_filename['counts'] != 3].empty:
        print("You have missed annotations for the following files:")
        print(df_occurences_filename.loc[df_occurences_filename['counts'] != 3])


# Get df with cx and cy for specific annotation, thus annotation point 1, 2 or 3
# Annotation_number is either 0, 1 or 2
def get_df_for_one_annotation(annotation_number, used_df):

    """
    Get df with cx and cy for specific annotation, thus annotation point 1, 2 or 3
    :param annotation_number: 0, 1 or 2
    :param used_df: df to extract from: cx1-cy1, or cx2-cy2, or cx3-cy3
    :return: df containing cx1-cy1, or cx2-cy2, or cx3-cy3
    """
    # Get df with cx and cy for annotation number
    df_cx_cy = used_df.drop(used_df[used_df.region_id != annotation_number].index)
    # Rename columns cx and cy from for annotation to corresponding annotation number.
    column_number = annotation_number + 1
    df_cx_cy = df_cx_cy.rename(columns={'cx': 'cx' + str(column_number), 'cy': 'cy' + str(column_number)})

    return df_cx_cy


def drop_redundant_columns_merged_df(df_redundant_columns):

    """
    Removes redundant columns after merging three df's containing each one annotation
    :param df_redundant_columns: df containing redundant columns
    :return: df without redundant columns
    """

    # Drop redundant columns
    df_redundant_columns = df_redundant_columns.drop('file_size_y', axis=1)
    df_redundant_columns = df_redundant_columns.drop('file_size_x', axis=1)
    df_redundant_columns = df_redundant_columns.drop('region_id_x', axis=1)
    df_redundant_columns = df_redundant_columns.drop('region_id_y', axis=1)
    df_redundant_columns = df_redundant_columns.drop('region_id', axis=1)

    return df_redundant_columns


def get_mean_cx1_cx2_and_cy1_cy2(df_to_get_mean_of):

    """
    Get mean of cx1 and cx2, and get mean of cy1 and cy2. Add these two means as columns
    :param df_to_get_mean_of: df containing cx1, cx2, cy1, cy2
    :return: df with mean for cx1-cx2, and cy1-cy2
    """

    # Add column with mean of annotation cx1 and cx2
    df_to_get_mean_of['mean_cx_1_2'] = df_to_get_mean_of[['cx1', 'cx2']].mean(axis=1)
    df_to_get_mean_of['mean_cx_1_2'] = df_to_get_mean_of['mean_cx_1_2'].astype('int')

    # Add column with mean of annotation yx1 and yx2
    df_to_get_mean_of['mean_cy_1_2'] = df_to_get_mean_of[['cy1', 'cy2']].mean(axis=1)
    df_to_get_mean_of['mean_cy_1_2'] = df_to_get_mean_of['mean_cy_1_2'].astype('int')

    return df_to_get_mean_of


def get_df_with_mapping_column_filenames(df_to_map):

    """
    Read json file with mapping original and shuffled and add it to df which needs the mapping
    :param df_to_map: df which needs original filename attached to shuffled filename
    :return: return df also containing original filename
    """

    # Open JSON file and read into df
    f = open('original_shuffled_map.json')
    dict_mapping = json.load(f)
    df_mapping = pd.DataFrame(dict_mapping.items(), columns=['original_filename', 'shuffled_filename'])

    # Add .jpg in column shuffled_filename for joining with other df
    df_mapping['shuffled_filename'] = df_mapping['shuffled_filename'].astype(str) + ".jpg"

    # Change column name filename in df with annotations to shuffled_filename
    df_to_map = df_to_map.rename(columns={'filename': 'shuffled_filename'})

    # Merge the two df's based on the shuffled filename
    df_to_map = df_to_map.merge(df_mapping, on='shuffled_filename')

    return df_to_map


def get_json_file_meanpoint_one_annotator(df_for_json, name_annotator):

    """
    Returns json file with mean point optic disk for df of one annotator
    :param df_for_json: df to use
    :param name_annotator: name of annotator
    :return: saves json with mean point optic disk
    """

    # Get mean for cx and cy point
    cx_list = df_for_json['mean_cx_1_2'].tolist()
    cy_list = df_for_json['mean_cy_1_2'].tolist()

    # Load json file with images, but without annotations
    with open('annotations/empty_json_file.json') as json_file:
        data = json.load(json_file)

    # Add cx and cy to json file
    keys = data.keys()
    for i, key in enumerate(keys):
        cx = cx_list[i]
        cy = cy_list[i]
        data[key]['regions'] =[{"shape_attributes":{"name":"point","cx":cx,"cy":cy},"region_attributes":{}}]


    # Save the json file
    with open('annotations/avg_optic_disk_annotator_'+ name_annotator + '.json', 'w') as fp:
        json.dump(data, fp)


# Returns df with the annotations of all annotators and the means
def get_df_four_annotators(df_rutger, df_valentin, df_robert, df_fauve):

    """
    Returns df combining annotations of the four annotators
    :param df_rutger: df with annotations rutger and mean optic disk
    :param df_valentin: df with annotations valentin and mean optic disk
    :param df_robert: df with annotations robert and mean optic disk
    :param df_fauve: df with annotations fauve and mean optic disk
    :return: df with data from the four annotators
    """

    # Rename annotation points rutger
    df_rutger = rename_columns_author(df_rutger, 'rut')
    # Rename annotation points valentin
    df_valentin = rename_columns_author(df_valentin, 'valen')
    # Rename annotation points robert
    df_robert = rename_columns_author(df_robert, 'rob')
    # Rename annotation points fauve
    df_fauve = rename_columns_author(df_fauve, 'fauve')

    # Merge the four dataframes
    dfs_to_merge = [df_rutger, df_valentin, df_robert, df_fauve]
    merged_df = ft.reduce(lambda left, right: pd.merge(left, right, on=['shuffled_filename','original_filename',
                                                                        'file_size']), dfs_to_merge)

    # Calculate mean between cx1 and cx2 all annotators
    merged_df['mean_cx_1_2_annotators'] = merged_df[['mean_cx_1_2_rut', 'mean_cx_1_2_valen',
                                                     'mean_cx_1_2_rob', 'mean_cx_1_2_fauve']].mean(axis=1)
    merged_df['mean_cx_1_2_annotators'] = merged_df['mean_cx_1_2_annotators'].astype('int')

    # Calculate mean between cy1 and cy2 all annotators
    merged_df['mean_cy_1_2_annotators'] = merged_df[['mean_cy_1_2_rut', 'mean_cy_1_2_valen',
                                                     'mean_cy_1_2_rob', 'mean_cy_1_2_fauve']].mean(axis=1)
    merged_df['mean_cy_1_2_annotators'] = merged_df['mean_cy_1_2_annotators'].astype('int')

    # Calculate mean between cx3 all annotators
    merged_df['mean_cx3_annotators'] = merged_df[['cx3_rut', 'cx3_valen',
                                                     'cx3_rob', 'cx3_fauve']].mean(axis=1)
    merged_df['mean_cx3_annotators'] = merged_df['mean_cx3_annotators'].astype('int')

    # Calculate mean between cy3 all annotators
    merged_df['mean_cy3_annotators'] = merged_df[['cy3_rut', 'cy3_valen',
                                                     'cy3_rob', 'cy3_fauve']].mean(axis=1)
    merged_df['mean_cy3_annotators'] = merged_df['mean_cy3_annotators'].astype('int')

    return merged_df


def rename_columns_author(df, name):

    """
    Rename columns cx1, cx2, cx3, cy1, cy2, cy3, mean_cx_1_2, mean_cy_1_2 to column containing name annotator
    :param df: df annotator
    :param name: name annotator
    :return: df with columns renamed after annotator
    """

    df = df.rename(columns={'cx1': 'cx1_' + name, 'cx2': 'cx2_' + name, 'cx3': 'cx3_' + name,
                                          'cy1': 'cy1_' + name, 'cy2': 'cy2_' + name, 'cy3': 'cy3_' + name,
                                          'mean_cx_1_2': 'mean_cx_1_2_' + name, 'mean_cy_1_2': "mean_cy_1_2_" + name})
    return df


def get_json_file_meanpoint_all_annotator(df_for_json):

    """
    Saves json file with avg optic disk and fovea all annotators
    :param df_for_json:
    :return
    """
    # Get mean for cx and cy point
    cx_list_optic_disk = df_for_json['mean_cx_1_2_annotators'].tolist()
    cy_list_optic_disk = df_for_json['mean_cy_1_2_annotators'].tolist()

    cx_list_fovea = df_for_json['mean_cx3_annotators'].tolist()
    cy_list_fovea = df_for_json['mean_cy3_annotators'].tolist()

    # Load json file with images, but without annotations
    with open('annotations/empty_json_file.json') as json_file:
        data = json.load(json_file)

    # Add cx and cy to json file
    keys = data.keys()
    for i, key in enumerate(keys):

        # Get optic disk coordinate for picture
        cx1 = cx_list_optic_disk[i]
        cy1 = cy_list_optic_disk[i]

        # Get fovea coordinate for picture
        cx2 = cx_list_fovea[i]
        cy2 = cy_list_fovea[i]

        # Save the two points
        data[key]['regions'] = [{"shape_attributes": {"name": "point", "cx": cx1, "cy": cy1}, "region_attributes": {}},
                                {"shape_attributes": {"name": "point", "cx": cx2, "cy": cy2}, "region_attributes": {}}]

    # Save the json file
    with open('annotations/avg_optic_disk_fovea_all_annotators.json', 'w') as fp:
        json.dump(data, fp)


# Df with annotations Rutger
df_rutger = get_df_of_annotations_file("annotations/Annotation_Rutger_1000_images_corrected.csv")

# Df with annotations Valentin
df_valentin = get_df_of_annotations_file("annotations/valentin_corrected.csv")

# Df with annotations fauve
df_fauve = get_df_of_annotations_file("annotations/Annotations_fwevers_corrected.csv")

# Df with annotations robert
df_robert = get_df_of_annotations_file("annotations/robert.csv")

# Save json file with avg optic disk for annotator Rutger
get_json_file_meanpoint_one_annotator(df_rutger, "Rutger")

# Save json file with avg optic disk for annotator Valentin
get_json_file_meanpoint_one_annotator(df_valentin, "Valentin")

# Save json file with avg optic disk for annotator Fauve
get_json_file_meanpoint_one_annotator(df_fauve, "Fauve")

# Save json file with avg optic disk for annotator Robert
get_json_file_meanpoint_one_annotator(df_robert, "Robert")

# Get merged df with mean optic disk and fovea all annotators
df_all_annotators = get_df_four_annotators(df_rutger, df_valentin, df_fauve, df_robert)

# Save json file with avg optic disk and fovea for all annotators
get_json_file_meanpoint_all_annotator(df_all_annotators)

# Save csv file of df with all information
df_all_annotators.to_csv('csv_all_annotators.csv')












"""
trying_to_write_csv = get_df_of_annotations_file("Annotation_Rutger_1000_images.csv")
trying_to_write_csv = trying_to_write_csv.drop('cx1', axis=1)
trying_to_write_csv = trying_to_write_csv.drop('cy1', axis=1)
trying_to_write_csv = trying_to_write_csv.drop('cx2', axis=1)
trying_to_write_csv = trying_to_write_csv.drop('cy2', axis=1)
trying_to_write_csv = trying_to_write_csv.drop('cx3', axis=1)
trying_to_write_csv = trying_to_write_csv.drop('cy3', axis=1)
trying_to_write_csv = trying_to_write_csv.drop('original_filename', axis=1)


trying_to_write_csv['file_attributes'] = "{}"
trying_to_write_csv['region_id'] = 0
trying_to_write_csv['region_attributes'] = "{}"

#"{""name"":""point"",""cx"":1826,""cy"":1232}"
trying_to_write_csv['mean_cx_1_2'] = trying_to_write_csv['mean_cx_1_2'].astype(str)
trying_to_write_csv['mean_cy_1_2'] = trying_to_write_csv['mean_cy_1_2'].astype(str)

trying_to_write_csv['region_shape_attributes'] = "{\"name\":\"point\",\"cx\":" + trying_to_write_csv['mean_cx_1_2']\
                                                 + "," +  "\"cy\":" + trying_to_write_csv['mean_cy_1_2'] + "}"

trying_to_write_csv = trying_to_write_csv.rename(columns={'shuffled_filename': 'filename'})

trying_to_write_csv = trying_to_write_csv.drop('mean_cx_1_2', axis=1)
trying_to_write_csv = trying_to_write_csv.drop('mean_cy_1_2', axis=1)
trying_to_write_csv['region_count'] = 1

trying_to_write_csv = trying_to_write_csv[['filename', 'file_size', 'file_attributes', 'region_count', 'region_id',
                                           'region_shape_attributes', 'region_attributes']]

import csv
csv.QUOTE_NONNUMERIC

# Headers
#           filename  file_size file_attributes  region_count  region_id               region_shape_attributes region_attributes
trying_to_write_csv.to_csv('trying_csv.csv', quoting=csv.QUOTE_NONNUMERIC, index=False)

#print(trying_to_write_csv.to_string())
"""







