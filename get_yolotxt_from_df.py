# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
#from PIL import Image
#from PIL import ImageOps
def print_hi(name):

    # Import csv with annotation data
    df = pd.read_csv('csv_all_annotators.csv')
    # Import csv with scaling data
    df_img_info = pd.read_csv('img_info.csv')

    #df = df.drop('cx3_rut', axis=1)
    #df = df.drop('cy3_rut', axis=1)
    #df = df.drop('cx3_valen', axis=1)
    #df = df.drop('cy3_valen', axis=1)
    #df = df.drop('cx3_rob', axis=1)
    #df = df.drop('cy3_rob', axis=1)
    df = df.drop('file_size', axis=1)

    df_img_info = df_img_info.rename(columns={'new_file': 'shuffled_filename'})
    df_img_info['shuffled_filename'] = df_img_info['shuffled_filename'].str.replace('.png', '')
    df_img_info['shuffled_filename'] = df_img_info['shuffled_filename'] + ".jpg"
    print(df)
    print(df_img_info)
    df = pd.merge(df,df_img_info, on='shuffled_filename', how='left')

    # Determine mean distance per annotator
    names = ['rut', 'valen', 'rob', 'fauve']
    for name in names:
        df['mean_distance_'+ name] = np.sqrt((df['cx2_'+''+ name] - df['cx1_'+ name])**2 + (df['cy2_'+ name] - df['cy1_'+ name])**2)
        df['mean_distance_'+ name] = df['mean_distance_'+ name].astype('int')

    # Mean distance all annotators
    df['mean_distance_annotators'] = (df['mean_distance_rut'] + df['mean_distance_valen'] + df['mean_distance_rob'] +
                                      df['mean_distance_fauve'])/4
    df['mean_distance_annotators'] = df['mean_distance_annotators'].astype('int')

    # Mean fovea
    df['mean_fovea_cx'] = (df['cx3_rut'] + df['cx3_valen'] + df['cx3_rob'] + df['cx3_fauve'])/4
    df['mean_fovea_cx'] = df['mean_fovea_cx'].astype('int')

    df['mean_fovea_cy'] = (df['cy3_rut'] + df['cy3_valen'] + df['cy3_rob'] + df['cy3_fauve'])/4
    df['mean_fovea_cy'] = df['mean_fovea_cy'].astype('int')


    # Create empty columns
    #df["resized_mean_distance_annotators"] = np.nan
    #df["resized_mean_cx_1_2_annotators"] = np.nan
    #df["resized_mean_cy_1_2_annotators"] = np.nan
    #df["resized_mean_fovea_cx"] = np.nan
    #df["resized_mean_fovea_cy"] = np.nan

    # make scaling value below 1
    df["scaling"] = 1/df["scaling"]

    # Delta van toepassing op original naar squared, scaling is wel van squared naar squared resized

    # Get mean distance resized by multiplying with scaling factor
    df["resized_mean_distance_annotators"] = df['mean_distance_annotators'] * df["scaling"]
    # Get new x and coordinate optic disk
    df["resized_mean_cx_1_2_annotators"] = df["mean_cx_1_2_annotators"] * df["scaling"]
    df["resized_mean_cy_1_2_annotators"] = df["mean_cy_1_2_annotators"] * df["scaling"]
    # Get new x and y coordinate fovea
    df["resized_mean_fovea_cx"] = df["mean_fovea_cx"] * df["scaling"]
    df["resized_mean_fovea_cy"] = df["mean_fovea_cy"] * df["scaling"]


    print(df.to_string())
    # Create txt files with information
    for x in range(1000):
        if x == 0:
            print("x",df.iloc[x]["resized_mean_cx_1_2_annotators"])
            print("y",df.iloc[x]["resized_mean_cy_1_2_annotators"])
        # per image give "class center width height", normalize values between 0 and 1 by dividing by image size (640)
        normalized_x_center = df.iloc[x]["resized_mean_cx_1_2_annotators"] / 640
        normalized_y_center = df.iloc[x]["resized_mean_cy_1_2_annotators"] / 640
        normalized_width_height = df.iloc[x]["resized_mean_distance_annotators"] / 640
        combined_information_optic_disk = "0 "+ str(normalized_x_center) + " "+ str(normalized_y_center) + " " + \
                               str(normalized_width_height) + " " + str(normalized_width_height)
        normalized_x_fovea = df.iloc[x]["resized_mean_fovea_cx"] / 640
        normalized_y_fovea = df.iloc[x]["resized_mean_fovea_cy"] / 640
        combined_information_fovea = "1 " + str(normalized_x_fovea) + " "+ str(normalized_y_fovea) + " " + \
                               str(normalized_width_height) + " " + str(normalized_width_height)

        len_string = len(df.iloc[x]["shuffled_filename"])
        name_file = df.iloc[x]["shuffled_filename"][:len_string - 4]

        # nu MET fovea
        string_to_write = combined_information_optic_disk + "\n" + combined_information_fovea

        with open("txt_files_labels/"+ name_file + ".txt", "w+") as f:
            f.writelines(string_to_write)




    """
    for x in range(1000):

        img_name = ''
        if x < 10:
            img = cv2.imread('cfp/DEV0000' + str(x) + '.jpg')
            img_name = 'DEV0000' + str(x) + '.jpg'
        elif x < 100:
            img = cv2.imread('cfp/DEV000' + str(x) + '.jpg')
            img_name = 'DEV000' + str(x) + '.jpg'
        else:
            img = cv2.imread('cfp/DEV00' + str(x) + '.jpg')
            img_name = 'DEV00' + str(x) + '.jpg'

        from PIL import Image
        im = Image.open("cfp/"+img_name)
        width, height = im.size  # Get dimensions

        smallest_side = 0
        if height > width:
            smallest_side = width
        else:
            smallest_side = height

        left = (width - smallest_side) / 2
        top = (height - smallest_side) / 2
        right = (width + smallest_side) / 2
        bottom = (height + smallest_side) / 2

        # Crop the center of the image
        crop_img = im.crop((left, top, right, bottom))

        crop_img.save('resized/resized_'+img_name)
        

        
        
        image = cv2.imread("cfp/"+img_name)
        # Get original width and height
        old_height, old_width, channels = img.shape
        
        smallest_side = 0
        if old_height > old_width:
            smallest_side = old_width
        else:
            smallest_side = old_height

        # Crop door het midden van de image te nemen
        center = image.shape
        x_center = center[1] / 2 - old_width / 2
        y_center = center[0] / 2 - old_height / 2

        crop_img = img[int(y_center):int(y_center + smallest_side), int(x_center):int(x_center + smallest_side)]
        
        # old resizing
        #resized_image = cv2.resize(crop_img, (640, 640))
        
        # Calculate resizeing factors
        Rx = 640 / old_width
        Ry = 640 /old_height


        # Calculate new coordinates
        df.at[x, 'resized_mean_distance_annotators'] = df.iloc[x]["mean_distance_annotators"] * Rx
        df.at[x, 'resized_mean_cx_1_2_annotators'] = df.iloc[x]["mean_cx_1_2_annotators"] * Rx
        df.at[x, 'resized_mean_cy_1_2_annotators'] = df.iloc[x]["mean_cy_1_2_annotators"] * Ry
        df.at[x, 'resized_mean_fovea_cx'] = df.iloc[x]["mean_fovea_cx"] * Rx
        df.at[x, 'resized_mean_fovea_cy'] = df.iloc[x]["mean_fovea_cy"] * Ry

        # Save resized image
        #print(img_name)
        cv2.imwrite('resized/resized_'+img_name, crop_img)

    df['resized_mean_distance_annotators'] = df['resized_mean_distance_annotators'].astype('int')
    df['resized_mean_cx_1_2_annotators'] = df['resized_mean_cx_1_2_annotators'].astype('int')
    df['resized_mean_cy_1_2_annotators'] = df['resized_mean_cy_1_2_annotators'].astype('int')
    df['resized_mean_fovea_cx'] = df['resized_mean_fovea_cx'].astype('int')
    df['resized_mean_fovea_cy'] = df['resized_mean_fovea_cy'].astype('int')
    print(df.to_string())
    """



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
