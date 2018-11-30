import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os


csv_data = pd.read_csv("/home/ubuntu/Data/AI4AOI/train/train.csv")

train_path = "/home/ubuntu/Data/AI4AOI/train/"
dest_train_path = "/home/ubuntu/Data/AOI_training/train/"

bitmap_list = ["bitmap_files_0", "bitmap_files_1" ,"bitmap_files_2" ,"bitmap_files_3" ,"bitmap_files_4" ,"bitmap_files_5" ,"bitmap_files_6" ,"bitmap_files_7" ,"bitmap_files_8"]


for index, row in csv_data.iterrows():
    print("Processing {} image..".format(index+1))
    list_nan = row.isna()[row.isna() == True].index.get_values()
    missing_images = len(list_nan)
    grayscale_image = cv2.imread(os.path.join(train_path, row['parent_name'], row['png_file']))
    temp_image = None
    for bitmap in bitmap_list:
        if bitmap not in list_nan:
            opt2gray_image = cv2.imread(os.path.join(train_path, row['parent_name'], row[bitmap]))
            if temp_image is None:
                temp_image = cv2.merge((grayscale_image, opt2gray_image))
            else:
                temp_image = cv2.merge((temp_image, opt2gray_image))
    for i in range(len(list_nan)):
        temp_image = cv2.merge((temp_image, grayscale_image))
    print(temp_image.shape)
    assert(temp_image.shape[2] == 10)
    if(row['json_content.component.repair result']==1):
        dest_path = os.path.join(dest_train_path, "Bad_parts")
    else:
        dest_path = os.path.join(dest_train_path, "Good_parts")
        
#         cv2.imwrite(os.path.join(dest_path, row['png_file']), temp_image)