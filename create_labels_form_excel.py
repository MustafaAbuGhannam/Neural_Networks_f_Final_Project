import cv2
from numpy import imag
import pandas as pd
import random
import os
from PIL import Image

def create_label_file(file_1, file_2, num,filename):
    match_pairs = pd.read_csv(file_1, header= None, sep= ',',nrows= num )
    miss_match_pairs = pd.read_csv(file_2,header= None ,sep= ',',nrows= num )
    labels = pd.concat([match_pairs, miss_match_pairs])
    labels= labels.sample(frac=1)
    labels.to_csv(filename, index=False, sep=',', header=0)

def find_match_pairs(file, len_, file_name):
    max_col = file.columns.size
    to_return = []

    while len(to_return) < len_:
        to_add = []
        col = random.randint(0, max_col - 1)
        max_row = file.iat[0, col]
        row1 = random.randint(1, max_row)
        row2 = random.randint(1, max_row)
        to_add.append(file.iat[row1, col])
        to_add.append(file.iat[row2, col])
        to_add.append('0')
        counter = to_return.count(to_add)
        if counter == 0:
            to_return.append(to_add)
        
    
    csv_file = pd.DataFrame(to_return)
    csv_file = csv_file.sample(frac=1)
    csv_file.to_csv(file_name,
                        index=False, sep=',', header=0)
    

    return to_return


def find_miss_match_pairs(file, len_, file_name):
    max_col = file.columns.size
    to_return = []
    while len(to_return) < len_:
        to_add = []
        col1 = random.randint(0, max_col - 1)
        col2 = -1
        while 1:
            col2 = random.randint(0, max_col - 1)
            if col2 != col1:
                break
        max_row1 = file.iat[0, col1]
        max_row2 = file.iat[0, col2]
        row1 = random.randint(1, max_row1)
        row2 = random.randint(1, max_row2)
        to_add.append(file.iat[row1, col1])
        to_add.append(file.iat[row2, col2])
        to_add.append('1')
        counter = to_return.count(to_add)
        if counter == 0:
            to_return.append(to_add)
    csv_file = pd.DataFrame(to_return)
    csv_file = csv_file.sample(frac=1)
    csv_file.to_csv(file_name,
                        index=False, sep=',', header=0)
    
    return to_return


def resize(file):
    min_1 = 0
    min_2 = 0
    path_to_return_1 = ''
    path_to_return_2 = ''
    for i in range(12514):
        path = file.iat[i, 0]
        image = cv2.imread('english_data_set/'+ path, 0)
        if image.shape[0] > min_1:
            min_1 = image.shape[0]
            path_to_return_1 = path

    for i in range(12514):
        path = file.iat[i, 0]
        image = cv2.imread('english_data_set/'+ path, 0)
        if image.shape[1] > min_2:
            min_2 = image.shape[1]
            path_to_return_2 = path
    for i in range(12514):
        path = file.iat[i, 0]
        image = Image.open('english_data_set/'+ path)
        image = image.resize((1760,70))
        image.save('english_data_set/'+ path)
            

    return [min_1, min_2, path_to_return_1, path_to_return_2]
    

    

if __name__ == '__main__':
    
    print('reading files')
    test_file  = pd.read_excel('0-40D.xlsx')
    train_file = pd.read_excel('41-200D.xlsx')
    # all_images  = pd.read_excel('All_Image_Path.xlsx')

    # print('creating pairs....')
    find_match_pairs(test_file, 4000, 'match_labels_from_english_for_test.csv')
    find_match_pairs(train_file, 10000, 'match_labels_from_english_for_train.csv')
    find_miss_match_pairs(test_file, 4000, 'miss_match_labels_from_english_for_test.csv')
    find_miss_match_pairs(train_file, 10000, 'miss_match_labels_from_english_for_train.csv')

    print('creating labels for test and for train...')
    create_label_file('match_labels_from_english_for_test.csv', 'miss_match_labels_from_english_for_test.csv', 2000, 'Test_labels_for_english.csv')
    create_label_file('match_labels_from_english_for_train.csv', 'miss_match_labels_from_english_for_train.csv', 5000, 'Train_labels_for_english.csv')
    
    # resize(all_images)