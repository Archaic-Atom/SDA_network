# -*- coding: utf-8 -*-
import os
import glob


# define sone struct
ROOT_PATH = 'I:/Documents/Database/'  # root path
FOLDER_NAME_FORMAT = '%04d/'
RAW_DATA_FOLDER = 'frames_cleanpass/TRAIN/%s/'
LABLE_FOLDER = 'disparity/TRAIN/%s/'
LEFT_FOLDER = 'left/'
RIGHT_FOLDER = 'right/'
FILE_NAME = '%s%04d'
RAW_DATA_TYPE = '.png'
LABEL_TYPE = '.pfm'
TrainListPath = './Dataset/trainlist_scene_flow.txt'
LabelListPath = './Dataset/labellist_scene_flow.txt'
FOLDER_NUM = 750
ID_NUM = 3


def convert_num_to_char(folder_id: int):
    res = 'None'
    if folder_id == 0:
        res = 'A'
    elif folder_id == 1:
        res = 'B'
    elif folder_id == 2:
        res = 'C'
    return res


def gen_raw_path(folder_id: int, folder_num: int, file_folder: str, num: int) -> str:
    path = ROOT_PATH + RAW_DATA_FOLDER % folder_id + FOLDER_NAME_FORMAT % folder_num + \
        FILE_NAME % (file_folder, num) + RAW_DATA_TYPE
    return path


def gen_label_path(folder_id: int, folder_num: int, file_folder: str, num: int)->str:
    path = ROOT_PATH + LABLE_FOLDER % folder_id + FOLDER_NAME_FORMAT % folder_num + \
        FILE_NAME % (file_folder, num) + LABEL_TYPE
    return path


def open_file():
    if os.path.exists(TrainListPath):
        os.remove(TrainListPath)
    if os.path.exists(LabelListPath):
        os.remove(LabelListPath)

    fd_train_list = open(TrainListPath, 'a')
    fd_label_list = open(LabelListPath, 'a')

    return fd_train_list, fd_label_list


def output_data(outputFile, data):
    outputFile.write(str(data) + '\n')
    outputFile.flush()


def gen_list_flyingthing(fd_train_list: object, fd_label_list: object)->int:
    total = 0
    for idx in range(ID_NUM):
        for folder_num in range(FOLDER_NUM):
            num = 6
            while True:
                folder_id = convert_num_to_char(idx)
                raw_left_path = gen_raw_path(folder_id, folder_num, LEFT_FOLDER, num)
                raw_right_path = gen_raw_path(folder_id, folder_num, RIGHT_FOLDER, num)
                lable_path = gen_label_path(folder_id, folder_num, LEFT_FOLDER, num)

                raw_left_path_is_exists = os.path.exists(raw_left_path)
                raw_right_path_is_exists = os.path.exists(raw_right_path)
                lable_path_is_exists = os.path.exists(lable_path)

                if (not raw_left_path_is_exists) and \
                        (not raw_right_path_is_exists) and (not lable_path_is_exists):
                    break
                output_data(fd_train_list, raw_left_path)
                output_data(fd_train_list, raw_right_path)
                output_data(fd_label_list, lable_path)
                num = num + 1
                total = total + 1
    return total


def gen_list_driving(fd_train_list: object, fd_label_list: object)->int:
    folder_list = ['15mm_focallength/scene_backwards/fast',
                   '15mm_focallength/scene_backwards/slow',
                   '15mm_focallength/scene_forwards/fast',
                   '15mm_focallength/scene_forwards/slow',
                   '35mm_focallength/scene_backwards/fast',
                   '35mm_focallength/scene_backwards/slow',
                   '35mm_focallength/scene_forwards/fast',
                   '35mm_focallength/scene_forwards/slow']
    total = produce_list(folder_list, fd_train_list, fd_label_list)

    return total


def gen_list_monkey(fd_train_list: object, fd_label_list: object)->int:
    folder_list = ['a_rain_of_stones_x2',
                   'eating_camera2_x2',
                   'eating_naked_camera2_x2',
                   'eating_x2',
                   'family_x2',
                   'flower_storm_augmented0_x2',
                   'flower_storm_augmented1_x2',
                   'flower_storm_x2',
                   'funnyworld_augmented0_x2',
                   'funnyworld_augmented1_x2',
                   'funnyworld_camera2_augmented0_x2',
                   'funnyworld_camera2_augmented1_x2',
                   'funnyworld_camera2_x2',
                   'funnyworld_x2',
                   'lonetree_augmented0_x2',
                   'lonetree_augmented1_x2',
                   'lonetree_difftex2_x2',
                   'lonetree_difftex_x2',
                   'lonetree_winter_x2',
                   'lonetree_x2',
                   'top_view_x2',
                   'treeflight_augmented0_x2',
                   'treeflight_augmented1_x2',
                   'treeflight_x2']
    total = produce_list(folder_list, fd_train_list, fd_label_list)

    return total


def produce_list(folder_list: list, fd_train_list: object, fd_label_list: object)->int:
    total = 0
    for i in range(len(folder_list)):
        img_folder_path = ROOT_PATH + RAW_DATA_FOLDER % folder_list[i]
        gt_foler_path = ROOT_PATH + LABLE_FOLDER % folder_list[i]

        # print img_folder_path

        left_files = glob.glob(img_folder_path + LEFT_FOLDER + '*' + RAW_DATA_TYPE)

        for j in range(len(left_files)):
            name = os.path.basename(left_files[j])
            pos = name.find('.png')
            name = name[0:pos]
            # print name

            left_img_path = img_folder_path + LEFT_FOLDER + name + RAW_DATA_TYPE
            right_img_path = img_folder_path + RIGHT_FOLDER + name + RAW_DATA_TYPE
            gt_img_path = gt_foler_path + LEFT_FOLDER + name + LABEL_TYPE

            raw_left_path_is_exists = os.path.exists(left_img_path)
            raw_right_path_is_exists = os.path.exists(right_img_path)
            lable_path_is_exists = os.path.exists(gt_img_path)

            if (not raw_left_path_is_exists) and \
                    (not raw_right_path_is_exists) and (not lable_path_is_exists):
                print("\"" + left_img_path + "\"" + "is not exist!!!")
                break

            #data_str = left_img_path + ',' + right_img_path + ',' + gt_img_path
                #output_data(fd_train_list, data_str)
            output_data(fd_train_list, left_img_path)
            output_data(fd_train_list, right_img_path)
            output_data(fd_label_list, gt_img_path)
            total = total + 1

    return total


def main():
    fd_train_list, fd_label_list = open_file()
    flying_num = gen_list_flyingthing(fd_train_list, fd_label_list)
    print(flying_num)
    driving_num = gen_list_driving(fd_train_list, fd_label_list)
    print(driving_num)
    monkey_num = gen_list_monkey(fd_train_list, fd_label_list)
    print(monkey_num)
    total = flying_num + driving_num + monkey_num
    print(total)


if __name__ == '__main__':
    main()
