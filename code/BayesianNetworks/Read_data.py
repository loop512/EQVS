import os
import numpy as np


def get_file_names(file_num):
    file_list = []
    suffix = '.g729a'
    for i in range(1, file_num+1):
        file_name = str(i) + suffix
        file_list.append(file_name)
    return file_list


def read_files(foler, file_list):
    temp_data = None
    for file in file_list:
        file_path = folder+file
        data = np.loadtxt(file_path)
        if temp_data is None:
            temp_data = data
        else:
            temp_data = np.concatenate((temp_data, data), axis=0)
    return temp_data


def save_file(folder, numpy_array, header):
    name = folder + 'data.csv'
    np.savetxt(name, numpy_array, header=header, delimiter=',', comments='')


root = 'D:\\博士\\贝叶斯网络\\data\\'
subroots = ['cover', 'stego0.5', 'stego1.0']

for subroot in subroots:
    folder = root + subroot + '\\'
    head = 'L0,L1,L2,L3,P1,P0,C1,S1,GA1,GB1,P2,C2,S2,GA2,GB2'
    file_num = 10
    file_list = get_file_names(file_num)
    data = read_files(folder, file_list)
    save_file(folder, data, head)
