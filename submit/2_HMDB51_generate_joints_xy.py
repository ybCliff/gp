import numpy as np
import os
joints_root_path = "D:/graduation_project/workspace/dataset/HMDB51/"
ori_path = joints_root_path + 'ori/'
x_path = joints_root_path + 'ori_x/'
y_path = joints_root_path + 'ori_y/'
if not os.path.exists(x_path):
    os.makedirs(x_path)
if not os.path.exists(y_path):
    os.makedirs(y_path)
joints_keys = 18

# filename = "0_1.txt"
# pre, post = filename.split('.')
file_list = os.listdir(ori_path)
for filename in file_list:
    file_to_read = open(ori_path + filename, 'r')

    record = file_to_read.readline()
    statistics = file_to_read.readline()
    length = len(statistics.split(' '))
    x_matrix = np.zeros((length, joints_keys))
    y_matrix = np.zeros((length, joints_keys))

    content = file_to_read.read()
    file_to_read.close()
    content = content.split("\n")
    print(filename, length, len(content))
    # assert length == len(content)

    for i in range(len(content)):
        col = content[i].split(' ')
        col.pop()
        # print(col)
        for j in range(len(col)):
            detail = col[j].split(':')          # 'key' 'x,y'
            coor = detail[1].split(',')         # 'x' 'y'
            x_matrix[i][int(detail[0])] = float(coor[0])
            y_matrix[i][int(detail[0])] = float(coor[1])

    file_to_write = open(x_path + filename, 'w')
    for i in range(len(x_matrix)):
        print(','.join([str(k) for k in x_matrix[i]]), file=file_to_write)
    file_to_write.close()
    file_to_write = open(y_path + filename, 'w')
    for i in range(len(y_matrix)):
        print(','.join([str(k) for k in y_matrix[i]]), file=file_to_write)
    file_to_write.close()
# for i in range(length):
#     content = file_to_read.readline()
#     print(content)
