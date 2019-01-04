import numpy as np
import cv2 as cv
import pytube
from pytube import YouTube
import os

src_path_train = '/home/SENSETIME/panzheng/Documents/Run/stereo-mag/RealEstate10K/train'
src_path_test = '/home/SENSETIME/panzheng/Documents/Run/stereo-mag/RealEstate10K/test'
path_dataset = '/media/SENSETIME\\panzheng/Data/stereo_dataset/'
train_v = 'train_video'
train_img = 'train_image/'
test_v = 'test_video'
test_img = 'test_image/'
test_pos = 'test_cam_pos/'
train_pos = 'train_cam_pos/'

#read txt, download videos and crop image frames

#test file
test_files = os.listdir(src_path_test)
test_url = []
test_timestamp = []
count = 0
for file in test_files:
    data = []
    timestamp = []
    file_path = src_path_test + '/' + file
    with open(file_path, 'r') as f:
        for line in f:
            data.append(line)
    url = data[0]
    for i in range(1, len(data)):
        tmp_list = data[i].split()
        timestamp.append(int(tmp_list[0])/1000) #round timestamp into integer
        #data[i] = data[i] + '\n'
    #write camera parameters
    cam_pos_name = 'test_pos_v%d.txt' % count
    count += 1
    with open(path_dataset + test_pos + cam_pos_name, 'w') as p:
        p.writelines(data[1:])
    test_url.append(url)
    test_timestamp.append(timestamp)

#get videos
img_count = 0
for i in range(len(test_url)):
    #download youtube video from url
    if i % 100 == 0:
        print('Processed ' + str(i) + ' test videos...')
    ind = img_count/200000
    img_subfile = path_dataset + test_img + 'test_i%d' % ind
    isExists = os.path.exists(img_subfile)
    if not isExists:
        os.mkdir(img_subfile)

    try:
        yt = YouTube(test_url[i])
        stream = yt.streams.first()
        stream.download(path_dataset + test_v)
        default_name = stream.default_filename
        video_name = 'test_v%d.mp4' % i
        new_path = path_dataset + test_v +'/' + video_name
        os.rename(path_dataset + test_v + '/' + default_name, new_path)
        print('finished downloading ' + video_name)

        #get image frames
        video = cv.VideoCapture(new_path)
        for j in range(len(test_timestamp[i])):
            image_name = img_subfile + '/' + 'test_v%d_image%d.jpg' % (i, j)
            video.set(cv.CAP_PROP_POS_MSEC, test_timestamp[i][j])
            success, image = video.read()
            if success == True:
                cv.imwrite(image_name, image)
                img_count += 1
            else:
                print('problem with video %d, image %d') % (i, j)
        os.remove(new_path) #remove video to save memory
        print('finished processing ' + video_name)
    except: #pytube.exceptions.RegexMatchError/pytube.exceptions.VideoUnavailable/...
        print('skipped error video %d') % i

print
print('Finished test set!')
print

#train file
train_files = os.listdir(src_path_train)
train_url = []
train_timestamp = []
count = 0
for file in train_files:
    data = []
    timestamp = []
    file_path = src_path_train + '/' + file
    with open(file_path, 'r') as f:
        for line in f:
            data.append(line)
    url = data[0]
    for i in range(1, len(data)):
        tmp_list = data[i].split()
        timestamp.append(int(tmp_list[0])/1000) #round timestamp into integer
        #data[i] = data[i] + '\n'
    #write camera parameters
    cam_pos_name = 'train_pos_v%d.txt' % count
    count += 1
    # with open(path_dataset + train_pos + cam_pos_name, 'w') as p:
    #     p.writelines(data[1:])
    train_url.append(url)
    train_timestamp.append(timestamp)

#get videos
count_img = 8675060
for i in range(68326, len(train_url)):
    #download youtube video from url
    if i % 100 == 0:
        print('Processed ' + str(i) + ' train videos...')
    ind = count_img/200000
    img_subfile = path_dataset + train_img + 'train_i%d' % ind
    isExists = os.path.exists(img_subfile)
    if not isExists:
        os.mkdir(img_subfile)

    try:
        yt = YouTube(train_url[i])
        stream = yt.streams.first()
        stream.download(path_dataset + train_v)
        default_name = stream.default_filename
        video_name = 'train_v%d.mp4' % i
        new_path = path_dataset + train_v +'/' + video_name
        os.rename(path_dataset + train_v + '/' + default_name, new_path)
        print('finished downloading ' + video_name)

        #get image frames
        video = cv.VideoCapture(new_path)
        for j in range(len(train_timestamp[i])):
            image_name = img_subfile + '/' + 'train_v%d_image%d.jpg' % (i, j)
            video.set(cv.CAP_PROP_POS_MSEC, train_timestamp[i][j])
            success, image = video.read()
            if success == True:
                cv.imwrite(image_name, image)
                count_img += 1
            else:
                print('problem with video %d, image %d') % (i, j)
        os.remove(new_path)
        print('finished processing ' + video_name)
    except: #pytube.exceptions.RegexMatchError/pytube.exceptions.VideoUnavailable/...
        print('skipped error video %d') % i

print('Finished train set!')
