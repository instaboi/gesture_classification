'''
Date: 2020-09-28 06:35:30
LastEditors: Tianling Lyu
LastEditTime: 2020-09-29 09:06:13
FilePath: \gesture_classification\build_dataset.py
'''

import os
import glob
import csv
import pickle
from argparse import ArgumentParser

import dv
import numpy as np

# function to load .aedat file
def load_aedat_file(filename):
    events = []
    with dv.LegacyAedatFile(filename) as f:
        width = f.frame_width
        height = f.frame_height
        for event in f:
            events.append(event)
    print("{}: [{}, {}]".format(filename, height, width))
    return width, height, events

# function to load .csv file
def load_csv_file(filename):
    csvs = []
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            csvs.append(row)
    return csvs

# convert events to a list containing images and corresponding labels
def events_to_images(width, height, events, label, fps):
    images = {}
    divide = 1000000 / fps
    current_event = 0
    for i_gesture in range(len(label)):
        # label info
        gesture_class = label[i_gesture]["class"]
        start_time = int(label[i_gesture]["startTime_usec"])
        end_time = int(label[i_gesture]["endTime_usec"])
        while events[current_event].timestamp < start_time:
            current_event += 1
        images[str(gesture_class)] = []
        # to frames
        current_starttime = start_time
        current_endtime = current_starttime + divide
        while current_endtime <= end_time:
            img = np.zeros([height, width], dtype=np.int8)
            while events[current_event].timestamp < current_endtime:
                x = events[current_event].x
                y = events[current_event].y
                polarity = events[current_event].polarity
                if polarity == 0:
                    polarity = -1
                # write event to image
                if img[y, x] == 0: 
                    img[y, x] = polarity
                elif img[y, x] == -polarity: # deal with multi-label
                    img[y, x] = 0
                # next event
                current_event += 1
            # store frame and goto the next
            images[str(gesture_class)].append(img)
            current_starttime = current_endtime
            current_endtime += divide
    return images

def load_data_series(base_folder, list_file, fps):
    with open(os.path.join(base_folder, list_file), "r") as f:
        lines = f.readlines()
    data_series = []
    for line in lines:
        aedat_filename = os.path.join(base_folder, line[:-1])
        csv_filename = os.path.join(base_folder, line[:-7] + "_labels.csv")
        # load files
        width, height, events = load_aedat_file(aedat_filename)
        labels = load_csv_file(csv_filename)
        # convert to images
        data_series.append(events_to_images(128, 128, events, labels, fps))
    return data_series

def load_data_series_pre(base_folder, list_file, fps):
    with open(os.path.join(base_folder, list_file), "r") as f:
        lines = f.readlines()
    data_series = []
    count = 0
    for line in lines:
        if count >= 2:
            break
        aedat_filename = os.path.join(base_folder, line[:-1])
        csv_filename = os.path.join(base_folder, line[:-7] + "_labels.csv")
        # load files
        width, height, events = load_aedat_file(aedat_filename)
        labels = load_csv_file(csv_filename)
        # convert to images
        data_series.append(events_to_images(128, 128, events, labels, fps))
        count += 1
    return data_series

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-path", type=str, dest="data_path", default="/home/ubuntu/dataset/DvsGesture/", help="Path to dataset.")
    parser.add_argument("--list-file", type=str, dest="list_file", default="trials_to_train.txt", help="txt file containing target files.")
    parser.add_argument("--save-path", type=str, dest="save_path", default="train.pickle", help="Path to the output file.")
    parser.add_argument("--fps", type=int, dest="fps", default=60, help="Video fps.")
    parser.add_argument("--pre", type=bool, dest="pre", default=False, help="Build pre-train dataset to tune parameters.")
    args = parser.parse_args()
    if args.pre:
        data_series = load_data_series_pre(args.data_path, args.list_file, args.fps)
    else:
        data_series = load_data_series(args.data_path, args.list_file, args.fps)
    with open(args.save_path, "wb") as f:
        pickle.dump(data_series, f)
