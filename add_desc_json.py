import os
import json
import csv
from datetime import datetime, timedelta
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import multiprocessing as mp
import shutil
from ransac_icp_cmd import write_json

def find_closest_timestamp(data,target_timestamp):
    closest_timestamp = min(data.keys(),key = lambda k:abs(k-target_timestamp))
    return closest_timestamp

def create_json_data_from_trans(trans):
    data_write={
        "desc":{
            "ego2global":trans.tolist()
        }
    }    
    return data_write

def location_data2transformation(data):
    position = np.array([
         float(data['position_x']),
         float(data['position_y']),
         float(data['position_z'])
    ])
    quat = np.array([
        float(data['orientation_x']),
        float(data['orientation_y']),
        float(data['orientation_z']),
        float(data['orientation_w'])
    ])
    rotation_matrix = R.from_quat(quat).as_matrix()
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = position
    return transformation_matrix

def gnss_data2transformation(data):
    position = np.array([
         float(data['B']),
         float(data['L']),
         float(data['H'])
    ])
    euler = np.array([
        float(data['r']),
        float(data['p']),
        float(data['y'])
    ])
    rotation_matrix = R.from_euler('zxy', euler, degrees=True).as_matrix()
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = position
    return transformation_matrix

# def get_transformation_matrix_from_location_gnss(location_data,gnss_data,first_lidar_timestamp):
#     transformation_gnss={}
#     transformation_location={}
#     for timestamp,location in location_data.items():
#         transformation_location[timestamp] = location_data2transformation(location)
#     for timestamp,gnss in gnss_data.items():
#         transformation_gnss[timestamp] = gnss_data2transformation(gnss)

#     first_lidar_timestamp_location = find_closest_timestamp(location_data,first_lidar_timestamp)
#     first_lidar_timestamp_gnss = find_closest_timestamp(gnss_data,first_lidar_timestamp)

#     first_transformation_location = transformation_location[first_lidar_timestamp_location]
#     first_transformation_gnss = transformation_gnss[first_lidar_timestamp_gnss]   

#     for timestamp,transformation in transformation_location.items():
#         transformation_location[timestamp] = np.array(np.linalg.inv(transformation) @ first_transformation_location)
    
#     for timestamp, transformation in transformation_gnss.items():
#         transformation_gnss[timestamp] = np.array(np.linalg.inv(transformation) @ first_transformation_gnss)
    
#     return transformation_location,transformation_gnss

# def create_json_data_new(transformation_location,transformation_gnss,location_data,gnss_data,gnss_entry,location_entry):
#     if int(gnss_data[gnss_entry]['status'])==2:
#         transformation = transformation_gnss[gnss_entry]
#     else:
#         transformation = transformation_location[location_entry]
    
#     rotation = transformation[:3, :3]
#     euler = R.from_matrix(rotation).as_euler('zxy',degrees=True)
#     translation = transformation[:3, 3]
#     data={
#         "desc":{
#             "ego2global_rotation":{
#             "RotX":euler[0],
#             "RotY":euler[1],
#             "RotZ":euler[2]
#                                 },
#             "ego2global_translation":[translation[0],translation[1],translation[2]],
#             "future":0
#             }
#         }
#     return data

def create_transformation_gnss_location(gnss_data,location_data,gnss_entry,location_entry):
    if int(gnss_data[gnss_entry]['status'])==2:
        data_entry = gnss_data[gnss_entry]
        transform_matrix = gnss_data2transformation(data_entry)

    else:
        data_entry = location_data[location_entry]
        transform_matrix = location_data2transformation(data_entry)
    # data = create_json_data_from_trans(transform_matrix)
    return transform_matrix,data_entry

def parse_timestamp(timestamp):
# Convert yyyy-mmmm-dddd-hhhh-mmmm-ssss to seconds
    dt = datetime.strptime(timestamp, "%Y-%m-%d-%H-%M-%S")
    return int(dt.timestamp())

def read_csv(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = [row for row in reader]
    return headers, rows

def read_txt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    headers = lines[0].strip().split(', ')
    rows = [line.strip().split(', ') for line in lines[1:]]
    return headers, rows

def find_closest_entry(data,target_timestamp):
    closest_entry = min(data,key = lambda entry:abs(float(entry['timestamp'])-target_timestamp))
    return closest_entry

def read_ins_json(file_path):
    with open(file_path,'r') as f:
        data = json.load(f)
    return data

def proces_ex(cm):
    os.system(cm)

def ins_data2transformation(entry):
    quaternion = [entry["localization"]["orientation"]["x"], entry["localization"]["orientation"]["y"],entry["localization"]["orientation"]["z"],entry["localization"]["orientation"]["w"]]
    translation = [entry["localization"]["position"]["x"], entry["localization"]["position"]["y"], entry["localization"]["position"]["z"]]
    rotation = R.from_quat(quaternion).as_matrix()
    translation = np.array(translation).reshape(3, 1)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation.flatten()
    return transformation_matrix

def get_lidar_gnss_transformation_matrix():
    quaternion = [0.0021254234233800515, 0.004810933675242768, 0.7074617812900604,-0.7067320323194207]
    translation = [0.04050681247465682, -0.004926740824803962, -0.5872170205758633]
    rotation = R.from_quat(quaternion).as_matrix()
    translation = np.array(translation).reshape(3, 1)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation.flatten()
    return transformation_matrix 

def create_relative_translation_data(T0,Ti,T_lidar_gnss):
    transformation_matrix_gnss = np.array(np.linalg.inv(Ti) @ T0)
    T_pcd0_pcdi = np.linalg.inv(T_lidar_gnss)@transformation_matrix_gnss@T_lidar_gnss
    data_write = create_json_data_from_trans(T_pcd0_pcdi)
    return data_write

def write_json_label(label_dir,adjusted_timestamp,filename,gnss_data,location_data):
    gnss_entry = find_closest_timestamp(gnss_data, adjusted_timestamp)
    if abs(gnss_entry-adjusted_timestamp)>0.05:
        print(f'[WARN] ----Time differenc between gnss.txt and data timestamp is too large')
    location_entry = find_closest_timestamp(location_data, adjusted_timestamp)
    if abs(location_entry-adjusted_timestamp)>0.05:
        print(f'[WARN] ----Time differenc between location.txt and data timestamp is too large')
    if gnss_entry and location_entry:
        T,entry = create_transformation_gnss_location(gnss_data,location_data,gnss_entry,location_entry)
        data_write = create_json_data_from_trans(T)
        json_path = os.path.join(label_dir,filename,filename + '__desc.json')
        write_json(json_path,data_write)
    return T,entry


def main(x_label_path, y_sweeps_path):
    for dir_name in os.listdir(x_label_path):#dir_name: #ppl_bag_xxx/...  2024-....  
        if dir_name.endswith("__label"):
            dir_name_prefix = dir_name.split('__')[0]
            print(f'[INFO] ----Start bag: {dir_name_prefix}')
            label_dir = os.path.join(x_label_path, dir_name)
            sweep_dir = os.path.join(y_sweeps_path, dir_name_prefix+'__sweeps')
            # Parse alig.csv
            try:
                alig_headers, alig_rows = read_csv(os.path.join(sweep_dir, "timestamp_align_merged.csv"))
            except FileNotFoundError as e:
                print(f'[ERROR] ----Skipping bag {dir_name_prefix} due to missing file: {e}')
                continue
            # Read gnss.txt and location.txt
            try:
                gnss_headers, gnss_rows = read_txt(os.path.join(sweep_dir, "gnss.txt"))
            except FileNotFoundError as e:
                print(f'[ERROR] ----Skipping bag {dir_name_prefix} due to missing file: {e}')
                continue
            try:
                location_headers, location_rows = read_txt(os.path.join(sweep_dir, "location.txt"))
            except FileNotFoundError as e:
                print(f'[ERROR] ----Skipping bag {dir_name_prefix} due to missing file: {e}')
                continue

            gnss_data = {float(row[2]): dict(zip(gnss_headers, row)) for row in gnss_rows}
            location_data = {float(row[0]): dict(zip(location_headers, row)) for row in location_rows}

            first_lidar_timestamp = float(alig_rows[0][1].replace('.pcd',''))
            # location_transfomations, gnss_transformations = get_transformation_matrix_from_location_gnss(location_data,gnss_data,first_lidar_timestamp)
            base_timestamp = parse_timestamp(dir_name.split('_')[0])

            ##### for label
            filenames = [f for f in os.listdir(label_dir) if f.isdigit()]
            filenames_sorted = sorted(filenames,key=int)
            timestamp_diff0 = int(filenames_sorted[0])
            adjusted_timestamp0 = abs(base_timestamp - float(timestamp_diff0)/1000.0) + first_lidar_timestamp
            T0,enrty0 = write_json_label(label_dir,adjusted_timestamp0,filenames_sorted[0],gnss_data,location_data)
            for i in range(1,len(filenames_sorted)):
                timestamp_diff = int(filenames_sorted[i])
                adjusted_timestamp = abs(base_timestamp - float(timestamp_diff)/1000.0) + first_lidar_timestamp
                Ti,entryi = write_json_label(label_dir,adjusted_timestamp,filenames_sorted[i],gnss_data,location_data)
                data_write = create_relative_translation_data(T0,Ti,T_lidar_gnss)
                label_json_file = os.path.join(label_dir,filenames_sorted[i],filenames_sorted[i] + '__desc.json')
                write_json(label_json_file,data_write)

            ##### for sweeps
            for sub2_dir_name in os.listdir(sweep_dir):
                if sub2_dir_name.isdigit():#1706342918400/...
                    additional_first_frame_folder_label = os.path.join(label_dir,sub2_dir_name)
                    additional_first_frame_folder_sweeps = os.path.join(sweep_dir,sub2_dir_name,sub2_dir_name)
                    if os.path.exists(additional_first_frame_folder_sweeps):
                        shutil.rmtree(additional_first_frame_folder_sweeps)
                    if os.path.exists(additional_first_frame_folder_label):
                        sweep_path = os.path.join(sweep_dir,sub2_dir_name)
                        shutil.copytree(additional_first_frame_folder_label,additional_first_frame_folder_sweeps)
                        filenames = [f for f in os.listdir(sweep_path) if f.isdigit()]
                        filenames_sorted = sorted(filenames,key=int)
                        tmp_ins_json = './tmp_ins_json/' + sub2_dir_name + '.json'
                        entrys = []
                        for i in range(len(filenames_sorted)):
                            timestamp_diff = int(filenames_sorted[i])
                            adjusted_timestamp = abs(base_timestamp - float(timestamp_diff)/1000.0) + first_lidar_timestamp
                            gnss_entry = find_closest_timestamp(gnss_data, adjusted_timestamp)
                            location_entry = find_closest_timestamp(location_data, adjusted_timestamp)
                            trans,entry = create_transformation_gnss_location(gnss_data,location_data,gnss_entry,location_entry)
                            entrys.append(entry)
                        with open(tmp_ins_json, 'w') as file:
                            json.dump(entrys, file, indent=4)
                        tmp = 'python3 '+ 'ransac_icp_cmd.py ' + ' ' + sweep_path + ' ' + tmp_ins_json + ' ' + additional_first_frame_folder_sweeps
                        cmd.append(tmp)
                    else:
                        print(f'[WARN] ----Skipping frame in sweeeps {sub2_dir_name}')
            with mp.Pool(processes=12) as pool:
                results = pool.starmap_async(proces_ex, [(cm,) for cm in cmd])
                pool.close()
                pool.join()

            print(f'[INFO] ----Finish bag: {dir_name_prefix}')
        elif dir_name.endswith("_label"):# maichi

            lidar2rfu_transform = np.eye(4)
            rotation = [[0, -1, 0],[1, 0, 0],[0, 0, 1]]
            lidar2rfu_transform[:3, :3] = rotation
            T_lidar_gnss = get_lidar_gnss_transformation_matrix()@lidar2rfu_transform

            dir_name_prefix = dir_name.split('_label')[0]
            print(f'[INFO] ----Start bag: {dir_name_prefix}')
            label_dir = os.path.join(x_label_path, dir_name)
            sweep_dir = os.path.join(y_sweeps_path, dir_name_prefix+'_sweeps')
            cmd = []
            all_ins_file = os.path.join(sweep_dir,'all_ins.json')
            if os.path.isfile(all_ins_file):
                ins_data = read_ins_json(all_ins_file)
                ##### for label
                filenames = [f for f in os.listdir(label_dir) if f.isdigit()]
                filenames_sorted = sorted(filenames,key=int)

                pcd_timestamp0 = float(filenames_sorted[0])/1000.0
                entry0 = find_closest_entry(ins_data,pcd_timestamp0)
                T0 = ins_data2transformation(entry0)
                data_write = create_json_data_from_trans(T0)
                label_json_file = os.path.join(label_dir,filenames_sorted[0],filenames_sorted[0] + '__desc.json')
                write_json(label_json_file,data_write)

                for i in range(1,len(filenames_sorted)):
                    label_json_file = os.path.join(label_dir,filenames_sorted[i],filenames_sorted[i] + '__desc.json')
                    if os.path.exists(label_json_file):
                            continue 
                    pcd_timestamp = float(filenames_sorted[i])/1000.0
                    entryi = find_closest_entry(ins_data,pcd_timestamp)
                    Ti = ins_data2transformation(entryi)
                    data_write = create_relative_translation_data(T0,Ti,T_lidar_gnss)
                    write_json(label_json_file,data_write)

                ##### for sweeps   
                for sub2_dir_name in os.listdir(sweep_dir):
                    if sub2_dir_name.isdigit():#1706342918400/...
                        additional_first_frame_folder_label = os.path.join(label_dir,sub2_dir_name)
                        additional_first_frame_folder_sweeps = os.path.join(sweep_dir,sub2_dir_name,sub2_dir_name)
                        if os.path.exists(additional_first_frame_folder_sweeps):
                            shutil.rmtree(additional_first_frame_folder_sweeps)

                        if os.path.exists(additional_first_frame_folder_label):
                            sweep_path = os.path.join(sweep_dir,sub2_dir_name)
                            shutil.copytree(additional_first_frame_folder_label,additional_first_frame_folder_sweeps)

                            filenames = [f for f in os.listdir(sweep_path) if f.isdigit()]
                            filenames_sorted = sorted(filenames,key=int)
                            tmp_ins_json = './tmp_ins_json/' + sub2_dir_name + '.json'
                            entrys = []
                            for i in range(len(filenames_sorted)):
                                pcd_timestamp = float(filenames_sorted[i])/1000.0
                                entry = find_closest_entry(ins_data,pcd_timestamp)
                                entrys.append(entry)
                            with open(tmp_ins_json, 'w') as file:
                                json.dump(entrys, file, indent=4)
                            tmp = 'python3 '+ 'ransac_icp_cmd.py ' + ' ' + sweep_path + ' ' + tmp_ins_json + ' ' + additional_first_frame_folder_sweeps
                            cmd.append(tmp)
                        else:
                            print(f'[WARN] ----Skipping frame in sweeeps {sub2_dir_name}')
                with mp.Pool(processes=12) as pool:
                    results = pool.starmap_async(proces_ex, [(cm,) for cm in cmd])
                    pool.close()
                    pool.join()
            else:
                print(f'no ins.json file in {sweep_dir}')
        else:
            print('[ERROR] Could not find __label direction:',dir_name)
            continue

if __name__ == "__main__":
    input_path = '/media/robosense/data_all/maichi'
    label_folders = []
    sweeps_folders = []

    for subdir_name in os.listdir(input_path):
        subdir_path = os.path.join(input_path, subdir_name)
        if os.path.isdir(subdir_path):
            if subdir_name.endswith('_label'):
                prefix = subdir_name[:-6]
                label_folders.append(prefix)
            elif subdir_name.endswith('_sweeps'):
                prefix = subdir_name[:-7]
                sweeps_folders.append(prefix)

    matching_folders = []
    for prefix in label_folders:
        if prefix in sweeps_folders:
            print(f'[INFO] Start batch: {prefix}')
            main(os.path.join(input_path, prefix+'_label'), os.path.join(input_path, prefix+'_sweeps'))#bd_xxx_label.sweeps/...
            print(f'[INFO] Finish batch: {prefix}')
            # matching_folders.append((label_folders[prefix], sweeps_folders[prefix]))
        else:
            print(f'[ERROR] No match _sweeps folder for prefix: {prefix}')
            continue