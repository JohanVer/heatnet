import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import os
import os.path
import fnmatch
from PIL import Image
import random
import cv2
import torch
import numpy as np
import copy
from glob import glob
import matplotlib.pyplot as plt

# for darkness calculations
import ephem
import math
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

image_stats = {'mean': [0.35675976, 0.37380189, 0.3764753],
               'std': [0.32064945, 0.32098866, 0.32325324]}


def applyClaheCV(clahe, images):
    for idx in range(len(images)):
        image_cv = cv2.cvtColor(images[idx], cv2.COLOR_RGB2LAB)
        lab_planes = cv2.split(image_cv)
        lab_planes[0] = clahe.apply(lab_planes[0])
        image_cv = cv2.merge(lab_planes)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_LAB2RGB)
        images[idx] = image_cv
    return images

def normalizeRGBBatch(images, stats=image_stats):
    mean = torch.Tensor(stats['mean']).cuda()
    std = torch.Tensor(stats['std']).cuda()
    for image in images:
            for c in range(image.size(1)):
                image[:, c, ...] = ((image[:, c, ...] - mean[c]) / std[c])
    return images

def readFiles(path):
    with open(path) as file:
        content = file.readlines()
        content = [x.strip() for x in content]
    return content

def searchForFiles(name, path):
    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, name):
            matches.append(os.path.join(root, filename))
    return matches

def getImageStats():
    return image_stats

def filter_test_data(paths, stamps):
    filtered_paths = []
    for p in paths:
        split_path = os.path.basename(p).replace('.', '_').split('_')
        digits = []
        for s in split_path:
            if s.isdigit():
                digits.append(int(s))
        test_file = False
        for t in stamps:
            if digits[0] == t[0] and digits[1] == t[1]:
                pass
            else:
                test_file = True

        if test_file:
            filtered_paths.append(p)

    return filtered_paths

def stampSortFun(file):
    split_path = os.path.basename(file).split('_')
    digits = []
    for s in split_path:
        fn = s.replace('.png', '')
        if fn.isdigit():
            digits.append(int(fn))

    total = str(digits[0]).zfill(10) + "." + str(digits[1]).zfill(10)

    return float(total)

def getTestStamps(files):
    stamps = []
    for filename in files:
        split_path = filename.split('_')
        digits = []
        for s in split_path:
            if s.isdigit():
                digits.append(int(s))

        total = str(digits[0]).zfill(10) + str(digits[1]).zfill(10)
        stamps.append(int(total))
    return stamps

def sort_day_night(fl_ir_files, is_summer_time=False):

    fl_rgb_day_files, fl_ir_day_files, fl_label_day_files = [],[],[]
    fl_rgb_night_files, fl_ir_night_files, fl_label_night_files = [],[],[]
    sun_altitudes_day, sun_altitudes_night = [],[]
    observer = ephem.Observer()
    observer.lat, observer.lon, observer.elevation = '47.9959', '7.85222', 278  # freiburg GPS coordinates

    for fl_ir_file in fl_ir_files:
        # create filename
        fl_rgb_file = fl_ir_file.replace('fl_ir_aligned', 'fl_rgb')
        fl_label_file = fl_ir_file.replace('fl_ir_aligned', 'fl_rgb_labels')

        if not os.path.exists(fl_rgb_file):
            continue

        if 'night' in fl_ir_file:  # night

            # night - no labels are available
            fl_rgb_night_files.append(fl_rgb_file)
            fl_ir_night_files.append(fl_ir_file)
            sun_altitudes_night.append(0)

        else:  # day

            # day - labels should be available
            if not os.path.exists(fl_label_file):
                # print('Could not find {}'.format(fl_label_file))
                continue

            fl_rgb_day_files.append(fl_rgb_file)
            fl_ir_day_files.append(fl_ir_file)
            fl_label_day_files.append(fl_label_file)
            sun_altitudes_day.append(0)


    return fl_rgb_day_files, fl_ir_day_files, \
           fl_rgb_night_files, fl_ir_night_files, \
           fl_label_day_files, fl_label_night_files, \
           sun_altitudes_day, sun_altitudes_night


class ThermalDataLoaderInfer(data.Dataset):
    def __init__(self, db_path):

        self.list_dataset_paths = []

        fl_ir_files = glob(os.path.join(db_path, 'fl_ir_aligned/*.png'))
        # fl_rgb_files = glob(os.path.join(db_path, 'fl_rgb/*.png'))

        fl_ir_files.sort(key=stampSortFun)
        # fl_rgb_files.sort(key=stampSortFun)

        # self.fl_rgb_files = fl_rgb_files
        self.fl_ir_files = fl_ir_files

        print('Found %d image-pairs for inference' % (len(self.fl_ir_files)))

    def __getitem__(self, index):
        # get day files from index
        # rgb_day_file = self.fl_rgb_files[index]
        ir_day_file = self.fl_ir_files[index]
        rgb_day_file = ir_day_file.replace('fl_ir_aligned', 'fl_rgb')

        # READ
        rgb_day = cv2.imread(rgb_day_file)
        ir_day = cv2.imread(ir_day_file, cv2.IMREAD_ANYDEPTH)

        # COLORS
        rgb_day = cv2.cvtColor(rgb_day, cv2.COLOR_BGR2RGB)

        # resizing
        res = (960, 320)
        rgb_day = cv2.resize(rgb_day, res , interpolation=cv2.INTER_LINEAR)
        ir_day = cv2.resize(ir_day, res, interpolation=cv2.INTER_LINEAR)

        # Crop results in 320 * 700
        rgb_day = rgb_day[:, 150:850, :]
        ir_day = ir_day[:, 150:850]

        # normalize IR data (is in range 0, 2**16 --> crop to relevant range(20800, 27000))
        minval = 21800
        maxval = 25000

        ir_day[ir_day < minval] = minval
        ir_day[ir_day > maxval] = maxval

        ir_day = (ir_day - minval) / (maxval - minval)

        ir_day = ir_day.astype(np.float32)

        ir_day = Image.fromarray(ir_day)
        rgb_day = Image.fromarray(rgb_day)

        rgb_day = F.to_tensor(rgb_day)
        ir_day = F.to_tensor(ir_day)

        # Normalization
        rgb_org = rgb_day.clone()
        ir_org = ir_day.clone()
        rgb_day = F.normalize(rgb_day, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ir_day = F.normalize(ir_day, mean=[0.5], std=[0.5])

        out_dict = {}
        out_dict['rgb'] = rgb_day
        out_dict['rgb_org'] = rgb_org
        out_dict['ir'] = ir_day
        out_dict['ir_org'] = ir_org
        return out_dict

    def __len__(self):
        return len(self.fl_ir_files)

class ThermalDataLoader(data.Dataset):
    def __init__(self, db_path, contrast_enhancement=False, load_right=True, split=None, test_stamps=None, db_stats=None):

        self.list_dataset_paths = []

        fl_rgb_files = glob(os.path.join(db_path, '*/*/fl_rgb/*.png'))
        fl_label_files = glob(os.path.join(db_path, '*/*/fl_rgb_labels/*.png'))
        fl_ir_files = glob(os.path.join(db_path, '*/*/fl_ir_aligned/*.png'))

        fl_rgb_files.sort()
        fl_ir_files.sort()
        fl_label_files.sort()

        if test_stamps:
            # fl_rgb_files = filter_test_data(fl_rgb_files, test_stamps)
            # fl_label_files = filter_test_data(fl_label_files, test_stamps)
            fl_ir_files = filter_test_data(fl_ir_files, test_stamps)

        self.fl_rgb_day_files, self.fl_ir_day_files, \
        self.fl_rgb_night_files, self.fl_ir_night_files, \
        self.fl_label_day_files, self.fl_label_night_files, \
        self.sun_altitudes_day, self.sun_altitudes_night = sort_day_night(fl_ir_files)


        self.fl_rgb_day_files_train,   self.fl_rgb_day_files_test, \
        self.fl_ir_day_files_train,    self.fl_ir_day_files_test, \
        self.fl_label_day_files_train, self.fl_label_day_files_test, \
        self.sun_altitudes_day_train, self.sun_altitudes_day_test = train_test_split(self.fl_rgb_day_files,
                                                                                      self.fl_ir_day_files,
                                                                                      self.fl_label_day_files,
                                                                                      self.sun_altitudes_day,
                                                                                      test_size = 0.01,
                                                                                      random_state = 42)
        self.fl_rgb_night_files_train, self.fl_rgb_night_files_test, \
        self.fl_ir_night_files_train, self.fl_ir_night_files_test, \
        self.sun_altitudes_night_train, self.sun_altitudes_night_test = train_test_split(self.fl_rgb_night_files,
                                                                                       self.fl_ir_night_files,
                                                                                        self.sun_altitudes_night,
                                                                                       test_size = 0.01,
                                                                                       random_state = 42)


        if split == 'train':
            self.fl_rgb_day_files = self.fl_rgb_day_files_train
            self.fl_ir_day_files = self.fl_ir_day_files_train
            self.fl_rgb_night_files = self.fl_rgb_night_files_train
            self.fl_ir_night_files = self.fl_ir_night_files_train
            self.fl_label_day_files = self.fl_label_day_files_train
            self.sun_altitudes_night = self.sun_altitudes_night_train
            self.sun_altitudes_day = self.sun_altitudes_day_train

        elif split == 'test':
            self.fl_rgb_day_files = self.fl_rgb_day_files_test
            self.fl_ir_day_files = self.fl_ir_day_files_test
            self.fl_rgb_night_files = self.fl_rgb_night_files_test
            self.fl_ir_night_files = self.fl_ir_night_files_test
            self.fl_label_day_files = self.fl_label_day_files_test
            self.sun_altitudes_night = self.sun_altitudes_night_test
            self.sun_altitudes_day = self.sun_altitudes_day_test

        else:
            ValueError('unknown split')


        print('-----------------------------')

        print(len(self.fl_rgb_day_files))
        print(len(self.fl_rgb_night_files))
        print(len(self.fl_ir_day_files))
        print(len(self.fl_ir_night_files))
        print(len(self.fl_label_day_files))

        self.rgb_day_files = self.fl_rgb_day_files
        self.label_day_files = self.fl_label_day_files
        self.ir_day_files = self.fl_ir_day_files
        self.rgb_night_files = self.fl_rgb_night_files
        self.ir_night_files = self.fl_ir_night_files

        # define modificators
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.contrast_enhancement = contrast_enhancement
        self.load_right = load_right

        print('Day image Tuples Found: %d' % (len(self.rgb_day_files)))
        print('Night image Tuples Found: %d' % (len(self.rgb_night_files)))

        self.length = len(self.rgb_day_files)
        print("Current number of image pair in thermal dataset: %d " % (self.length))

        self.width = 640
        self.height = 320
        self.db_stats = db_stats

    def __getitem__(self, index):
        # get day files from index
        rgb_day_file = self.rgb_day_files[index]
        ir_day_file = self.ir_day_files[index]
        label_day_file = self.label_day_files[index]

        sun_altitude_day = self.sun_altitudes_day[index]

        # get night files randomly
        rand_idx = random.randint(0, len(self.fl_rgb_night_files) - 1)
        rgb_night_file = self.rgb_night_files[rand_idx]
        ir_night_file = self.ir_night_files[rand_idx]
        sun_altitude_night = self.sun_altitudes_night[rand_idx]

        # READ
        rgb_day = cv2.imread(rgb_day_file)
        ir_day = cv2.imread(ir_day_file, cv2.IMREAD_ANYDEPTH)
        rgb_night = cv2.imread(rgb_night_file)
        ir_night = cv2.imread(ir_night_file, cv2.IMREAD_ANYDEPTH)
        label_day = cv2.imread(label_day_file, cv2.IMREAD_GRAYSCALE)

        # COLORS
        rgb_day = cv2.cvtColor(rgb_day, cv2.COLOR_BGR2RGB)
        rgb_night = cv2.cvtColor(rgb_night, cv2.COLOR_BGR2RGB)

        # resizing
        res = (960, 320)
        rgb_day = cv2.resize(rgb_day, res , interpolation=cv2.INTER_LINEAR)
        ir_day = cv2.resize(ir_day, res, interpolation=cv2.INTER_LINEAR)
        rgb_night = cv2.resize(rgb_night, res, interpolation=cv2.INTER_LINEAR)
        ir_night = cv2.resize(ir_night, res, interpolation=cv2.INTER_LINEAR)
        label_day = cv2.resize(label_day, res, interpolation=cv2.INTER_NEAREST)

        if self.contrast_enhancement:
            applyClaheCV(self.clahe, rgb_day)
            applyClaheCV(self.clahe, rgb_night)

        # Crop results in 320 * 700
        ir_day = ir_day[:, 150:850]
        ir_night = ir_night[:, 150:850]
        label_day = label_day[:, 150:850]
        rgb_day = rgb_day[:, 150:850, :]
        rgb_night = rgb_night[:, 150:850, :]

        i, j, h, w = transforms.RandomCrop.get_params(Image.fromarray(rgb_day), (self.height, self.width))

        ir_day = ir_day[i:(i+h), j:(j+w)]
        ir_night = ir_night[i:(i+h), j:(j+w)]
        label_day = label_day[i:(i+h), j:(j+w)]
        rgb_day = rgb_day[i:(i+h), j:(j+w), :]
        rgb_night = rgb_night[i:(i+h), j:(j+w), :]

        # normalize IR data (is in range 0, 2**16 --> crop to relevant range(20800, 27000))
        minval = 21800
        maxval = 25000

        ir_day[ir_day < minval] = minval
        ir_day[ir_day > maxval] = maxval

        ir_night[ir_night < minval] = minval
        ir_night[ir_night > maxval] = maxval

        ir_day = (ir_day - minval) / (maxval - minval)
        ir_night = (ir_night - minval) / (maxval - minval)

        # Modality block dropping (i_d, j_d, h_d, w_d)
        drop_lenght_h = int(random.uniform(100, 300))
        drop_lenght_w = int(random.uniform(100, 500))
        i_d, j_d, h_d, w_d = transforms.RandomCrop.get_params(Image.fromarray(rgb_day),
                                                             (drop_lenght_h, drop_lenght_w))
        mod_drop_params = torch.Tensor([i_d, j_d, h_d, w_d])

        ''' Perform other data augmentations (random crop already implemented):
                    - FlipLR 
                    - Normalize to [-1, 1]
                    - Rotate
                '''
        # convert to PIL images
        ir_day = ir_day.astype(np.float32)
        ir_night = ir_night.astype(np.float32)

        ir_day = Image.fromarray(ir_day)
        ir_night = Image.fromarray(ir_night)
        rgb_day = Image.fromarray(rgb_day)
        rgb_night = Image.fromarray(rgb_night)
        label_day = Image.fromarray(label_day, mode='L')


        # Random horizontal flipping
        if random.random() > 0.5:
            ir_day = F.hflip(ir_day)
            rgb_day = F.hflip(rgb_day)
            label_day = F.hflip(label_day)

        if random.random() > 0.5:
            ir_night = F.hflip(ir_night)
            rgb_night = F.hflip(rgb_night)

        # random rotation
        if random.random() > 0.5:
            angle = (random.random() - 0.5) * 40  # random angle in [-20, 20]
            ir_day = F.rotate(ir_day, angle, resample=Image.BILINEAR)
            rgb_day = F.rotate(rgb_day, angle, resample=Image.BILINEAR)
            # label_day = F.rotate(label_day, angle, resample=Image.NEAREST, fill=12)
            label_day = F.rotate(label_day, angle, resample=Image.NEAREST)

        if random.random() > 0.5:
            angle = (random.random() - 0.5) * 40  # random angle in [-20, 20]
            ir_night = F.rotate(ir_night, angle, resample=Image.BILINEAR)
            rgb_night = F.rotate(rgb_night, angle, resample=Image.BILINEAR)

        # To Tensor
        label_day = np.array(label_day).astype(np.uint8)


        rgb_day = F.to_tensor(rgb_day)
        rgb_night = F.to_tensor(rgb_night)
        label_day = torch.from_numpy(label_day)
        ir_day = F.to_tensor(ir_day)
        ir_night = F.to_tensor(ir_night)


        # Normalization

        if self.db_stats:
            rgb_day = F.normalize(rgb_day, **self.db_stats)
            rgb_night = F.normalize(rgb_night, **self.db_stats)
        else:
            rgb_day = F.normalize(rgb_day, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            rgb_night = F.normalize(rgb_night, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

        ir_day = F.normalize(ir_day, mean=[0.5], std=[0.5])
        ir_night = F.normalize(ir_night, mean=[0.5], std=[0.5])

        out_dict = {}
        out_dict['rgb_day'] = rgb_day
        out_dict['label_day'] = label_day
        out_dict['rgb_night'] = rgb_night
        out_dict['ir_day'] = ir_day
        out_dict['ir_night'] = ir_night
        out_dict['sun_altitude_day'] = sun_altitude_day
        out_dict['sun_altitude_night'] = sun_altitude_night
        out_dict['mod_drop_params'] = mod_drop_params

        return out_dict

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

class ThermalTestDataLoader(data.Dataset):
        def __init__(self, ir_paths, rgb_paths, label_paths, normalize=True, db_stats=None):

            self.ir_files = ir_paths
            self.rgb_files = rgb_paths
            self.label_files = label_paths

            print('Number of IR image files: ', len(self.ir_files))
            print('Number of RGB image files: ', len(self.rgb_files))
            assert (len(self.ir_files) == len(self.rgb_files))
            assert (len(self.rgb_files) == len(self.label_files))

            self.length = len(self.rgb_files)

            # normalize IR data (is in range 0, 2**16 --> crop to relevant range(20800, 27000))
            self.minval = 21800
            self.maxval = 25000
            self.normalize = normalize
            self.db_stats = db_stats

        def __getitem__(self, index):
            rgb_file = self.rgb_files[index]
            ir_file = self.ir_files[index]
            label_file = self.label_files[index]

            # READ
            rgb_im = cv2.imread(rgb_file)
            ir_im = cv2.imread(ir_file, cv2.IMREAD_ANYDEPTH)
            label = np.load(label_file)

            # COLORS
            rgb_im = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2RGB)

            # resizing
            res = (960, 320)
            rgb_im = cv2.resize(rgb_im, res, interpolation=cv2.INTER_LINEAR)
            ir_im = cv2.resize(ir_im, res, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, res, interpolation=cv2.INTER_NEAREST)

            rgb_im = rgb_im[:, 148:852, :]
            ir_im = ir_im[:, 148:852]
            label = label[:, 148:852]

            ir_im[ir_im < self.minval] = self.minval
            ir_im[ir_im > self.maxval] = self.maxval
            ir_im = (ir_im - self.minval) / (self.maxval - self.minval)

            ir_im = torch.from_numpy(ir_im.astype(np.float32)).unsqueeze(0)
            ir_im_org = ir_im.clone()

            rgb_im = torch.from_numpy(rgb_im).permute(2, 0, 1).float() / 255.
            rgb_im_org = rgb_im.clone()
            label = torch.from_numpy(label)

            if self.normalize:
                if self.db_stats:
                    rgb_im = F.normalize(rgb_im, **self.db_stats)
                else:
                    rgb_im = F.normalize(rgb_im, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

                ir_im = F.normalize(ir_im, mean=[0.5], std=[0.5])

            out_dict = {}
            out_dict['rgb'] = rgb_im
            out_dict['rgb_org'] = rgb_im_org
            out_dict['label'] = label
            out_dict['ir'] = ir_im
            out_dict['ir_org'] = ir_im_org
            return out_dict

        def __len__(self):
            return self.length

