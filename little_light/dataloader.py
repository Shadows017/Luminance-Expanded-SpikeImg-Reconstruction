import numpy as np
from torchvision import transforms as T
import torch
import os
import random
import re
from torch.utils.data import Dataset, DataLoader
try:
    import cv2
except ImportError:
    cv2 = None
from PIL import Image


def normalize_eta_list(etas):
    if etas is None or etas == '':
        return None
    if isinstance(etas, str):
        etas = [part for part in re.split(r'[\s,]+', etas.strip()) if part]
    if isinstance(etas, (int, float)):
        etas = [etas]
    return [float(eta) for eta in etas]


def eta_key(eta):
    return round(float(eta), 8)


def parse_sample_key(path):
    """Parse eta and content identity from X4K spike/GT filenames.

    Supported examples:
        lambda2.0_occ008.649_f4305.dat
        lambda2.0_occ008.649_f4305_key_id7.png
        000_part1_key_id21.dat
    """
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    key_match = re.search(r'_key_id(?P<key_id>\d+)$', stem)
    key_frame = key_match.group('key_id') if key_match else ''
    base = re.sub(r'_key_id\d+$', '', stem)

    eta = None
    eta_text = ''
    content_key = base
    light_match = re.match(r'^lambda(?P<eta>\d+(?:\.\d+)?)(?:_|$)(?P<body>.*)$', base)
    if light_match:
        eta_text = light_match.group('eta')
        eta = float(eta_text)
        content_key = light_match.group('body') or base

    parts = [part for part in content_key.split('_') if part]
    scene_id = parts[0] if parts else content_key
    time_id = ''
    patch_id = ''
    for part in parts[1:]:
        if re.match(r'^[fFtT]?\d+$', part) or re.match(r'^[fF]\d+', part):
            time_id = part
        elif re.search(r'(patch|crop|x\d+|y\d+)', part, re.IGNORECASE):
            patch_id = part
    if not time_id and len(parts) > 1:
        time_id = parts[-1]
    if not patch_id and len(parts) > 2:
        patch_id = '_'.join(parts[1:-1])

    return {
        'name': name,
        'stem': stem,
        'data_name': base,
        'content_key': content_key,
        'scene_id': scene_id,
        'time_id': time_id,
        'patch_id': patch_id,
        'key_frame': key_frame,
        'eta': eta,
        'eta_text': eta_text,
    }

class Load_DataSet_Reds(Dataset):
    def __init__(self , dataset_path , mode=None):
        self.dataset_path = dataset_path
        self.mode = mode
        self.filename_list = os.listdir(os.path.join(self.dataset_path,self.mode,'input'))

    def __getitem__(self, index):
        if self.mode == 'train':
            data_index = self.filename_list[index]
            dat_path = os.path.join(self.dataset_path,self.mode,'input',data_index)
            data  = self.get_np_from_dat(dat_path)
            data = torch.from_numpy(data.copy()).type(torch.FloatTensor)
            data_name = data_index.split('.')[0].split('id')[0]
            ik_7 = self.get_gt(os.path.join(self.dataset_path,self.mode,'gt',data_name + 'id' + str(7) + '.png'),False)
            ik_14 = self.get_gt(os.path.join(self.dataset_path,self.mode,'gt',data_name + 'id' + str(14) + '.png'),False)
            ik_21 = self.get_gt(os.path.join(self.dataset_path,self.mode,'gt',data_name + 'id' + str(21) + '.png'),False)
            ik_28 = self.get_gt(os.path.join(self.dataset_path,self.mode,'gt',data_name + 'id' + str(28) + '.png'),False)
            ik_35 = self.get_gt(os.path.join(self.dataset_path,self.mode,'gt',data_name + 'id' + str(35) + '.png'),False)
            
            return data,ik_7,ik_14,ik_21,ik_28,ik_35,data_name
        elif self.mode == 'test':
            data_index = self.filename_list[index]
            dat_path = os.path.join(self.dataset_path,self.mode,'input',data_index)
            data_name = data_index.split('.')[0]
            gt_path = os.path.join(self.dataset_path,self.mode,'gt',data_name+'.png')
            #print(dat_path)
            data  = self.get_np_from_dat(dat_path)
            data = torch.from_numpy(data.copy()).type(torch.FloatTensor)
            data = data[151 - 21 : 151 + 20, :, :]
            gt = self.get_gt(gt_path,False)
            return data,gt,data_name
    
    def __len__(self):
        return len(self.filename_list)

    def get_np_from_dat(self,file_path):
        f =  open(file_path,'rb')
        video_seq = f.read()
        video_seq = np.frombuffer(video_seq, 'b')
        spike_matrix = self.RawToSpike(video_seq,250,400,True)
        return spike_matrix

    def RawToSpike(self,video_seq, h, w,flipud):
        video_seq = np.array(video_seq).astype(np.uint8)
        img_size = h*w
        img_num = len(video_seq)//(img_size//8)
        SpikeMatrix = np.zeros([img_num, h, w], np.uint8)
        pix_id = np.arange(0,h*w)
        pix_id = np.reshape(pix_id, (h, w))
        comparator = np.left_shift(1, np.mod(pix_id, 8))
        byte_id = pix_id // 8

        for img_id in np.arange(img_num):
            id_start = img_id*img_size//8
            id_end = id_start + img_size//8
            cur_info = video_seq[id_start:id_end]
            data = cur_info[byte_id]
            result = np.bitwise_and(data, comparator)
            if flipud:
                SpikeMatrix[img_id, :, :] = np.flipud((result == comparator))
            else:
                SpikeMatrix[img_id, :, :] = (result == comparator)

        return SpikeMatrix

    def get_gt(self,png_path,is_resize):
        if is_resize:
            w,h = 400,250
            img = np.array(Image.open(png_path).convert('L').resize((w, h),Image.ANTIALIAS))
        else:
            img = np.array(Image.open(png_path).convert('L'))
        return torch.from_numpy(img.copy()).type(torch.FloatTensor).unsqueeze(0)



class Load_DataSet_classA(Dataset):
    def __init__(self , dataset_path , mode=None):
        self.dataset_path = dataset_path
        self.mode = mode
        self.filename_list = os.listdir(os.path.join(self.dataset_path,'test'))

    def __getitem__(self, index):
        data_index = self.filename_list[index]
        dat_path = os.path.join(self.dataset_path,'test',data_index)
        #print(dat_path)
        data  = self.get_np_from_dat(dat_path)
        data = torch.from_numpy(data.copy()).type(torch.FloatTensor)
        data_name = data_index.split('.')[0]
        return data,data_name
    
    def __len__(self):
        return len(self.filename_list)

    def get_np_from_dat(self,file_path):
        f =  open(file_path,'rb')
        video_seq = f.read()
        video_seq = np.frombuffer(video_seq, 'b')
        spike_matrix = self.RawToSpike(video_seq,250,400,True)
        return spike_matrix

    def RawToSpike(self,video_seq, h, w,flipud):
        video_seq = np.array(video_seq).astype(np.uint8)
        img_size = h*w
        img_num = len(video_seq)//(img_size//8)
        SpikeMatrix = np.zeros([img_num, h, w], np.uint8)
        pix_id = np.arange(0,h*w)
        pix_id = np.reshape(pix_id, (h, w))
        comparator = np.left_shift(1, np.mod(pix_id, 8))
        byte_id = pix_id // 8

        for img_id in np.arange(img_num):
            id_start = img_id*img_size//8
            id_end = id_start + img_size//8
            cur_info = video_seq[id_start:id_end]
            data = cur_info[byte_id]
            result = np.bitwise_and(data, comparator)
            if flipud:
                SpikeMatrix[img_id, :, :] = np.flipud((result == comparator))
            else:
                SpikeMatrix[img_id, :, :] = (result == comparator)

        return SpikeMatrix

class Load_DataSet_X4K(Dataset):
    def __init__(self, dataset_path, mode=None, light=None, dataset_mode='legacy',
                 etas=None, sample_k=None, random_light_subset=False,
                 return_meta=False, min_lights=1):
        self.dataset_path = dataset_path
        self.mode = mode
        self.dataset_mode = dataset_mode
        self.light = self.normalize_light(light)
        self.etas = normalize_eta_list(etas)
        if self.etas is None and self.light:
            self.etas = [float(self.light)]
        self.sample_k = sample_k
        self.random_light_subset = random_light_subset
        self.return_meta = return_meta
        self.min_lights = max(1, int(min_lights))
        self.input_dir = os.path.join(self.dataset_path, self.mode, 'input')
        self.gt_dir = os.path.join(self.dataset_path, self.mode, 'gt')
        if self.dataset_mode not in ['legacy', 'single', 'grouped']:
            raise ValueError('dataset_mode must be legacy, single, or grouped')
        if self.dataset_mode == 'grouped':
            self.groups = self.build_groups()
            self.group_keys = sorted(self.groups.keys())
        else:
            self.filename_list = self.get_filename_list()

    def __getitem__(self, index):
        if self.dataset_mode == 'legacy':
            return self.get_legacy_item(index)
        if self.dataset_mode == 'single':
            return self.get_single_item(index)
        if self.dataset_mode == 'grouped':
            return self.get_grouped_item(index)
        raise ValueError('Unsupported dataset_mode {}'.format(self.dataset_mode))

    def __len__(self):
        if self.dataset_mode == 'grouped':
            return len(self.group_keys)
        return len(self.filename_list)

    def get_legacy_item(self, index):
        data_index = self.filename_list[index]
        dat_path = os.path.join(self.input_dir, data_index)
        data = self.get_np_from_dat(dat_path)
        data = torch.from_numpy(data.copy()).type(torch.FloatTensor)
        data_name = self.get_data_name(data_index)
        if self.mode == 'train':
            ik_7 = self.get_gt(self.get_x4k_gt_path(data_name, [7]), False)
            ik_14 = self.get_gt(self.get_x4k_gt_path(data_name, [14]), False)
            ik_21 = self.get_gt(self.get_x4k_gt_path(data_name, [21]), False)
            ik_28 = self.get_gt(self.get_x4k_gt_path(data_name, [18, 28]), False)
            ik_35 = self.get_gt(self.get_x4k_gt_path(data_name, [35]), False)
            return data, ik_7, ik_14, ik_21, ik_28, ik_35, data_index
        gt = self.get_gt(self.get_x4k_gt_path(data_name, [21, 14]), False)
        return data, gt, data_index

    def get_single_item(self, index):
        data_index = self.filename_list[index]
        dat_path = os.path.join(self.input_dir, data_index)
        data = self.get_np_from_dat(dat_path)
        data = torch.from_numpy(data.copy()).type(torch.FloatTensor)
        data_name = self.get_data_name(data_index)
        gt = self.get_gt(self.get_x4k_gt_path(data_name, [21, 14, 18, 28, 7, 35]), False)
        meta = self.make_meta(data_index)
        eta = meta['eta'] if meta['eta'] != '' else 1.0
        eta = torch.tensor(float(eta), dtype=torch.float32)
        if self.return_meta:
            return data, gt, eta, meta
        return data, gt, eta

    def get_grouped_item(self, index):
        group_key = self.group_keys[index]
        group = self.groups[group_key]
        members = self.select_group_members(group)
        spikes = []
        etas = []
        data_names = []
        for eta_value, data_index in members:
            dat_path = os.path.join(self.input_dir, data_index)
            data = self.get_np_from_dat(dat_path)
            spikes.append(torch.from_numpy(data.copy()).type(torch.FloatTensor))
            etas.append(float(eta_value))
            data_names.append(self.get_data_name(data_index))
        gt = self.get_gt(self.get_x4k_gt_path(data_names[0], [21, 14, 18, 28, 7, 35]), False)
        meta = dict(group['meta'])
        meta['group_key'] = group_key
        meta['data_names'] = '|'.join(data_names)
        meta['etas'] = ','.join(['{:.8g}'.format(eta) for eta in etas])
        spikes = torch.stack(spikes, dim=0)
        etas = torch.tensor(etas, dtype=torch.float32)
        return spikes, gt, etas, meta

    def get_filename_list(self):
        filename_list = sorted([name for name in os.listdir(self.input_dir) if name.lower().endswith('.dat')])
        if self.dataset_mode == 'legacy' and self.light:
            filename_list = [name for name in filename_list if self.match_light(name)]
            if not filename_list:
                raise ValueError('No X4K input files found with prefix lambda{} in {}'.format(self.light, self.input_dir))
        if self.dataset_mode == 'single' and self.etas is not None:
            wanted = set([eta_key(eta) for eta in self.etas])
            filename_list = [name for name in filename_list if self.get_eta_value(name) in wanted]
            if not filename_list:
                raise ValueError('No X4K input files found for etas {} in {}'.format(self.etas, self.input_dir))
        return filename_list

    def build_groups(self):
        groups = {}
        wanted = None if self.etas is None else set([eta_key(eta) for eta in self.etas])
        for data_index in sorted([name for name in os.listdir(self.input_dir) if name.lower().endswith('.dat')]):
            parsed = parse_sample_key(data_index)
            eta_value = parsed['eta'] if parsed['eta'] is not None else 1.0
            eta_value = eta_key(eta_value)
            if wanted is not None and eta_value not in wanted:
                continue
            group_key = parsed['content_key']
            if group_key not in groups:
                meta = self.sanitize_meta(parsed)
                groups[group_key] = {'members': {}, 'meta': meta}
            groups[group_key]['members'][eta_value] = data_index

        valid_groups = {}
        for group_key, group in groups.items():
            if wanted is not None:
                if all(eta in group['members'] for eta in wanted):
                    valid_groups[group_key] = group
            elif len(group['members']) > 0:
                valid_groups[group_key] = group
        if not valid_groups:
            raise ValueError('No grouped X4K samples found for etas {} in {}'.format(self.etas, self.input_dir))
        return valid_groups

    def select_group_members(self, group):
        if self.etas is not None:
            eta_values = [eta_key(eta) for eta in self.etas]
        else:
            eta_values = sorted(group['members'].keys())

        max_k = len(eta_values)
        if self.sample_k is not None:
            max_k = min(max_k, int(self.sample_k))
        if self.random_light_subset:
            min_k = min(self.min_lights, max_k)
            k = random.randint(min_k, max_k)
            eta_values = sorted(random.sample(eta_values, k))
        elif self.sample_k is not None:
            eta_values = eta_values[:max_k]

        return [(eta_value, group['members'][eta_value]) for eta_value in eta_values]

    def normalize_light(self, light):
        if light is None:
            return ''
        light = str(light).strip()
        if light.startswith('lambda'):
            light = light[len('lambda'):]
        return light

    def match_light(self, data_index):
        parsed_eta = self.get_eta_value(data_index)
        if parsed_eta is None:
            return False
        return parsed_eta == eta_key(float(self.light))

    def get_eta_value(self, data_index):
        parsed = parse_sample_key(data_index)
        if parsed['eta'] is None:
            return None
        return eta_key(parsed['eta'])

    def get_data_name(self, data_index):
        data_name = os.path.splitext(os.path.basename(data_index))[0]
        if '_key_id' in data_name:
            data_name = data_name.split('_key_id')[0]
        return data_name

    def make_meta(self, data_index):
        meta = self.sanitize_meta(parse_sample_key(data_index))
        meta['data_index'] = data_index
        return meta

    def sanitize_meta(self, meta):
        clean = {}
        for key, value in meta.items():
            if value is None:
                clean[key] = ''
            elif isinstance(value, float):
                clean[key] = float(value)
            else:
                clean[key] = str(value)
        return clean

    def get_x4k_gt_path(self, data_name, ids):
        candidates = [data_name]
        parsed = parse_sample_key(data_name)
        if parsed['content_key'] not in candidates:
            candidates.append(parsed['content_key'])
        for base_name in candidates:
            for each_id in ids:
                gt_path = os.path.join(self.gt_dir, base_name + '_key_id' + str(each_id) + '.png')
                if os.path.exists(gt_path):
                    return gt_path
        return os.path.join(self.gt_dir, candidates[0] + '_key_id' + str(ids[0]) + '.png')

    def get_np_from_dat(self, file_path):
        with open(file_path, 'rb') as f:
            video_seq = f.read()
        video_seq = np.frombuffer(video_seq, 'b')
        spike_matrix = self.RawToSpike(video_seq, 1000, 1000, False)
        return spike_matrix

    def RawToSpike(self, video_seq, h, w, flipud):
        video_seq = np.array(video_seq).astype(np.uint8)
        img_size = h*w
        img_num = len(video_seq)//(img_size//8)
        SpikeMatrix = np.zeros([img_num, h, w], np.uint8)
        pix_id = np.arange(0, h*w)
        pix_id = np.reshape(pix_id, (h, w))
        comparator = np.left_shift(1, np.mod(pix_id, 8))
        byte_id = pix_id // 8

        for img_id in np.arange(img_num):
            id_start = img_id*img_size//8
            id_end = id_start + img_size//8
            cur_info = video_seq[id_start:id_end]
            data = cur_info[byte_id]
            result = np.bitwise_and(data, comparator)
            if flipud:
                SpikeMatrix[img_id, :, :] = np.flipud((result == comparator))
            else:
                SpikeMatrix[img_id, :, :] = (result == comparator)

        return SpikeMatrix

    def get_gt(self, png_path, is_resize):
        img = np.array(Image.open(png_path).convert('L'))
        return torch.from_numpy(img.copy()).type(torch.FloatTensor).unsqueeze(0)
