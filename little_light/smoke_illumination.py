import os
import sys
import tempfile
import types

import numpy as np
import torch
import torch.nn as nn

from dataloader import Load_DataSet_X4K

if 'timm.models.layers' not in sys.modules:
    timm_module = types.ModuleType('timm')
    timm_models_module = types.ModuleType('timm.models')
    timm_layers_module = types.ModuleType('timm.models.layers')

    class DropPath(nn.Identity):
        pass

    def to_2tuple(value):
        return value if isinstance(value, tuple) else (value, value)

    def trunc_normal_(tensor, std=.02):
        return nn.init.trunc_normal_(tensor, std=std)

    timm_layers_module.DropPath = DropPath
    timm_layers_module.to_2tuple = to_2tuple
    timm_layers_module.trunc_normal_ = trunc_normal_
    sys.modules['timm'] = timm_module
    sys.modules['timm.models'] = timm_models_module
    sys.modules['timm.models.layers'] = timm_layers_module

from models.SwinSF_1000 import LASwinSFLite, SwinSpikeFormer


class TinyX4KDataset(Load_DataSet_X4K):
    def get_np_from_dat(self, file_path):
        return np.zeros((41, 20, 20), dtype=np.uint8)

    def get_gt(self, png_path, is_resize):
        return torch.zeros(1, 20, 20, dtype=torch.float32)


def make_tiny_tree(root, etas):
    for split in ['train', 'test']:
        input_dir = os.path.join(root, split, 'input')
        gt_dir = os.path.join(root, split, 'gt')
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)
        for eta in etas:
            data_name = 'lambda{}_occ008.649_f4305'.format(eta)
            open(os.path.join(input_dir, data_name + '.dat'), 'wb').close()
            for key_id in [7, 14, 21, 18, 35]:
                open(os.path.join(gt_dir, data_name + '_key_id{}.png'.format(key_id)), 'wb').close()


def build_tiny_backbone():
    return SwinSpikeFormer(img_size=(20, 20), patch_size=4, in_chans=41, ref_ch=28, out_chans=1,
                           embed_dim=8, depths=[1], num_heads=[1], window_size=5,
                           mlp_ratio=2., qkv_bias=True, qk_scale=None,
                           drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                           norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                           use_checkpoint=False, upscale=1, img_range=1.,
                           upsampler='', resi_connection='1conv')


def build_tiny_lite_model():
    return LASwinSFLite(backbone=build_tiny_backbone(), descriptor_dim=16,
                        use_light_code=True, use_ldf_lite=True,
                        use_lsa_lite=True)


def memory_text():
    if not torch.cuda.is_available():
        return 'gpu_mem allocated/reserved: n/a'
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    return 'gpu_mem allocated/reserved: {:.1f}MB/{:.1f}MB'.format(allocated, reserved)


def main():
    etas = [0.1, 0.3, 0.5, 0.7, 1.0, 2.0]
    with tempfile.TemporaryDirectory() as tmp_dir:
        make_tiny_tree(tmp_dir, etas)
        dataset = TinyX4KDataset(dataset_path=tmp_dir, mode='train',
                                 dataset_mode='grouped', etas=etas,
                                 return_meta=True)
        spikes_group, gt, eta_tensor, meta = dataset[0]
        assert spikes_group.shape == (6, 41, 20, 20)
        assert gt.shape == (1, 20, 20)
        assert eta_tensor.shape == (6,)
        assert meta['group_key'] == 'occ008.649_f4305'
        print('grouped dataset shape ok:', spikes_group.shape, gt.shape, eta_tensor.shape)

    model = build_tiny_lite_model()
    spikes = torch.randn(1, 6, 41, 20, 20)
    etas_tensor = torch.tensor([etas], dtype=torch.float32)
    target_index = torch.tensor([0], dtype=torch.long)
    gt = torch.zeros(1, 1, 20, 20)
    pred_mid, _, _ = model(spikes, etas_tensor, target_index)
    assert pred_mid.shape == (1, 1, 20, 20)
    loss = nn.SmoothL1Loss(beta=1.0)(pred_mid, gt)
    loss.backward()
    print('K=6 target-only forward/backward ok:', pred_mid.shape,
          'loss', float(loss.detach()), memory_text())

    pred_single, _, _ = model(spikes[:, 0], etas_tensor[:, 0], 0)
    assert pred_single.shape == (1, 1, 20, 20)
    print('K=1 single-light forward ok:', pred_single.shape, memory_text())

    baseline = build_tiny_backbone()
    baseline_pred, _, _ = baseline(torch.randn(1, 41, 20, 20))
    assert baseline_pred.shape == (1, 1, 20, 20)
    print('baseline forward ok:', baseline_pred.shape)
    print('descriptor path is [B,K,D]; backbone saw target [B,T,H,W], not [B*K,T,H,W].')


if __name__ == '__main__':
    main()
