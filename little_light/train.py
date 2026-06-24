import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from tqdm import tqdm

from dataloader import Load_DataSet_Reds, Load_DataSet_X4K

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            print('tensorboard is not installed; scalar logs will be printed only.')

        def add_scalar(self, *args, **kwargs):
            return None


def set_parser():
    parser = argparse.ArgumentParser(description='SwinSpikeFormer')
    parser.add_argument('--data_mode', type=str, choices=['250', '1000'], help='the resolution of the datasets')
    parser.add_argument('--dataset_path', default='./datasets/spike_x4k', help='the dataset for training and testing')
    parser.add_argument('--device', default='cuda:0', help='the gpu device used in training')
    parser.add_argument('--device_ids', type=str, default='', help='comma-separated gpu ids')
    parser.add_argument('--load_model', type=str, default='', help='the model saved in training and loaded in testing')
    parser.add_argument('--epochs', type=int, default=900, metavar='N', help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='the momentum for SGD optimizer')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='the weight decay for optimizer')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size of trainset')
    parser.add_argument('--resume_path', type=str, default='', help='Path for resume model.')
    parser.add_argument('--light', type=str, default='', help='legacy lambda/light value for X4K data')

    parser.add_argument('--model', type=str, choices=['swinsf', 'la_swinsf_lite'], default='swinsf')
    parser.add_argument('--dataset_mode', type=str, choices=['legacy', 'single', 'grouped'], default='legacy')
    parser.add_argument('--etas', nargs='*', type=float, default=None)
    parser.add_argument('--sample_k', type=int, default=None)
    parser.add_argument('--random_light_subset', action='store_true')
    parser.add_argument('--use_light_code', action='store_true')
    parser.add_argument('--use_ldf_lite', action='store_true')
    parser.add_argument('--use_lsa_lite', action='store_true')
    parser.add_argument('--target_eta', type=float, default=None)
    parser.add_argument('--random_target_eta', action='store_true')
    parser.add_argument('--descriptor_dim', type=int, default=64)
    parser.add_argument('--conversion_rate', type=float, default=0.6)
    parser.add_argument('--lambda_cons', type=float, default=0.0)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--amp', action='store_true')
    return parser.parse_args()


def get_device(args):
    train_on_gpu = torch.cuda.is_available()
    print('cuda is available, training on gpu!' if train_on_gpu else 'Training on cpu!')
    device = torch.device(args.device if train_on_gpu else 'cpu')
    if args.device_ids:
        device_ids = [int(part) for part in args.device_ids.replace(',', ' ').split()]
    elif train_on_gpu and ':' in args.device:
        device_ids = [int(args.device.split(':')[-1])]
    else:
        device_ids = [0]
    print('device_ids:', device_ids)
    return device, device_ids


def build_base_model(args):
    if args.data_mode == '250':
        from models.SwinSF_250 import SwinSpikeFormer
        return SwinSpikeFormer(img_size=(250, 400), patch_size=2, in_chans=41, ref_ch=28, out_chans=1,
                               embed_dim=96, depths=[6, 6], num_heads=[2, 2],
                               window_size=5, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                               drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                               norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                               use_checkpoint=False, upscale=1, img_range=1., upsampler='',
                               resi_connection='1conv')

    from models.SwinSF_1000 import SwinSpikeFormer
    return SwinSpikeFormer(img_size=(1000, 1000), patch_size=4, in_chans=41, ref_ch=28, out_chans=1,
                           embed_dim=64, depths=[6, 6], num_heads=[2, 2],
                           window_size=5, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                           drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                           norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                           use_checkpoint=False, upscale=2, img_range=1., upsampler='',
                           resi_connection='1conv')


def build_model(args):
    base_model = build_base_model(args)
    if args.model != 'la_swinsf_lite':
        return base_model
    if args.data_mode != '1000':
        raise ValueError('la_swinsf_lite is implemented for data_mode 1000')
    from models.SwinSF_1000 import LASwinSFLite
    return LASwinSFLite(
        backbone=base_model,
        descriptor_dim=args.descriptor_dim,
        use_light_code=args.use_light_code,
        use_ldf_lite=args.use_ldf_lite,
        use_lsa_lite=args.use_lsa_lite,
    )


def build_datasets(args):
    if args.data_mode == '250':
        return (
            Load_DataSet_Reds(dataset_path=args.dataset_path, mode='train'),
            Load_DataSet_Reds(dataset_path=args.dataset_path, mode='test'),
        )

    if args.dataset_mode == 'legacy':
        return (
            Load_DataSet_X4K(dataset_path=args.dataset_path, mode='train', light=args.light),
            Load_DataSet_X4K(dataset_path=args.dataset_path, mode='test', light=args.light),
        )

    return (
        Load_DataSet_X4K(dataset_path=args.dataset_path, mode='train', dataset_mode=args.dataset_mode,
                         etas=args.etas, sample_k=args.sample_k,
                         random_light_subset=args.random_light_subset,
                         return_meta=True),
        Load_DataSet_X4K(dataset_path=args.dataset_path, mode='test', dataset_mode=args.dataset_mode,
                         etas=args.etas, sample_k=args.sample_k,
                         random_light_subset=False, return_meta=True),
    )


def load_checkpoint_if_needed(net, args):
    if not args.load_model:
        return
    if not os.path.isfile(args.load_model):
        print("=> checkpoint '{}' not found".format(args.load_model))
        return
    print("=> loading checkpoint '{}'".format(args.load_model))
    checkpoint = torch.load(args.load_model, map_location='cpu')
    state = checkpoint['net'] if isinstance(checkpoint, dict) and 'net' in checkpoint else checkpoint
    try:
        net.load_state_dict(state)
    except RuntimeError as exc:
        module = net.module if hasattr(net, 'module') else net
        if hasattr(module, 'backbone'):
            mapped = {}
            for key, value in state.items():
                if key.startswith('module.backbone.') or key.startswith('backbone.'):
                    mapped[key] = value
                elif key.startswith('module.'):
                    mapped['module.backbone.' + key[len('module.'):]] = value
                else:
                    mapped['module.backbone.' + key] = value
            missing, unexpected = net.load_state_dict(mapped, strict=False)
            print('=> loaded baseline weights into LA-SwinSF-Lite backbone')
            print('missing keys:', len(missing), 'unexpected keys:', len(unexpected))
        else:
            raise exc
    print("=> loaded checkpoint '{}'".format(args.load_model))


def tensor_psnr(pred, target):
    pred = torch.clamp(pred, 0, 255)
    target = torch.clamp(target, 0, 255)
    mse = torch.mean((pred - target) ** 2, dim=tuple(range(1, pred.dim())))
    return 20.0 * torch.log10(torch.tensor(255.0, device=pred.device)) - 10.0 * torch.log10(mse + 1e-8)


def normalize_prediction_by_eta(pred, eta_target, args):
    if args.conversion_rate is None or args.conversion_rate <= 0:
        return pred
    scale = args.conversion_rate * torch.clamp(eta_target, min=1e-8)
    return pred / scale.view(-1, 1, 1, 1)


def select_target_index(etas, args):
    if etas.dim() == 1:
        etas_2d = etas.view(-1, 1)
    else:
        etas_2d = etas
    b, k = etas_2d.shape
    if k == 1:
        target_index = torch.zeros(b, dtype=torch.long, device=etas.device)
    elif args.random_target_eta:
        target_index = torch.randint(0, k, (b,), device=etas.device)
    elif args.target_eta is not None:
        target_value = torch.tensor(float(args.target_eta), device=etas.device, dtype=etas_2d.dtype)
        target_index = torch.argmin(torch.abs(etas_2d - target_value), dim=1)
    else:
        target_index = torch.zeros(b, dtype=torch.long, device=etas.device)
    eta_target = torch.gather(etas_2d, 1, target_index.view(b, 1)).squeeze(1)
    return target_index, eta_target


def gpu_memory_text(device):
    if device.type != 'cuda' or not torch.cuda.is_available():
        return 'gpu_mem allocated/reserved: n/a'
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    return 'gpu_mem allocated/reserved: {:.1f}MB/{:.1f}MB'.format(allocated, reserved)


def eta_unique_text(etas):
    return [float(value) for value in torch.unique(etas.detach().cpu()).tolist()]


def train_val_lite(net, train_loader, val_loader, device, args):
    save_path = os.path.join('checkpoint', datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))
    os.makedirs(save_path, exist_ok=True)
    writer = SummaryWriter(save_path)
    criterion = nn.SmoothL1Loss(reduction='mean', beta=1.0)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                 weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=300, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == 'cuda')
    best = {'epoch': 0, 'psnr': 0.0}

    for epoch in range(args.epochs):
        train_stats = run_lite_epoch(net, train_loader, device, criterion, args,
                                     optimizer=optimizer, scaler=scaler,
                                     split='train', epoch=epoch)
        val_stats = run_lite_epoch(net, val_loader, device, criterion, args,
                                   optimizer=None, scaler=None,
                                   split='test', epoch=epoch)
        scheduler.step()

        for split, stats in [('train', train_stats), ('test', val_stats)]:
            writer.add_scalar('{}/loss_total'.format(split), stats['loss_total'], epoch)
            writer.add_scalar('{}/loss_rec'.format(split), stats['loss_rec'], epoch)
            writer.add_scalar('{}/loss_cons'.format(split), stats['loss_cons'], epoch)
            writer.add_scalar('{}/psnr_target'.format(split), stats['psnr_target'], epoch)

        print('epoch : {}'.format(epoch))
        print('train target_eta {} eta_unique {} loss_rec {:.6f} loss_cons {:.6f} loss_total {:.6f} PSNR target {:.4f} {}'.format(
            train_stats['target_eta'], train_stats['eta_unique'], train_stats['loss_rec'],
            train_stats['loss_cons'], train_stats['loss_total'], train_stats['psnr_target'],
            train_stats['gpu_memory']))
        print('test  target_eta {} eta_unique {} loss_rec {:.6f} loss_cons {:.6f} loss_total {:.6f} PSNR target {:.4f} {}'.format(
            val_stats['target_eta'], val_stats['eta_unique'], val_stats['loss_rec'],
            val_stats['loss_cons'], val_stats['loss_total'], val_stats['psnr_target'],
            val_stats['gpu_memory']))

        if val_stats['psnr_target'] > best['psnr']:
            best = {'epoch': epoch, 'psnr': val_stats['psnr_target']}
            state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, os.path.join(save_path, 'best_model_psnr:{:.4f}_epoch:{}_.pth'.format(
                best['psnr'], best['epoch'])))
            print('save the best model!')
        print('now the best psnr is {:.4f}, appear in epoch {}'.format(best['psnr'], best['epoch']))


def run_lite_epoch(net, loader, device, criterion, args, optimizer, scaler, split, epoch):
    is_train = optimizer is not None
    net.train(is_train)
    losses_total, losses_rec, losses_cons, psnrs = [], [], [], []
    target_etas_seen = []
    eta_unique_seen = set()
    memory_snapshot = gpu_memory_text(device)

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in tqdm(iter(loader)):
            if args.dataset_mode == 'grouped':
                spikes, gt, etas, _ = batch
            else:
                spikes, gt, etas = batch[:3]
            spikes = spikes.to(device)
            gt = gt.to(device)
            etas = etas.to(device=device, dtype=torch.float32)
            target_index, eta_target = select_target_index(etas, args)
            target_etas_seen.extend([float(v) for v in eta_target.detach().cpu().tolist()])
            eta_unique_seen.update(eta_unique_text(etas))

            if is_train:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == 'cuda'):
                pred_raw, _, _ = net(spikes, etas, target_index)
                pred = torch.clamp(normalize_prediction_by_eta(pred_raw, eta_target, args), 0, 255)
                loss_rec = criterion(pred, gt)
                loss_cons = pred.new_tensor(0.0)
                if args.lambda_cons > 0 and spikes.dim() == 5 and spikes.shape[1] > 1:
                    ref_index = torch.argmax(etas if etas.dim() == 2 else etas.view(-1, 1), dim=1)
                    with torch.no_grad():
                        pred_ref_raw, _, _ = net(spikes, etas, ref_index)
                        eta_ref = torch.gather(etas, 1, ref_index.view(-1, 1)).squeeze(1) if etas.dim() == 2 else etas
                        pred_ref = torch.clamp(normalize_prediction_by_eta(pred_ref_raw, eta_ref, args), 0, 255)
                    loss_cons = criterion(pred, pred_ref)
                loss_total = loss_rec + args.lambda_cons * loss_cons

            if is_train:
                scaler.scale(loss_total).backward()
                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()

            losses_total.append(float(loss_total.detach().cpu()))
            losses_rec.append(float(loss_rec.detach().cpu()))
            losses_cons.append(float(loss_cons.detach().cpu()))
            psnrs.extend([float(v) for v in tensor_psnr(pred.detach(), gt).detach().cpu().tolist()])
            memory_snapshot = gpu_memory_text(device)

    return {
        'loss_total': float(np.mean(losses_total)) if losses_total else 0.0,
        'loss_rec': float(np.mean(losses_rec)) if losses_rec else 0.0,
        'loss_cons': float(np.mean(losses_cons)) if losses_cons else 0.0,
        'psnr_target': float(np.mean(psnrs)) if psnrs else 0.0,
        'target_eta': sorted(set([round(v, 8) for v in target_etas_seen])),
        'eta_unique': sorted(eta_unique_seen),
        'gpu_memory': memory_snapshot,
    }


def train_val_legacy(net, train_loader, val_loader, device, args):
    save_path = os.path.join('checkpoint', datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))
    os.makedirs(save_path, exist_ok=True)
    writer = SummaryWriter(save_path)
    criterion = nn.SmoothL1Loss(reduction='mean', beta=1.0)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                 weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=300, gamma=0.5)
    best = {'epoch': 0, 'psnr': 0.0}

    for epoch in range(args.epochs):
        train_loss, train_psnr = [], []
        net.train()
        for data_iter, _, ik_14, ik_21, ik_28, _, _ in tqdm(iter(train_loader)):
            data_iter = data_iter.to(device)
            ik_iter = ik_21.to(device)
            ik_l = ik_14.to(device)
            ik_r = ik_28.to(device)
            optimizer.zero_grad()
            img_pred, im_l, im_r = net(data_iter)
            img_pred = torch.clamp(img_pred, 0, 255)
            im_l = torch.clamp(im_l, 0, 255)
            im_r = torch.clamp(im_r, 0, 255)
            loss_value = criterion(ik_iter, img_pred) + 0.1 * (criterion(ik_l, im_l) + criterion(ik_r, im_r))
            loss_value.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()
            train_loss.append(loss_value.item())
            train_psnr.append(float(tensor_psnr(img_pred.detach(), ik_iter).mean().cpu()))

        val_loss, val_psnr = [], []
        net.eval()
        with torch.no_grad():
            for data_iter, ik_iter, _ in tqdm(iter(val_loader)):
                data_iter = data_iter.to(device)
                ik_iter = ik_iter.to(device)
                img_pred, _, _ = net(data_iter)
                img_pred = torch.clamp(img_pred, 0, 255)
                loss_value = criterion(ik_iter, img_pred)
                val_loss.append(loss_value.item())
                val_psnr.append(float(tensor_psnr(img_pred, ik_iter).mean().cpu()))

        scheduler.step()
        train_loss_mean = float(np.mean(train_loss)) if train_loss else 0.0
        val_loss_mean = float(np.mean(val_loss)) if val_loss else 0.0
        train_psnr_mean = float(np.mean(train_psnr)) if train_psnr else 0.0
        val_psnr_mean = float(np.mean(val_psnr)) if val_psnr else 0.0
        writer.add_scalar('train/loss', train_loss_mean, epoch)
        writer.add_scalar('train/psnr', train_psnr_mean, epoch)
        writer.add_scalar('test/loss', val_loss_mean, epoch)
        writer.add_scalar('test/psnr', val_psnr_mean, epoch)
        print('epoch : {} train_loss : {} train_psnr : {} test_loss : {} test_psnr : {}'.format(
            epoch, train_loss_mean, train_psnr_mean, val_loss_mean, val_psnr_mean))

        if val_psnr_mean > best['psnr']:
            best['epoch'] = epoch
            best['psnr'] = val_psnr_mean
            state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, os.path.join(save_path, 'best_model_psnr:{:.4f}_epoch:{}_.pth'.format(
                best['psnr'], best['epoch'])))
            print('save the best model!')
        print('now the best psnr is {}, appear in epoch {}'.format(best['psnr'], best['epoch']))


def main():
    args = set_parser()
    if args.model == 'la_swinsf_lite' and args.dataset_mode == 'legacy':
        args.dataset_mode = 'grouped'
    device, device_ids = get_device(args)
    net = build_model(args)
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net, device_ids=device_ids)
    print('the network is {}\n{}'.format(net, '-' * 120))
    load_checkpoint_if_needed(net, args)
    net.to(device)

    spike_train, spike_test = build_datasets(args)
    train_loader = data.DataLoader(dataset=spike_train, batch_size=args.batch_size, shuffle=True)
    test_loader = data.DataLoader(dataset=spike_test, batch_size=1, shuffle=False)

    if args.model == 'la_swinsf_lite':
        train_val_lite(net, train_loader, test_loader, device, args)
    else:
        train_val_legacy(net, train_loader, test_loader, device, args)


if __name__ == '__main__':
    main()
