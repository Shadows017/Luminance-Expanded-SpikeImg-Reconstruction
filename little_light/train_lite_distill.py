import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from tqdm import tqdm

from dataloader import Load_DataSet_X4K

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            print('tensorboard is not installed; scalar logs will be printed only.')

        def add_scalar(self, *args, **kwargs):
            return None


def set_parser():
    parser = argparse.ArgumentParser(description='LA-SwinSF-Lite single-light distillation')
    parser.add_argument('--dataset_path', default='./datasets/spike_x4k')
    parser.add_argument('--teacher_model', required=True, help='grouped teacher checkpoint')
    parser.add_argument('--student_init', default='', help='optional checkpoint to initialize student')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--etas', nargs='*', type=float, default=[0.1, 0.3, 0.5, 0.7, 1.0, 2.0])
    parser.add_argument('--target_eta', type=float, default=0.3)
    parser.add_argument('--random_target_eta', action='store_true')
    parser.add_argument('--descriptor_dim', type=int, default=64)
    parser.add_argument('--use_light_code', action='store_true')
    parser.add_argument('--use_ldf_lite', action='store_true')
    parser.add_argument('--use_lsa_lite', action='store_true')
    parser.add_argument('--conversion_rate', type=float, default=0.6)
    parser.add_argument('--lambda_distill', type=float, default=0.2)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--save_root', default='checkpoint')
    return parser.parse_args()


def build_model(args):
    from models.SwinSF_1000 import LASwinSFLite, SwinSpikeFormer

    backbone = SwinSpikeFormer(img_size=(1000, 1000), patch_size=4, in_chans=41, ref_ch=28, out_chans=1,
                               embed_dim=64, depths=[6, 6], num_heads=[2, 2],
                               window_size=5, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                               drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                               norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                               use_checkpoint=False, upscale=2, img_range=1.,
                               upsampler='', resi_connection='1conv')
    return LASwinSFLite(backbone=backbone,
                        descriptor_dim=args.descriptor_dim,
                        use_light_code=args.use_light_code,
                        use_ldf_lite=args.use_ldf_lite,
                        use_lsa_lite=args.use_lsa_lite)


def clean_state_dict(state):
    if isinstance(state, dict) and 'net' in state:
        state = state['net']
    cleaned = {}
    for key, value in state.items():
        if key.startswith('module.'):
            key = key[len('module.'):]
        cleaned[key] = value
    return cleaned


def load_model_weights(model, path, strict=True):
    checkpoint = torch.load(path, map_location='cpu')
    state = clean_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    print("loaded '{}'".format(path))
    if missing or unexpected:
        print('missing keys:', len(missing), 'unexpected keys:', len(unexpected))


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
    else:
        target_value = torch.tensor(float(args.target_eta), device=etas.device, dtype=etas_2d.dtype)
        target_index = torch.argmin(torch.abs(etas_2d - target_value), dim=1)
    eta_target = torch.gather(etas_2d, 1, target_index.view(b, 1)).squeeze(1)
    return target_index, eta_target


def gather_target_spikes(spikes, target_index):
    b, k, t, h, w = spikes.shape
    gather_idx = target_index.view(b, 1, 1, 1, 1).expand(-1, 1, t, h, w)
    return torch.gather(spikes, 1, gather_idx).squeeze(1)


def normalize_prediction_by_eta(pred, eta_target, args):
    if args.conversion_rate is None or args.conversion_rate <= 0:
        return pred
    scale = args.conversion_rate * torch.clamp(eta_target, min=1e-8)
    return pred / scale.view(-1, 1, 1, 1)


def tensor_psnr(pred, target):
    pred = torch.clamp(pred, 0, 255)
    target = torch.clamp(target, 0, 255)
    mse = torch.mean((pred - target) ** 2, dim=tuple(range(1, pred.dim())))
    return 20.0 * torch.log10(torch.tensor(255.0, device=pred.device)) - 10.0 * torch.log10(mse + 1e-8)


def gpu_memory_text(device):
    if device.type != 'cuda' or not torch.cuda.is_available():
        return 'gpu_mem allocated/reserved: n/a'
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    return 'gpu_mem allocated/reserved: {:.1f}MB/{:.1f}MB'.format(allocated, reserved)


def run_epoch(student, teacher, loader, device, criterion, optimizer, scaler, args, split):
    is_train = optimizer is not None
    student.train(is_train)
    teacher.eval()
    losses, rec_losses, distill_losses, psnrs = [], [], [], []
    target_etas = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for spikes, gt, etas, _ in tqdm(iter(loader)):
            spikes = spikes.to(device)
            gt = gt.to(device)
            etas = etas.to(device=device, dtype=torch.float32)
            target_index, eta_target = select_target_index(etas, args)
            target_spikes = gather_target_spikes(spikes, target_index)
            target_etas.extend([float(v) for v in eta_target.detach().cpu().tolist()])

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp and device.type == 'cuda'):
                with torch.no_grad():
                    teacher_raw, _, _ = teacher(spikes, etas, target_index)
                    teacher_pred = torch.clamp(normalize_prediction_by_eta(teacher_raw, eta_target, args), 0, 255)

                student_raw, _, _ = student(target_spikes, eta_target, None)
                student_pred = torch.clamp(normalize_prediction_by_eta(student_raw, eta_target, args), 0, 255)
                loss_rec = criterion(student_pred, gt)
                loss_distill = criterion(student_pred, teacher_pred.detach())
                loss = loss_rec + args.lambda_distill * loss_distill

            if is_train:
                scaler.scale(loss).backward()
                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()

            losses.append(float(loss.detach().cpu()))
            rec_losses.append(float(loss_rec.detach().cpu()))
            distill_losses.append(float(loss_distill.detach().cpu()))
            psnrs.extend([float(v) for v in tensor_psnr(student_pred.detach(), gt).detach().cpu().tolist()])

    return {
        'loss': float(np.mean(losses)) if losses else 0.0,
        'loss_rec': float(np.mean(rec_losses)) if rec_losses else 0.0,
        'loss_distill': float(np.mean(distill_losses)) if distill_losses else 0.0,
        'psnr': float(np.mean(psnrs)) if psnrs else 0.0,
        'target_eta': sorted(set([round(v, 8) for v in target_etas])),
        'gpu_memory': gpu_memory_text(device),
        'split': split,
    }


def main():
    args = set_parser()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    train_set = Load_DataSet_X4K(dataset_path=args.dataset_path, mode='train',
                                 dataset_mode='grouped', etas=args.etas,
                                 return_meta=True)
    val_set = Load_DataSet_X4K(dataset_path=args.dataset_path, mode='test',
                               dataset_mode='grouped', etas=args.etas,
                               return_meta=True)
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = data.DataLoader(val_set, batch_size=1, shuffle=False)

    teacher = build_model(args).to(device)
    load_model_weights(teacher, args.teacher_model, strict=True)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    student = build_model(args).to(device)
    if args.student_init:
        load_model_weights(student, args.student_init, strict=False)
    else:
        load_model_weights(student, args.teacher_model, strict=False)

    criterion = nn.SmoothL1Loss(reduction='mean', beta=1.0)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                 weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=300, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == 'cuda')

    save_path = os.path.join(args.save_root, 'distill_' + datetime.now().strftime('%Y-%m-%d_%H_%M_%S'))
    os.makedirs(save_path, exist_ok=True)
    writer = SummaryWriter(save_path)
    best_psnr = 0.0

    for epoch in range(args.epochs):
        train_stats = run_epoch(student, teacher, train_loader, device, criterion,
                                optimizer, scaler, args, 'train')
        val_stats = run_epoch(student, teacher, val_loader, device, criterion,
                              None, None, args, 'test')
        scheduler.step()

        writer.add_scalar('train/loss', train_stats['loss'], epoch)
        writer.add_scalar('train/loss_rec', train_stats['loss_rec'], epoch)
        writer.add_scalar('train/loss_distill', train_stats['loss_distill'], epoch)
        writer.add_scalar('train/psnr_single', train_stats['psnr'], epoch)
        writer.add_scalar('test/loss', val_stats['loss'], epoch)
        writer.add_scalar('test/loss_rec', val_stats['loss_rec'], epoch)
        writer.add_scalar('test/loss_distill', val_stats['loss_distill'], epoch)
        writer.add_scalar('test/psnr_single', val_stats['psnr'], epoch)

        print('epoch {}'.format(epoch))
        print('train eta {} loss {:.6f} rec {:.6f} distill {:.6f} single_psnr {:.4f} {}'.format(
            train_stats['target_eta'], train_stats['loss'], train_stats['loss_rec'],
            train_stats['loss_distill'], train_stats['psnr'], train_stats['gpu_memory']))
        print('test  eta {} loss {:.6f} rec {:.6f} distill {:.6f} single_psnr {:.4f} {}'.format(
            val_stats['target_eta'], val_stats['loss'], val_stats['loss_rec'],
            val_stats['loss_distill'], val_stats['psnr'], val_stats['gpu_memory']))

        if val_stats['psnr'] > best_psnr:
            best_psnr = val_stats['psnr']
            state = {'net': student.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, os.path.join(save_path, 'best_student_psnr:{:.4f}_epoch:{}_.pth'.format(best_psnr, epoch)))
            print('save best single-light student!')


if __name__ == '__main__':
    main()

