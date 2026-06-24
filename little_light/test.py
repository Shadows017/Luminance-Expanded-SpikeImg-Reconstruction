import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from tqdm import tqdm

try:
    import cv2
except ImportError:
    cv2 = None

from dataloader import Load_DataSet_Reds, Load_DataSet_X4K, Load_DataSet_classA

try:
    from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
    from skimage.metrics import structural_similarity as skimage_ssim
except ImportError:
    skimage_psnr = None
    skimage_ssim = None


SEEN_ETAS = [0.1, 0.3, 0.5, 0.7, 1.0, 2.0]
UNSEEN_ETAS = [0.2, 0.6, 1.2, 1.5]


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in ["true", "1", "yes", "y"]


def set_parser():
    parser = argparse.ArgumentParser(description="SwinSF / LA-SwinSF-Lite Test")

    parser.add_argument("--data_mode", type=str, choices=["250", "1000", "no_gt"], required=True)
    parser.add_argument("--dataset_path", default="./datasets/spike_x4k")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--load_model", type=str, default="")
    parser.add_argument("--save_image", type=str2bool, default=False)
    parser.add_argument("--save_path", default="results")
    parser.add_argument("--light", type=str, default="", help="legacy lambda/light value for X4K data")

    parser.add_argument("--model", type=str, choices=["swinsf", "la_swinsf_lite"], default="swinsf")
    parser.add_argument("--dataset_mode", type=str, choices=["legacy", "single", "grouped"], default="legacy")

    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--etas", nargs="*", type=float, default=None)
    parser.add_argument("--single_light", action="store_true")
    parser.add_argument("--grouped_eval", action="store_true")
    parser.add_argument(
        "--student_from_grouped",
        action="store_true",
        help="Use grouped test set, extract target eta, then run single-light student path.",
    )
    parser.add_argument("--target_eta", type=float, default=None)

    parser.add_argument("--use_light_code", action="store_true")
    parser.add_argument("--use_ldf_lite", action="store_true")
    parser.add_argument("--use_lsa_lite", action="store_true")
    parser.add_argument("--descriptor_dim", type=int, default=64)

    parser.add_argument("--conversion_rate", type=float, default=0.6)
    parser.add_argument(
        "--data_range",
        type=float,
        default=255.0,
        help="Use 255 if GT/pred are 0~255; use 1 if GT/pred are 0~1.",
    )
    parser.add_argument("--print_debug", action="store_true")
    parser.add_argument("--max_samples", type=int, default=0, help="Stop after N samples if >0, for quick debugging.")

    return parser.parse_args()


def get_device(args):
    test_on_gpu = torch.cuda.is_available()
    print("cuda is available, testing on gpu!" if test_on_gpu else "Testing on cpu!")
    device = torch.device(args.device if test_on_gpu else "cpu")
    if test_on_gpu and ":" in args.device:
        device_ids = [int(args.device.split(":")[-1])]
    else:
        device_ids = [0]
    return device, device_ids


def build_base_model(args):
    if args.data_mode in ["250", "no_gt"]:
        from models.SwinSF_250 import SwinSpikeFormer
        return SwinSpikeFormer(
            img_size=(250, 400), patch_size=2, in_chans=41, ref_ch=28, out_chans=1,
            embed_dim=96, depths=[6, 6], num_heads=[2, 2], window_size=5,
            mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0,
            attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm,
            ape=False, patch_norm=True, use_checkpoint=False, upscale=1,
            img_range=1.0, upsampler="", resi_connection="1conv",
        )

    from models.SwinSF_1000 import SwinSpikeFormer
    return SwinSpikeFormer(
        img_size=(1000, 1000), patch_size=4, in_chans=41, ref_ch=28, out_chans=1,
        embed_dim=64, depths=[6, 6], num_heads=[2, 2], window_size=5,
        mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0,
        attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm,
        ape=False, patch_norm=True, use_checkpoint=False, upscale=2,
        img_range=1.0, upsampler="", resi_connection="1conv",
    )


def build_model(args):
    base_model = build_base_model(args)
    if args.model != "la_swinsf_lite":
        return base_model
    if args.data_mode != "1000":
        raise ValueError("la_swinsf_lite currently uses models.SwinSF_1000.LASwinSFLite; please use --data_mode 1000")
    from models.SwinSF_1000 import LASwinSFLite
    return LASwinSFLite(
        backbone=base_model,
        descriptor_dim=args.descriptor_dim,
        use_light_code=args.use_light_code,
        use_ldf_lite=args.use_ldf_lite,
        use_lsa_lite=args.use_lsa_lite,
    )


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "net" in checkpoint:
            return checkpoint["net"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model" in checkpoint:
            return checkpoint["model"]
    return checkpoint


def strip_module_prefix(state):
    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value
    return cleaned


def load_checkpoint_if_needed(net, args):
    if not args.load_model:
        print("=> no checkpoint is provided")
        return
    if not os.path.isfile(args.load_model):
        raise FileNotFoundError("checkpoint '{}' not found".format(args.load_model))

    print("=> loading checkpoint '{}'".format(args.load_model))
    checkpoint = torch.load(args.load_model, map_location="cpu")
    state = strip_module_prefix(extract_state_dict(checkpoint))
    module = net.module if hasattr(net, "module") else net

    missing, unexpected = module.load_state_dict(state, strict=False)
    total_keys = len(module.state_dict())
    loaded_keys = total_keys - len(missing)

    print("=> try loading as full model")
    print("missing keys:", len(missing))
    print("unexpected keys:", len(unexpected))
    print("loaded keys approx: {}/{}".format(loaded_keys, total_keys))
    print("missing examples:", missing[:20])
    print("unexpected examples:", unexpected[:20])

    # Only for loading a vanilla SwinSF checkpoint into LA-SwinSF-Lite backbone.
    if args.model == "la_swinsf_lite" and loaded_keys < total_keys * 0.5:
        print("=> full-model loading seems poor; try mapping checkpoint into backbone")
        mapped = {}
        for key, value in state.items():
            if key.startswith("backbone."):
                mapped[key] = value
            else:
                mapped["backbone." + key] = value
        missing2, unexpected2 = module.load_state_dict(mapped, strict=False)
        loaded_keys2 = total_keys - len(missing2)
        print("=> try loading as backbone weights")
        print("missing keys:", len(missing2))
        print("unexpected keys:", len(unexpected2))
        print("loaded keys approx: {}/{}".format(loaded_keys2, total_keys))
        print("missing examples:", missing2[:20])
        print("unexpected examples:", unexpected2[:20])
        if loaded_keys2 <= loaded_keys:
            print("WARNING: backbone mapping did not improve loaded key count. Check checkpoint/model mismatch.")

    print("=> checkpoint loading finished")


def get_eval_etas(args):
    if args.eta is not None:
        return [args.eta]
    if args.etas:
        return args.etas
    if args.light:
        light = args.light[len("lambda"):] if args.light.startswith("lambda") else args.light
        return [float(light)]
    return None


def format_eta(eta):
    return "{:.8g}".format(float(eta))


def normalize_prediction_by_eta(pred, eta_target, args):
    if args.conversion_rate is None or args.conversion_rate <= 0:
        return pred
    eta_target = eta_target.float()
    scale = args.conversion_rate * torch.clamp(eta_target, min=1e-8)
    return pred / scale.view(-1, 1, 1, 1)


def target_index_for_eta(etas, target_eta):
    etas_2d = etas.view(-1, 1) if etas.dim() == 1 else etas
    if target_eta is None:
        return torch.zeros(etas_2d.shape[0], dtype=torch.long, device=etas.device)
    target_value = torch.tensor(float(target_eta), device=etas.device, dtype=etas_2d.dtype)
    return torch.argmin(torch.abs(etas_2d - target_value), dim=1)


def gather_target_spikes(spikes, target_index):
    """spikes: [B,K,T,H,W], target_index: [B], return [B,T,H,W]"""
    b, k, t, h, w = spikes.shape
    gather_idx = target_index.view(b, 1, 1, 1, 1).expand(-1, 1, t, h, w)
    return torch.gather(spikes, 1, gather_idx).squeeze(1)


def select_gt_mid(gt):
    if gt.dim() == 4:
        if gt.size(1) == 1:
            return gt
        if gt.size(1) == 3:
            return gt[:, 1:2]
        if gt.size(1) == 5:
            return gt[:, 2:3]
    if gt.dim() == 3:
        if gt.size(0) == 1:
            return gt
        if gt.size(0) == 3:
            return gt[1:2]
        if gt.size(0) == 5:
            return gt[2:3]
    raise ValueError("Unsupported gt shape: {}".format(tuple(gt.shape)))


def tensor_to_2d_numpy(x):
    if torch.is_tensor(x):
        x = x.detach().cpu()
    if x.dim() == 3:
        if x.size(0) != 1:
            raise ValueError("Expected [1,H,W], got {}".format(tuple(x.shape)))
        x = x[0]
    if x.dim() != 2:
        raise ValueError("Expected 2D tensor after squeeze, got {}".format(tuple(x.shape)))
    return x.numpy()


def to_uint8_image(x_np, data_range):
    x_np = np.asarray(x_np, dtype=np.float32)
    x_np = np.clip(x_np, 0, data_range)
    if abs(float(data_range) - 1.0) < 1e-8:
        x_np = x_np * 255.0
    return np.clip(x_np, 0, 255).astype(np.uint8)


def compute_psnr(pred_np, gt_np, data_range):
    pred_np = np.asarray(pred_np, dtype=np.float32)
    gt_np = np.asarray(gt_np, dtype=np.float32)
    if skimage_psnr is not None:
        return float(skimage_psnr(gt_np, pred_np, data_range=data_range))
    mse = np.mean((pred_np - gt_np) ** 2)
    return float("inf") if mse == 0 else float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))


def compute_ssim(pred_np, gt_np, data_range):
    pred_np = np.asarray(pred_np, dtype=np.float32)
    gt_np = np.asarray(gt_np, dtype=np.float32)
    if skimage_ssim is not None:
        return float(skimage_ssim(gt_np, pred_np, data_range=data_range))
    return math.nan


def tensor_psnr_like_train(pred, target):
    pred = torch.clamp(pred, 0, 255)
    target = torch.clamp(target, 0, 255)
    mse = torch.mean((pred - target) ** 2)
    return 20.0 * torch.log10(torch.tensor(255.0, device=pred.device)) - 10.0 * torch.log10(mse + 1e-8)


def get_lpips_model(device):
    try:
        import lpips
    except ImportError:
        return None
    model = lpips.LPIPS(net="alex")
    model.eval()
    return model.to(device)


def compute_lpips(pred_u8, gt_u8, lpips_model, device):
    if lpips_model is None:
        return math.nan
    pred = torch.from_numpy(pred_u8).float().to(device) / 127.5 - 1.0
    gt = torch.from_numpy(gt_u8).float().to(device) / 127.5 - 1.0
    pred = pred[None, None].expand(1, 3, pred.shape[0], pred.shape[1])
    gt = gt[None, None].expand(1, 3, gt.shape[0], gt.shape[1])
    with torch.no_grad():
        return float(lpips_model(pred, gt).item())


def meta_value(meta, key, default="sample"):
    if not isinstance(meta, dict) or key not in meta:
        return default
    value = meta[key]
    if isinstance(value, (list, tuple)):
        return str(value[0])
    if torch.is_tensor(value):
        if value.numel() == 1:
            return str(value.item())
        return str(value[0].item())
    return str(value)


def save_prediction(args, eta, name, spike_frame, gt_u8, pred_u8):
    if not args.save_image:
        return
    if cv2 is None:
        raise ImportError("cv2 is required when --save_image True")
    eta_dir = os.path.join(args.save_path, "eta_{}".format(format_eta(eta)))
    os.makedirs(eta_dir, exist_ok=True)
    cv2.imwrite(os.path.join(eta_dir, name + "_spike.png"), spike_frame)
    cv2.imwrite(os.path.join(eta_dir, name + "_ik_gt.png"), gt_u8)
    cv2.imwrite(os.path.join(eta_dir, name + "_ik_Ours.png"), pred_u8)


def update_metrics(metrics_by_eta, eta, name, spike_frame, pred, gt, args, lpips_model, device):
    pred_np = tensor_to_2d_numpy(pred)
    gt_np = tensor_to_2d_numpy(gt)
    pred_eval = np.clip(pred_np.astype(np.float32), 0, args.data_range)
    gt_eval = np.clip(gt_np.astype(np.float32), 0, args.data_range)
    pred_u8 = to_uint8_image(pred_eval, args.data_range)
    gt_u8 = to_uint8_image(gt_eval, args.data_range)

    p = compute_psnr(pred_eval, gt_eval, data_range=args.data_range)
    s = compute_ssim(pred_eval, gt_eval, data_range=args.data_range)
    lp = compute_lpips(pred_u8, gt_u8, lpips_model, device)

    eta_name = format_eta(eta)
    metrics_by_eta.setdefault(eta_name, {"psnr": [], "ssim": [], "lpips": []})
    metrics_by_eta[eta_name]["psnr"].append(float(p))
    metrics_by_eta[eta_name]["ssim"].append(float(s))
    metrics_by_eta[eta_name]["lpips"].append(float(lp))

    save_prediction(args, eta, name, spike_frame, gt_u8, pred_u8)
    print("eta {} {} psnr:{:.4f} ssim:{:.4f} lpips:{}".format(
        eta_name, name, p, s, "nan" if math.isnan(lp) else "{:.4f}".format(lp)
    ))


def print_debug_once(args, tag, spikes, gt, etas, pred_raw=None, pred=None):
    if not args.print_debug:
        return
    print("\n===== DEBUG {} =====".format(tag))
    print("spikes shape:", tuple(spikes.shape), "min/max/mean:",
          float(spikes.min()), float(spikes.max()), float(spikes.float().mean()))
    print("gt shape:", tuple(gt.shape), "min/max/mean:",
          float(gt.min()), float(gt.max()), float(gt.float().mean()))
    print("etas shape:", tuple(etas.shape), "values:", etas.detach().cpu().view(-1).tolist())
    if pred_raw is not None:
        print("pred_raw shape:", tuple(pred_raw.shape), "min/max/mean:",
              float(pred_raw.min()), float(pred_raw.max()), float(pred_raw.float().mean()))
    if pred is not None:
        print("pred_norm shape:", tuple(pred.shape), "min/max/mean:",
              float(pred.min()), float(pred.max()), float(pred.float().mean()))
    print("====================\n")


def evaluate_lite_single_eta(net, eta, device, args, lpips_model):
    dataset = Load_DataSet_X4K(
        dataset_path=args.dataset_path,
        mode="test",
        dataset_mode="single",
        etas=[eta],
        return_meta=True,
    )
    return evaluate_lite_single_loader(net, dataset, device, args, lpips_model)


def evaluate_lite_single_dataset(net, device, args, lpips_model):
    dataset = Load_DataSet_X4K(
        dataset_path=args.dataset_path,
        mode="test",
        dataset_mode="single",
        etas=None,
        return_meta=True,
    )
    return evaluate_lite_single_loader(net, dataset, device, args, lpips_model)


def evaluate_lite_single_loader(net, dataset, device, args, lpips_model):
    loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    metrics_by_eta = {}
    net.eval()

    with torch.no_grad():
        for sample_idx, (spikes, gt, etas, meta) in enumerate(tqdm(iter(loader))):
            spikes = spikes.to(device)
            gt = select_gt_mid(gt.to(device))
            etas = etas.to(device=device, dtype=torch.float32).view(-1)

            # Same as train_lite_distill.py student path.
            pred_raw, _, _ = net(spikes, etas, None)
            pred = normalize_prediction_by_eta(pred_raw, etas, args)

            if sample_idx == 0:
                print_debug_once(args, "single_loader", spikes, gt, etas, pred_raw, pred)
                if args.print_debug:
                    print("tensor_psnr_like_train:", float(tensor_psnr_like_train(pred, gt)))

            spikes_np = spikes.detach().cpu().numpy()
            if spikes_np.ndim == 4:
                spike_frame = (spikes_np[0, min(21, spikes_np.shape[1] - 1)] * 255).astype(np.uint8)
            elif spikes_np.ndim == 5:
                spike_frame = (spikes_np[0, 0, min(21, spikes_np.shape[2] - 1)] * 255).astype(np.uint8)
            else:
                raise ValueError("Unsupported spikes shape: {}".format(spikes_np.shape))

            name = meta_value(meta, "data_index", "sample")
            update_metrics(
                metrics_by_eta,
                float(etas[0].detach().cpu()),
                name,
                spike_frame,
                pred[0].detach().cpu(),
                gt[0].detach().cpu(),
                args,
                lpips_model,
                device,
            )

            if args.max_samples > 0 and sample_idx + 1 >= args.max_samples:
                break

    return metrics_by_eta


def evaluate_lite_student_from_grouped(net, eval_etas, device, args, lpips_model):
    dataset = Load_DataSet_X4K(
        dataset_path=args.dataset_path,
        mode="test",
        dataset_mode="grouped",
        etas=eval_etas,
        return_meta=True,
    )
    loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    metrics_by_eta = {}

    target_eta = args.target_eta if args.target_eta is not None else args.eta
    if target_eta is None:
        raise ValueError("--student_from_grouped needs --target_eta or --eta")

    net.eval()
    with torch.no_grad():
        for sample_idx, (spikes, gt, etas, meta) in enumerate(tqdm(iter(loader))):
            spikes = spikes.to(device)
            gt = select_gt_mid(gt.to(device))
            etas = etas.to(device=device, dtype=torch.float32)

            target_index = target_index_for_eta(etas, target_eta)
            eta_target = torch.gather(etas, 1, target_index.view(-1, 1)).squeeze(1)
            target_spikes = gather_target_spikes(spikes, target_index)

            # Same as train_lite_distill.py student validation path.
            pred_raw, _, _ = net(target_spikes, eta_target, None)
            pred = normalize_prediction_by_eta(pred_raw, eta_target, args)

            if sample_idx == 0:
                print_debug_once(args, "student_from_grouped", target_spikes, gt, eta_target, pred_raw, pred)
                if args.print_debug:
                    print("tensor_psnr_like_train:", float(tensor_psnr_like_train(pred, gt)))

            spikes_np = target_spikes.detach().cpu().numpy()
            spike_frame = (spikes_np[0, min(21, spikes_np.shape[1] - 1)] * 255).astype(np.uint8)
            name = meta_value(meta, "group_key", "sample")

            update_metrics(
                metrics_by_eta,
                float(eta_target[0].detach().cpu()),
                name,
                spike_frame,
                pred[0].detach().cpu(),
                gt[0].detach().cpu(),
                args,
                lpips_model,
                device,
            )

            if args.max_samples > 0 and sample_idx + 1 >= args.max_samples:
                break

    return metrics_by_eta


def evaluate_lite_grouped(net, eval_etas, device, args, lpips_model):
    dataset = Load_DataSet_X4K(
        dataset_path=args.dataset_path,
        mode="test",
        dataset_mode="grouped",
        etas=eval_etas,
        return_meta=True,
    )
    loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    metrics_by_eta = {}
    target_etas = [args.target_eta] if args.target_eta is not None else eval_etas

    net.eval()
    with torch.no_grad():
        for sample_idx, (spikes, gt, etas, meta) in enumerate(tqdm(iter(loader))):
            spikes = spikes.to(device)
            gt = select_gt_mid(gt.to(device))
            etas = etas.to(device=device, dtype=torch.float32)
            spikes_np = spikes.detach().cpu().numpy()

            for target_eta in target_etas:
                target_index = target_index_for_eta(etas, target_eta)
                eta_target = torch.gather(etas, 1, target_index.view(-1, 1)).squeeze(1)

                pred_raw, _, _ = net(spikes, etas, target_index)
                pred = normalize_prediction_by_eta(pred_raw, eta_target, args)

                if sample_idx == 0:
                    print_debug_once(args, "grouped_eval", spikes, gt, etas, pred_raw, pred)
                    if args.print_debug:
                        print("target_index:", target_index.detach().cpu().tolist())
                        print("eta_target:", eta_target.detach().cpu().tolist())
                        print("tensor_psnr_like_train:", float(tensor_psnr_like_train(pred, gt)))

                b_id = 0
                k_id = int(target_index[b_id].detach().cpu())
                data_names = meta_value(meta, "data_names", "sample").split("|")
                name = data_names[k_id] if k_id < len(data_names) else meta_value(meta, "group_key", "sample")
                spike_frame = (spikes_np[b_id, k_id, min(21, spikes_np.shape[2] - 1)] * 255).astype(np.uint8)

                update_metrics(
                    metrics_by_eta,
                    float(eta_target[b_id].detach().cpu()),
                    name,
                    spike_frame,
                    pred[b_id].detach().cpu(),
                    gt[b_id].detach().cpu(),
                    args,
                    lpips_model,
                    device,
                )

            if args.max_samples > 0 and sample_idx + 1 >= args.max_samples:
                break

    return metrics_by_eta


def summarize_metrics(metrics_by_eta):
    summary = {}
    for eta, values in sorted(metrics_by_eta.items(), key=lambda item: float(item[0])):
        summary[eta] = {
            "psnr": float(np.mean(values["psnr"])) if values["psnr"] else math.nan,
            "ssim": float(np.mean(values["ssim"])) if values["ssim"] else math.nan,
            "lpips": float(np.nanmean(values["lpips"])) if values["lpips"] else math.nan,
        }
    return summary


def print_eta_summary(summary):
    seen_psnr, unseen_psnr, all_psnr = [], [], []
    for eta, values in summary.items():
        eta_float = float(eta)
        print("eta {} average PSNR {:.4f} SSIM {:.4f} LPIPS {}".format(
            eta,
            values["psnr"],
            values["ssim"],
            "nan" if math.isnan(values["lpips"]) else "{:.4f}".format(values["lpips"]),
        ))
        if any(abs(eta_float - seen) < 1e-8 for seen in SEEN_ETAS):
            seen_psnr.append(values["psnr"])
        if any(abs(eta_float - unseen) < 1e-8 for unseen in UNSEEN_ETAS):
            unseen_psnr.append(values["psnr"])
        all_psnr.append(values["psnr"])

    if seen_psnr:
        print("average over seen etas PSNR {:.4f}".format(float(np.mean(seen_psnr))))
    if unseen_psnr:
        print("average over unseen etas PSNR {:.4f}".format(float(np.mean(unseen_psnr))))
    if all_psnr:
        print("average over selected etas PSNR {:.4f}".format(float(np.mean(all_psnr))))


def run_lite_evaluation(net, device, args):
    eval_etas = get_eval_etas(args)
    lpips_model = get_lpips_model(device)
    metrics_by_eta = {}

    if args.student_from_grouped:
        if eval_etas is None:
            raise ValueError("--student_from_grouped needs --etas or --eta")
        metrics_by_eta.update(evaluate_lite_student_from_grouped(net, eval_etas, device, args, lpips_model))
        print_eta_summary(summarize_metrics(metrics_by_eta))
        return

    if args.grouped_eval:
        if eval_etas is None:
            raise ValueError("Grouped evaluation needs --eta, --etas, or --light to define a light group")
        metrics_by_eta.update(evaluate_lite_grouped(net, eval_etas, device, args, lpips_model))
    elif eval_etas is None:
        print("No eta/light is specified. Testing the whole test/input dataset in single-light mode.")
        metrics_by_eta.update(evaluate_lite_single_dataset(net, device, args, lpips_model))
    else:
        for eta in eval_etas:
            metrics_by_eta.update(evaluate_lite_single_eta(net, eta, device, args, lpips_model))

    print_eta_summary(summarize_metrics(metrics_by_eta))


def run_legacy_evaluation(net, device, args):
    if args.data_mode == "250":
        spike_test = Load_DataSet_Reds(dataset_path=args.dataset_path, mode="test")
    elif args.data_mode == "1000":
        spike_test = Load_DataSet_X4K(dataset_path=args.dataset_path, mode="test", light=args.light)
    else:
        spike_test = Load_DataSet_classA(dataset_path=args.dataset_path, mode="test")

    test_loader = data.DataLoader(dataset=spike_test, batch_size=1, shuffle=False)

    if args.data_mode != "no_gt":
        loss = nn.SmoothL1Loss(reduction="mean", beta=1.0)
        loss_epoch_test, ssim_epoch_test, psnr_epoch_test = [], [], []
        net.eval()

        for sample_idx, (data_iter, gt_iter, name) in enumerate(tqdm(iter(test_loader))):
            name = name[0]
            with torch.no_grad():
                data_rec_by_tfitfp = np.array(data_iter).squeeze(0)
                data_iter = data_iter.to(device)
                gt_iter = select_gt_mid(gt_iter.to(device))

                img_pred, _, _ = net(data_iter)
                img_pred = torch.clamp(img_pred, 0, args.data_range)
                loss_value = loss(gt_iter, img_pred)

                pred_np = tensor_to_2d_numpy(img_pred[0].detach().cpu())
                gt_np = tensor_to_2d_numpy(gt_iter[0].detach().cpu())
                pred_eval = np.clip(pred_np.astype(np.float32), 0, args.data_range)
                gt_eval = np.clip(gt_np.astype(np.float32), 0, args.data_range)
                pred_u8 = to_uint8_image(pred_eval, args.data_range)
                gt_u8 = to_uint8_image(gt_eval, args.data_range)

                if args.save_image:
                    if cv2 is None:
                        raise ImportError("cv2 is required when --save_image True")
                    os.makedirs(args.save_path, exist_ok=True)
                    cv2.imwrite(os.path.join(args.save_path, name + "_spike.png"), data_rec_by_tfitfp[21, :, :] * 255)
                    cv2.imwrite(os.path.join(args.save_path, name + "_ik_gt.png"), gt_u8)
                    cv2.imwrite(os.path.join(args.save_path, name + "_ik_Ours.png"), pred_u8)

                p = compute_psnr(pred_eval, gt_eval, data_range=args.data_range)
                s = compute_ssim(pred_eval, gt_eval, data_range=args.data_range)
                loss_epoch_test.append(loss_value.item())
                psnr_epoch_test.append(p)
                ssim_epoch_test.append(s)
                print("{} psnr:{:.4f} ssim:{:.4f}".format(name, p, s))

            if args.max_samples > 0 and sample_idx + 1 >= args.max_samples:
                break

        print("total loss:{} psnr:{} ssim:{}".format(
            np.mean(loss_epoch_test), np.mean(psnr_epoch_test), np.mean(ssim_epoch_test)
        ))
    else:
        from metrics.niqe import niqe
        niqe_list = []
        net.eval()

        for sample_idx, (data_iter, name) in enumerate(tqdm(iter(test_loader))):
            name = name[0]
            with torch.no_grad():
                data_rec_by_tfitfp = np.array(data_iter).squeeze(0)
                data_iter = data_iter.to(device)
                data_iter = data_iter[:, 300:341, :, :]
                img_pred, _, _ = net(data_iter)
                img_pred = torch.clamp(img_pred, 0, args.data_range)
                pred_np = tensor_to_2d_numpy(img_pred[0].detach().cpu())
                pred_u8 = to_uint8_image(pred_np, args.data_range)

                if args.save_image:
                    if cv2 is None:
                        raise ImportError("cv2 is required when --save_image True")
                    os.makedirs(args.save_path, exist_ok=True)
                    cv2.imwrite(os.path.join(args.save_path, name + "_spike.png"), data_rec_by_tfitfp[21, :, :] * 255)
                    cv2.imwrite(os.path.join(args.save_path, name + "_ik_Ours.png"), pred_u8)

                niqe_ik = niqe(pred_u8)
                niqe_list.append(niqe_ik)
                print("{} niqe:{:.4f}".format(name, niqe_ik))

            if args.max_samples > 0 and sample_idx + 1 >= args.max_samples:
                break

        print("total niqe: {}".format(np.mean(niqe_list)))


def main():
    args = set_parser()
    device, device_ids = get_device(args)
    net = build_model(args)

    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net, device_ids=device_ids)

    print("the network is {}\n{}".format(net, "-" * 120))
    load_checkpoint_if_needed(net, args)
    net.to(device)
    torch.cuda.empty_cache()

    if args.model == "la_swinsf_lite":
        run_lite_evaluation(net, device, args)
    else:
        run_legacy_evaluation(net, device, args)


if __name__ == "__main__":
    main()

