import argparse
import os
import re

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from dataloader import parse_sample_key


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ["true", "1", "yes", "y"]


def set_parser():
    p = argparse.ArgumentParser(description="LA-SwinSF-Lite real-data single-light inference without GT")
    p.add_argument("--dataset_path", default="./datasets/real_x4k")
    p.add_argument("--input_dir", default="")
    p.add_argument("--recursive", type=str2bool, default=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--load_model", required=True)
    p.add_argument("--save_path", default="./save/real")
    p.add_argument("--eta", type=float, default=None,
                   help="Use this eta for every sample; otherwise parse eta/lambda from path, fallback 1.0")
    p.add_argument("--descriptor_dim", type=int, default=64)
    p.add_argument("--use_light_code", action="store_true")
    p.add_argument("--use_ldf_lite", action="store_true")
    p.add_argument("--use_lsa_lite", action="store_true")
    p.add_argument("--conversion_rate", type=float, default=0.0,
                   help="If >0, pred /= conversion_rate * eta. For real data, 0 disables this.")
    p.add_argument("--data_range", type=float, default=255.0)
    p.add_argument("--height", type=int, default=1000)
    p.add_argument("--width", type=int, default=1000)
    p.add_argument("--num_frames", type=int, default=41)
    p.add_argument("--frame_mode", choices=["first", "center", "last"], default="first")
    p.add_argument("--frame_start", type=int, default=None)
    p.add_argument("--preview_frame", type=int, default=21)
    p.add_argument("--flipud", action="store_true")
    p.add_argument("--save_spike", type=str2bool, default=True)
    p.add_argument("--print_debug", action="store_true")
    p.add_argument("--max_samples", type=int, default=0)
    return p.parse_args()


def build_model(args):
    from models.SwinSF_1000 import LASwinSFLite, SwinSpikeFormer
    backbone = SwinSpikeFormer(
        img_size=(args.height, args.width), patch_size=4,
        in_chans=args.num_frames, ref_ch=28, out_chans=1,
        embed_dim=64, depths=[6, 6], num_heads=[2, 2],
        window_size=5, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
        norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
        use_checkpoint=False, upscale=2, img_range=1.,
        upsampler="", resi_connection="1conv")
    return LASwinSFLite(backbone=backbone,
                        descriptor_dim=args.descriptor_dim,
                        use_light_code=args.use_light_code,
                        use_ldf_lite=args.use_ldf_lite,
                        use_lsa_lite=args.use_lsa_lite)


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "net" in ckpt:
            return ckpt["net"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        if "model" in ckpt:
            return ckpt["model"]
    return ckpt


def strip_module_prefix(state):
    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    return cleaned


def load_checkpoint(model, path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location="cpu")
    state = strip_module_prefix(extract_state_dict(ckpt))

    missing, unexpected = model.load_state_dict(state, strict=False)
    total = len(model.state_dict())
    loaded = total - len(missing)
    print(f"=> loading checkpoint '{path}'")
    print("=> try loading as full model")
    print("missing keys:", len(missing), "unexpected keys:", len(unexpected))
    print(f"loaded keys approx: {loaded}/{total}")
    print("missing examples:", missing[:20])
    print("unexpected examples:", unexpected[:20])

    # Fallback only when the checkpoint is a vanilla SwinSF backbone checkpoint.
    if loaded < total * 0.5:
        print("=> full-model loading seems poor; try mapping checkpoint into backbone")
        mapped = {}
        for k, v in state.items():
            mapped[k if k.startswith("backbone.") else "backbone." + k] = v
        missing2, unexpected2 = model.load_state_dict(mapped, strict=False)
        loaded2 = total - len(missing2)
        print("=> try loading as backbone weights")
        print("missing keys:", len(missing2), "unexpected keys:", len(unexpected2))
        print(f"loaded keys approx: {loaded2}/{total}")
        print("missing examples:", missing2[:20])
        print("unexpected examples:", unexpected2[:20])
    print("=> checkpoint loading finished")


def raw_to_spike(video_seq, h, w, flipud=False):
    video_seq = np.asarray(video_seq, dtype=np.uint8)
    img_size = h * w
    bytes_per_frame = img_size // 8
    if len(video_seq) < bytes_per_frame:
        raise ValueError(f"file has only {len(video_seq)} bytes; less than one frame")
    img_num = len(video_seq) // bytes_per_frame
    if len(video_seq) % bytes_per_frame != 0:
        print("WARNING: trailing bytes ignored:", len(video_seq) % bytes_per_frame)

    spike = np.zeros([img_num, h, w], np.uint8)
    pix_id = np.arange(0, h * w).reshape(h, w)
    comparator = np.left_shift(1, np.mod(pix_id, 8)).astype(np.uint8)
    byte_id = pix_id // 8

    for img_id in range(img_num):
        start = img_id * bytes_per_frame
        cur = video_seq[start:start + bytes_per_frame]
        data = cur[byte_id]
        frame = (np.bitwise_and(data, comparator) == comparator).astype(np.uint8)
        spike[img_id] = np.flipud(frame) if flipud else frame
    return spike


def read_dat(path, h, w, flipud=False):
    with open(path, "rb") as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    return raw_to_spike(buf, h, w, flipud=flipud)


def select_spike_window(spike, num_frames, frame_mode="first", frame_start=None):
    total = spike.shape[0]
    if total == num_frames:
        return spike
    if total < num_frames:
        pad = np.zeros((num_frames - total, spike.shape[1], spike.shape[2]), dtype=spike.dtype)
        return np.concatenate([spike, pad], axis=0)
    if frame_start is not None:
        start = int(frame_start)
    elif frame_mode == "first":
        start = 0
    elif frame_mode == "center":
        start = (total - num_frames) // 2
    elif frame_mode == "last":
        start = total - num_frames
    else:
        raise ValueError(frame_mode)
    start = max(0, min(start, total - num_frames))
    return spike[start:start + num_frames]


def get_input_dir(args):
    if args.input_dir:
        return args.input_dir
    test_input = os.path.join(args.dataset_path, "test", "input")
    if os.path.isdir(test_input):
        return test_input
    return args.dataset_path


def collect_dat_files(input_dir, recursive=True):
    if recursive:
        files = []
        for root, _, names in os.walk(input_dir):
            for name in names:
                if name.lower().endswith(".dat"):
                    files.append(os.path.join(root, name))
        return sorted(files)
    return sorted(os.path.join(input_dir, n) for n in os.listdir(input_dir) if n.lower().endswith(".dat"))


def parse_eta_from_text(text):
    patterns = [
        r"(?:lambda|eta|light)[_\-:=]*([0-9]+(?:\.[0-9]+)?)",
        r"(?:^|[/\\_\-])([0-9]+(?:\.[0-9]+)?)(?:[/\\_\-]|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


def get_eta(path, args):
    if args.eta is not None:
        return float(args.eta)
    try:
        parsed = parse_sample_key(path)
        if isinstance(parsed, dict) and parsed.get("eta") is not None:
            return float(parsed["eta"])
    except Exception:
        pass
    eta = parse_eta_from_text(path)
    return 1.0 if eta is None else float(eta)


def normalize_prediction_by_eta(pred, eta, args):
    if args.conversion_rate is None or args.conversion_rate <= 0:
        return pred
    scale = args.conversion_rate * torch.clamp(eta.float(), min=1e-8)
    return pred / scale.view(-1, 1, 1, 1)


def to_uint8_image(image, data_range):
    image = np.asarray(image, dtype=np.float32)
    image = np.clip(image, 0, data_range)
    if abs(float(data_range) - 1.0) < 1e-8:
        image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def save_uint8(path, image):
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)


def safe_stem(path, root):
    rel = os.path.relpath(path, root)
    stem = os.path.splitext(rel)[0]
    return stem.replace(os.sep, "__")


def print_debug_once(args, path, spike_full, spike_window, eta, pred_raw=None, pred=None):
    if not args.print_debug:
        return
    print("\n===== DEBUG real inference =====")
    print("file:", path)
    print("eta:", eta)
    print("spike_full shape:", spike_full.shape, "mean:", float(spike_full.mean()))
    print("spike_window shape:", spike_window.shape, "mean:", float(spike_window.mean()))
    if pred_raw is not None:
        print("pred_raw shape:", tuple(pred_raw.shape), "min/max/mean:",
              float(pred_raw.min()), float(pred_raw.max()), float(pred_raw.float().mean()))
    if pred is not None:
        print("pred_norm shape:", tuple(pred.shape), "min/max/mean:",
              float(pred.min()), float(pred.max()), float(pred.float().mean()))
    print("===============================\n")


def main():
    args = set_parser()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    input_dir = get_input_dir(args)
    dat_files = collect_dat_files(input_dir, recursive=args.recursive)
    if not dat_files:
        raise ValueError(f"No .dat files found in {input_dir}")
    os.makedirs(args.save_path, exist_ok=True)

    print("device:", device)
    print("input_dir:", input_dir)
    print("num dat files:", len(dat_files))
    print("num_frames:", args.num_frames, "frame_mode:", args.frame_mode, "frame_start:", args.frame_start)

    model = build_model(args).to(device)
    load_checkpoint(model, args.load_model)
    model.eval()

    with torch.no_grad():
        for idx, dat_path in enumerate(tqdm(dat_files)):
            spike_full = read_dat(dat_path, args.height, args.width, flipud=args.flipud)
            spike_window = select_spike_window(spike_full, args.num_frames, args.frame_mode, args.frame_start)
            eta_value = get_eta(dat_path, args)

            # Standardized single-light student path: model([B,T,H,W], [B], None)
            spikes = torch.from_numpy(spike_window.copy()).float().unsqueeze(0).to(device)
            eta = torch.tensor([eta_value], dtype=torch.float32, device=device)
            pred_raw, _, _ = model(spikes, eta, None)
            pred = normalize_prediction_by_eta(pred_raw, eta, args)

            if idx == 0:
                print_debug_once(args, dat_path, spike_full, spike_window, eta_value, pred_raw, pred)

            pred_np = pred[0, 0].detach().cpu().numpy()
            pred_u8 = to_uint8_image(pred_np, args.data_range)
            stem = safe_stem(dat_path, input_dir)
            save_uint8(os.path.join(args.save_path, stem + "_ik_Ours.png"), pred_u8)

            if args.save_spike:
                preview_id = min(args.preview_frame, spike_window.shape[0] - 1)
                spike_u8 = (spike_window[preview_id] * 255).astype(np.uint8)
                save_uint8(os.path.join(args.save_path, stem + "_spike.png"), spike_u8)

            print(f"{os.path.basename(dat_path)} eta {eta_value} frames {spike_full.shape[0]} -> {spike_window.shape[0]}")

            if args.max_samples > 0 and idx + 1 >= args.max_samples:
                break


if __name__ == "__main__":
    main()

