import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class TSA(nn.Module):
    r""" TSA

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_l , x_m ,x_r , mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x_m.shape

        q = self.to_q(x_l).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        k = self.to_q(x_r).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        v = self.to_q(x_m).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x_m = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x_m = self.proj(x_m)
        x_m = self.proj_drop(x_m)
        x_l = x_l.transpose(1, 2).reshape(B_, N, C)
        x_r = x_r.transpose(1, 2).reshape(B_, N, C)
        return x_l , x_m , x_r

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SAB(nn.Module):
    r""" Swin Spikesformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        is_mulattn(bool):use multi attention or not
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn_mul = TSA(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x_l , x_m , x_r, x_size):
        H, W = x_size
        B, L, C = x_m.shape
        # assert L == H * W, "input feature has wrong size"

        res_l,res_m,res_r = x_l , x_m , x_r
        x_l = self.norm1(x_l)
        x_m = self.norm1(x_m)
        x_r = self.norm1(x_r)
        x_l = x_l.view(B, H, W, C)
        x_m = x_m.view(B, H, W, C)
        x_r = x_r.view(B, H, W, C)
        # cyclic shift
        if self.shift_size > 0:
            shifted_x_l = torch.roll(x_l, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_x_m = torch.roll(x_m, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_x_r = torch.roll(x_r, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x_l = x_l
            shifted_x_m = x_m
            shifted_x_r = x_r

        # partition windows
        x_windows_l = window_partition(shifted_x_l, self.window_size)  # nW*B, window_size, window_size, C
        x_windows_l = x_windows_l.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        x_windows_m = window_partition(shifted_x_m, self.window_size)  # nW*B, window_size, window_size, C
        x_windows_m = x_windows_m.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        x_windows_r = window_partition(shifted_x_r, self.window_size)  # nW*B, window_size, window_size, C
        x_windows_r = x_windows_r.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        now_mask = self.attn_mask if self.input_resolution == x_size else self.calculate_mask(x_size).to(x_m.device)
        
        attn_windows_m = self.attn(x_windows_m, mask=now_mask)

        attn_windows_l,attn_windows_m_ref,attn_windows_r = self.attn_mul(x_windows_l ,x_windows_m,x_windows_r , mask=now_mask)  # nW*B, window_size*window_size, C

        attn_windows_m = attn_windows_m + 0.1 * attn_windows_m_ref
        # merge windows
        attn_windows_l = attn_windows_l.view(-1, self.window_size, self.window_size, C)
        shifted_x_l = window_reverse(attn_windows_l, self.window_size, H, W)  # B H' W' C
        attn_windows_m = attn_windows_m.view(-1, self.window_size, self.window_size, C)
        shifted_x_m = window_reverse(attn_windows_m, self.window_size, H, W)  # B H' W' C
        attn_windows_r = attn_windows_r.view(-1, self.window_size, self.window_size, C)
        shifted_x_r = window_reverse(attn_windows_r, self.window_size, H, W)  # B H' W' C
        # reverse cyclic shift
        if self.shift_size > 0:
            x_l = torch.roll(shifted_x_l, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            x_r = torch.roll(shifted_x_r, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            x_m = torch.roll(shifted_x_m, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_l = shifted_x_l
            x_r = shifted_x_r
            x_m = shifted_x_m
        x_l = x_l.view(B, H * W, C)
        x_m = x_m.view(B, H * W, C)
        x_r = x_r.view(B, H * W, C)

        # FFN
        x_l = res_l + self.drop_path(x_l)
        x_m = res_m + self.drop_path(x_m)
        x_r= res_r + self.drop_path(x_r)
        x_l = x_l + self.drop_path(self.mlp(self.norm2(x_l)))
        x_m = x_m + self.drop_path(self.mlp(self.norm2(x_m)))
        x_r = x_r + self.drop_path(self.mlp(self.norm2(x_r)))

        return x_l , x_m , x_r

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SAB(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x_l , x_m , x_r, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x_l , x_m , x_r = checkpoint.checkpoint(blk, x_l , x_m , x_r, x_size)
            else:
                x_l , x_m , x_r = blk(x_l , x_m , x_r, x_size)
        if self.downsample is not None:
            x_l , x_m , x_r = self.downsample(x_l , x_m , x_r)
        return x_l , x_m , x_r

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RSSB(nn.Module):
    """Residual Swin Transformer Block (RSSB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSSB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=dim, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=dim, embed_dim=dim,
            norm_layer=None)

    def forward(self, x_l,x_m , x_r, x_size):
        res_l,res_m,res_r = x_l,x_m,x_r
        x_patch_size = (x_size[0]//self.patch_size,x_size[1]//self.patch_size)
        x_l,x_m , x_r = self.residual_group( x_l,x_m , x_r, x_patch_size)
        x_l = self.patch_embed(self.conv(self.patch_unembed(x_l , x_size))) + res_l
        x_m = self.patch_embed(self.conv(self.patch_unembed(x_m , x_size))) + res_m
        x_r = self.patch_embed(self.conv(self.patch_unembed(x_r , x_size))) + res_r
        return x_l , x_m , x_r 
        #return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(patch_size[0],patch_size[1]), stride=(patch_size[0],patch_size[1]))
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.embed_dim = embed_dim

        self.patches_resolution = [self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        # Using ConvTranspose2d to "unembed" the patches
        self.unproj = nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=(self.patch_size[0],self.patch_size[1]) ,stride=(self.patch_size[0],self.patch_size[1]))

    def forward(self, x , x_size):
        B, N, C = x.shape

        # Reshape the input to the format expected by ConvTranspose2d
        x = x.transpose(1, 2).view(B, C, x_size[0] // self.patch_size[0] ,x_size[1] // self.patch_size[1])

        # Unproject patches to image
        x = self.unproj(x)

        return x



class SwinSpikeFormer(nn.Module):
    r""" SwinSpikeFormer
        A PyTorch impl of : `SwinSpikeformer`, based on Swin Transformer.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self, img_size=64, patch_size=1, in_chans=41, ref_ch = 28,out_chans = 1,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv',
                 **kwargs):
        super(SwinSpikeFormer, self).__init__()
        num_in_ch = in_chans
        num_out_ch = out_chans
        ref_ch = ref_ch
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first_ref = nn.Conv2d(ref_ch, embed_dim, 3, 1, 1)
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSSB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], 
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection
                         )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        #####################################################################################################
        ################################### 3, reconstruction ######################################
        self.conv_lm = nn.Sequential(nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1),
                                       nn.ReLU())
        self.conv_last = nn.Sequential(nn.Conv2d(embed_dim * 3, num_out_ch, 3, 1, 1),
                                       nn.ReLU())

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x):
        x_size = (x.shape[2], x.shape[3])
        x_l = x[:,0:28,:,:]
        x_m = x
        x_r = x[:,13:41,:,:]
        x_l = self.conv_first_ref(x_l)
        x_l = self.patch_embed(x_l)
        x_m = self.conv_first(x_m)
        x_m = self.patch_embed(x_m)
        x_r = self.conv_first_ref(x_r)
        x_r = self.patch_embed(x_r)
        if self.ape:
            x_l = x_l + self.absolute_pos_embed
            x_m = x_m + self.absolute_pos_embed
            x_r = x_r + self.absolute_pos_embed
        x_l = self.pos_drop(x_l)
        x_m = self.pos_drop(x_m)
        x_r = self.pos_drop(x_r)
        for layer in self.layers:
            x_l , x_m , x_r = layer(x_l , x_m , x_r, x_size)
        x_l = self.norm(x_l)  # B L C
        x_m = self.norm(x_m)  # B L C
        x_r = self.norm(x_r)  # B L C
        x_l = self.patch_unembed(x_l, x_size)
        x_m = self.patch_unembed(x_m, x_size)
        x_r = self.patch_unembed(x_r, x_size)
        x_m = torch.concat([x_l,x_m,x_r],dim=1)
        x_m = self.conv_last(x_m)
        x_l = self.conv_lm(x_l)
        x_r = self.conv_lm(x_r)
        return x_m,x_l,x_r

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops


class EtaEmbedding(nn.Module):
    """Continuous eta embedding for seen and unseen illumination values.

    Input eta shapes: [B], [B, 1], or [B, K].
    Output shape: [B, K, embed_dim].
    """

    def __init__(self, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def normalize_etas(self, etas, batch_size, num_lights, device):
        if etas is None:
            etas = torch.ones(batch_size, num_lights, device=device)
        elif not torch.is_tensor(etas):
            etas = torch.tensor(etas, dtype=torch.float32, device=device)
        else:
            etas = etas.to(device=device, dtype=torch.float32)

        if etas.dim() == 0:
            etas = etas.view(1, 1).expand(batch_size, num_lights)
        elif etas.dim() == 1:
            if etas.numel() == batch_size:
                etas = etas.view(batch_size, 1).expand(batch_size, num_lights)
            elif etas.numel() == num_lights:
                etas = etas.view(1, num_lights).expand(batch_size, num_lights)
            else:
                raise AssertionError('eta shape must be [B], [K], [B, 1], or [B, K]')
        elif etas.dim() == 2:
            if etas.shape[0] != batch_size:
                raise AssertionError('eta batch size mismatch')
            if etas.shape[1] == 1 and num_lights > 1:
                etas = etas.expand(batch_size, num_lights)
            elif etas.shape[1] != num_lights:
                raise AssertionError('eta light dimension mismatch')
        else:
            raise AssertionError('eta shape must be [B], [B, 1], or [B, K]')
        return etas

    def forward(self, etas, batch_size=None, num_lights=None, device=None):
        if torch.is_tensor(etas):
            device = etas.device if device is None else device
        if batch_size is None or num_lights is None:
            if etas.dim() == 1:
                batch_size, num_lights = etas.shape[0], 1
            else:
                batch_size, num_lights = etas.shape[0], etas.shape[1]
        etas = self.normalize_etas(etas, batch_size, num_lights, device)
        log_eta = torch.log(torch.clamp(etas, min=1e-8)).unsqueeze(-1)
        return self.net(log_eta)


class EtaFiLM(nn.Module):
    """Zero-initialized FiLM modulation, identity at initialization."""

    def __init__(self, channels, eta_embed_dim):
        super().__init__()
        self.to_gamma_beta = nn.Linear(eta_embed_dim, channels * 2)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, x, eta_embed):
        if eta_embed is None:
            return x
        if x.dim() == 5:
            gamma, beta = self.to_gamma_beta(eta_embed).chunk(2, dim=-1)
            gamma = gamma[:, :, :, None, None]
            beta = beta[:, :, :, None, None]
            return x * (1.0 + gamma) + beta
        if x.dim() == 6:
            gamma, beta = self.to_gamma_beta(eta_embed).chunk(2, dim=-1)
            gamma = gamma[:, :, None, :, None, None]
            beta = beta[:, :, None, :, None, None]
            return x * (1.0 + gamma) + beta
        raise AssertionError('EtaFiLM expects [B,K,C,H,W] or [B,K,3,C,H,W]')


class LightDimensionFilter(nn.Module):
    """Learnable residual filtering along the light dimension.

    Input/output shapes:
        [B, K, C, H, W] or [B, K, 3, C, H, W].
    K=1 returns identity.
    """

    def __init__(self, channels, eta_embed_dim=32):
        super().__init__()
        self.channels = channels
        self.filter = nn.Conv1d(channels, channels, kernel_size=3, padding=1,
                                groups=channels, bias=True)
        self.gate_from_feat = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.gate_from_eta = nn.Linear(eta_embed_dim, channels) if eta_embed_dim else None
        nn.init.zeros_(self.filter.weight)
        nn.init.zeros_(self.filter.bias)
        nn.init.zeros_(self.gate_from_feat[-1].weight)
        nn.init.zeros_(self.gate_from_feat[-1].bias)
        if self.gate_from_eta is not None:
            nn.init.zeros_(self.gate_from_eta.weight)
            nn.init.zeros_(self.gate_from_eta.bias)

    def forward(self, x, eta_embed=None):
        if x.dim() == 5:
            return self._forward_5d(x, eta_embed)
        if x.dim() == 6:
            b, k, t, c, h, w = x.shape
            x_flat = x.permute(0, 2, 1, 3, 4, 5).contiguous().view(b * t, k, c, h, w)
            eta_flat = None
            if eta_embed is not None:
                eta_flat = eta_embed[:, None, :, :].expand(b, t, k, eta_embed.shape[-1]).contiguous().view(b * t, k, -1)
            out = self._forward_5d(x_flat, eta_flat)
            return out.view(b, t, k, c, h, w).permute(0, 2, 1, 3, 4, 5).contiguous()
        raise AssertionError('LightDimensionFilter expects [B,K,C,H,W] or [B,K,3,C,H,W]')

    def _forward_5d(self, x, eta_embed=None):
        b, k, c, h, w = x.shape
        if k == 1:
            return x
        if c != self.channels:
            raise AssertionError('LDF channel mismatch')

        x_for_conv = x.permute(0, 3, 4, 2, 1).contiguous().view(b * h * w, c, k)
        delta = self.filter(x_for_conv)
        delta = delta.view(b, h, w, c, k).permute(0, 4, 3, 1, 2).contiguous()

        pooled = x.mean(dim=(-1, -2))
        gate_logits = self.gate_from_feat(pooled)
        if eta_embed is not None and self.gate_from_eta is not None:
            gate_logits = gate_logits + self.gate_from_eta(eta_embed)
        gate = torch.sigmoid(gate_logits).view(b, k, c, 1, 1)
        return x + gate * delta


class LightSpikeAttention(nn.Module):
    """Window-based attention over the light dimension.

    Input/output shapes:
        [B, K, C, H, W] or [B, K, 3, C, H, W].
    Spatial windows bound memory use; attention itself is over K.
    """

    def __init__(self, channels, num_heads=4, window_size=5, eta_embed_dim=32):
        super().__init__()
        if channels % num_heads != 0:
            raise AssertionError('channels must be divisible by num_heads')
        self.channels = channels
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (channels // num_heads) ** -0.5
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.eta_to_bias = nn.Linear(eta_embed_dim, channels) if eta_embed_dim else None
        self.lsa_scale = nn.Parameter(torch.zeros(1))
        if self.eta_to_bias is not None:
            nn.init.zeros_(self.eta_to_bias.weight)
            nn.init.zeros_(self.eta_to_bias.bias)

    def forward(self, x, eta_embed=None):
        if x.dim() == 5:
            return self._forward_5d(x, eta_embed)
        if x.dim() == 6:
            b, k, t, c, h, w = x.shape
            x_flat = x.permute(0, 2, 1, 3, 4, 5).contiguous().view(b * t, k, c, h, w)
            eta_flat = None
            if eta_embed is not None:
                eta_flat = eta_embed[:, None, :, :].expand(b, t, k, eta_embed.shape[-1]).contiguous().view(b * t, k, -1)
            out = self._forward_5d(x_flat, eta_flat)
            return out.view(b, t, k, c, h, w).permute(0, 2, 1, 3, 4, 5).contiguous()
        raise AssertionError('LightSpikeAttention expects [B,K,C,H,W] or [B,K,3,C,H,W]')

    def _forward_5d(self, x, eta_embed=None):
        b, k, c, h, w = x.shape
        if k == 1:
            return x
        if c != self.channels:
            raise AssertionError('LSA channel mismatch')

        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h or pad_w:
            x_pad = F.pad(x.view(b * k, c, h, w), (0, pad_w, 0, pad_h), mode='replicate')
            hp, wp = h + pad_h, w + pad_w
            x_pad = x_pad.view(b, k, c, hp, wp)
        else:
            hp, wp = h, w
            x_pad = x

        nh, nw = hp // ws, wp // ws
        tokens = x_pad.view(b, k, c, nh, ws, nw, ws)
        tokens = tokens.permute(0, 3, 5, 4, 6, 1, 2).contiguous()
        tokens = tokens.view(b * nh * nw * ws * ws, k, c)

        if eta_embed is not None and self.eta_to_bias is not None:
            eta_bias = self.eta_to_bias(eta_embed)
            eta_bias = eta_bias[:, None, None, None, None, :, :].expand(b, nh, nw, ws, ws, k, c)
            tokens = tokens + eta_bias.contiguous().view(b * nh * nw * ws * ws, k, c)

        qkv = self.qkv(tokens).view(tokens.shape[0], k, 3, self.num_heads, c // self.num_heads)
        q, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        attn = (q @ key.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ value).permute(0, 2, 1, 3).contiguous().view(tokens.shape[0], k, c)
        out = self.proj(out)
        out = out.view(b, nh, nw, ws, ws, k, c)
        out = out.permute(0, 5, 6, 1, 3, 2, 4).contiguous().view(b, k, c, hp, wp)
        out = out[:, :, :, :h, :w]
        return x + self.lsa_scale * out


class IlluminationAwareSwinSF(nn.Module):
    """Grouped/single-light wrapper around the original SwinSpikeFormer.

    Inputs:
        spikes [B,T,H,W] -> outputs [B,1,H,W]
        spikes [B,K,T,H,W] -> outputs [B,K,1,H,W]
        etas [B], [B,1], or [B,K]
    """

    def __init__(self, backbone=None, use_eta_embed=True, use_ldf=False,
                 use_lsa=False, eta_embed_dim=32, lsa_heads=4,
                 lsa_window_size=5, **backbone_kwargs):
        super().__init__()
        self.backbone = backbone if backbone is not None else SwinSpikeFormer(**backbone_kwargs)
        self.use_eta_embed = use_eta_embed
        self.use_ldf = use_ldf
        self.use_lsa = use_lsa
        channels = self.backbone.embed_dim
        self.eta_embedding = EtaEmbedding(eta_embed_dim) if use_eta_embed else None
        self.eta_film = EtaFiLM(channels, eta_embed_dim) if use_eta_embed else None
        self.ldf = LightDimensionFilter(channels, eta_embed_dim if use_eta_embed else 0) if use_ldf else None
        self.lsa = LightSpikeAttention(channels, num_heads=lsa_heads,
                                       window_size=lsa_window_size,
                                       eta_embed_dim=eta_embed_dim if use_eta_embed else 0) if use_lsa else None

    def forward(self, spikes, etas=None, return_features=False):
        if spikes.dim() == 4:
            spikes = spikes.unsqueeze(1)
            squeeze_light_dim = True
        elif spikes.dim() == 5:
            squeeze_light_dim = False
        else:
            raise AssertionError('spikes must be [B,T,H,W] or [B,K,T,H,W]')

        b, k, t, h, w = spikes.shape
        if t < 41:
            raise AssertionError('SwinSF_1000 expects at least 41 spike frames')
        spikes_flat = spikes.contiguous().view(b * k, t, h, w)
        x_size = (h, w)

        eta_embed = None
        if self.eta_embedding is not None:
            etas = self.eta_embedding.normalize_etas(etas, b, k, spikes.device)
            eta_embed = self.eta_embedding(etas, b, k, spikes.device)

        x_l, x_m, x_r = self.extract_spike_features(spikes_flat)
        branch_maps = self.tokens_to_branch_maps(x_l, x_m, x_r, b, k, x_size)
        features = {}
        if return_features:
            features['before_ldf'] = branch_maps

        if self.eta_film is not None:
            branch_maps = self.eta_film(branch_maps, eta_embed)
        if self.ldf is not None:
            branch_maps = self.ldf(branch_maps, eta_embed)
        if return_features:
            features['after_ldf'] = branch_maps
        if self.lsa is not None:
            branch_maps = self.lsa(branch_maps, eta_embed)
        if return_features:
            features['after_lsa'] = branch_maps

        x_l, x_m, x_r = self.branch_maps_to_tokens(branch_maps)
        x_m, x_l, x_r = self.forward_backbone_from_tokens(x_l, x_m, x_r, x_size)
        x_m = x_m.view(b, k, *x_m.shape[1:])
        x_l = x_l.view(b, k, *x_l.shape[1:])
        x_r = x_r.view(b, k, *x_r.shape[1:])

        if squeeze_light_dim:
            x_m, x_l, x_r = x_m[:, 0], x_l[:, 0], x_r[:, 0]
        if return_features:
            return x_m, x_l, x_r, features
        return x_m, x_l, x_r

    def extract_spike_features(self, x):
        x_l = x[:, 0:28, :, :]
        x_m = x
        x_r = x[:, 13:41, :, :]
        x_l = self.backbone.patch_embed(self.backbone.conv_first_ref(x_l))
        x_m = self.backbone.patch_embed(self.backbone.conv_first(x_m))
        x_r = self.backbone.patch_embed(self.backbone.conv_first_ref(x_r))
        if self.backbone.ape:
            x_l = x_l + self.backbone.absolute_pos_embed
            x_m = x_m + self.backbone.absolute_pos_embed
            x_r = x_r + self.backbone.absolute_pos_embed
        x_l = self.backbone.pos_drop(x_l)
        x_m = self.backbone.pos_drop(x_m)
        x_r = self.backbone.pos_drop(x_r)
        return x_l, x_m, x_r

    def tokens_to_branch_maps(self, x_l, x_m, x_r, batch_size, num_lights, x_size):
        ph, pw = self.backbone.patch_embed.patch_size
        hp, wp = x_size[0] // ph, x_size[1] // pw
        c = self.backbone.embed_dim
        branches = []
        for tokens in [x_l, x_m, x_r]:
            if tokens.shape[1] != hp * wp:
                raise AssertionError('patch token length does not match input size')
            branch = tokens.view(batch_size, num_lights, hp, wp, c).permute(0, 1, 4, 2, 3).contiguous()
            branches.append(branch)
        return torch.stack(branches, dim=2)

    def branch_maps_to_tokens(self, branch_maps):
        b, k, branches, c, hp, wp = branch_maps.shape
        if branches != 3:
            raise AssertionError('expected three temporal branches')
        tokens = []
        for branch_id in range(3):
            branch = branch_maps[:, :, branch_id]
            tokens.append(branch.permute(0, 1, 3, 4, 2).contiguous().view(b * k, hp * wp, c))
        return tokens[0], tokens[1], tokens[2]

    def forward_backbone_from_tokens(self, x_l, x_m, x_r, x_size):
        for layer in self.backbone.layers:
            x_l, x_m, x_r = layer(x_l, x_m, x_r, x_size)
        x_l = self.backbone.norm(x_l)
        x_m = self.backbone.norm(x_m)
        x_r = self.backbone.norm(x_r)
        x_l = self.backbone.patch_unembed(x_l, x_size)
        x_m = self.backbone.patch_unembed(x_m, x_size)
        x_r = self.backbone.patch_unembed(x_r, x_size)
        x_m = torch.concat([x_l, x_m, x_r], dim=1)
        x_m = self.backbone.conv_last(x_m)
        x_l = self.backbone.conv_lm(x_l)
        x_r = self.backbone.conv_lm(x_r)
        return x_m, x_l, x_r


def normalize_etas_lite(etas, batch_size, num_lights, device):
    if etas is None:
        return torch.ones(batch_size, num_lights, device=device)
    if not torch.is_tensor(etas):
        etas = torch.tensor(etas, dtype=torch.float32, device=device)
    else:
        etas = etas.to(device=device, dtype=torch.float32)
    if etas.dim() == 0:
        etas = etas.view(1, 1).expand(batch_size, num_lights)
    elif etas.dim() == 1:
        if etas.numel() == batch_size:
            etas = etas.view(batch_size, 1).expand(batch_size, num_lights)
        elif etas.numel() == num_lights:
            etas = etas.view(1, num_lights).expand(batch_size, num_lights)
        else:
            raise AssertionError('eta shape must be [B], [K], [B,1], or [B,K]')
    elif etas.dim() == 2:
        if etas.shape[0] != batch_size:
            raise AssertionError('eta batch size mismatch')
        if etas.shape[1] == 1 and num_lights > 1:
            etas = etas.expand(batch_size, num_lights)
        elif etas.shape[1] != num_lights:
            raise AssertionError('eta light dimension mismatch')
    else:
        raise AssertionError('eta shape must be [B], [B,1], or [B,K]')
    return etas


def normalize_target_index(target_index, batch_size, device):
    if target_index is None:
        return torch.zeros(batch_size, dtype=torch.long, device=device)
    if isinstance(target_index, int):
        return torch.full((batch_size,), target_index, dtype=torch.long, device=device)
    if not torch.is_tensor(target_index):
        target_index = torch.tensor(target_index, dtype=torch.long, device=device)
    else:
        target_index = target_index.to(device=device, dtype=torch.long)
    if target_index.dim() == 0:
        target_index = target_index.view(1).expand(batch_size)
    if target_index.dim() != 1 or target_index.shape[0] != batch_size:
        raise AssertionError('target_index must be int or Tensor [B]')
    return target_index


class LightDescriptorEncoder(nn.Module):
    """Encode a grouped spike stream into lightweight light descriptors.

    Inputs:
        spikes_group: [B, K, T, H, W]
        etas: [B, K]
    Output:
        desc: [B, K, D]
    """

    def __init__(self, descriptor_dim=64, temporal_bins=8, density_size=8,
                 hidden_dim=128, use_low_res_density=True):
        super().__init__()
        self.descriptor_dim = descriptor_dim
        self.temporal_bins = temporal_bins
        self.density_size = density_size
        self.use_low_res_density = use_low_res_density
        density_dim = density_size * density_size if use_low_res_density else 0
        in_dim = 3 + temporal_bins + density_dim
        self.temporal_pool = nn.AdaptiveAvgPool1d(temporal_bins)
        self.density_pool = nn.AdaptiveAvgPool2d((density_size, density_size))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, descriptor_dim),
        )

    def forward(self, spikes_group, etas):
        if spikes_group.dim() != 5:
            raise AssertionError('LightDescriptorEncoder expects spikes_group [B,K,T,H,W]')
        b, k, t, h, w = spikes_group.shape
        etas = normalize_etas_lite(etas, b, k, spikes_group.device)
        spikes = spikes_group.float()

        log_eta = torch.log(torch.clamp(etas, min=1e-8)).unsqueeze(-1)
        mean_rate = spikes.mean(dim=(2, 3, 4), keepdim=False).unsqueeze(-1)
        std_rate = spikes.std(dim=(2, 3, 4), keepdim=False).unsqueeze(-1)

        temporal_profile = spikes.mean(dim=(3, 4)).contiguous().view(b * k, 1, t)
        temporal_profile = self.temporal_pool(temporal_profile).view(b, k, self.temporal_bins)

        parts = [log_eta, mean_rate, std_rate, temporal_profile]
        if self.use_low_res_density:
            density = spikes.mean(dim=2).contiguous().view(b * k, 1, h, w)
            density = self.density_pool(density).view(b, k, self.density_size * self.density_size)
            parts.append(density)
        desc_input = torch.cat(parts, dim=-1)
        return self.mlp(desc_input)


class LightDimensionFilterLite(nn.Module):
    """Residual gated filtering over descriptors along K.

    Input/output: [B, K, D]. K=1 returns identity.
    """

    def __init__(self, descriptor_dim=64):
        super().__init__()
        self.descriptor_dim = descriptor_dim
        self.filter = nn.Conv1d(descriptor_dim, descriptor_dim, kernel_size=3,
                                padding=1, groups=descriptor_dim)
        self.gate = nn.Sequential(
            nn.Linear(descriptor_dim, descriptor_dim),
            nn.GELU(),
            nn.Linear(descriptor_dim, descriptor_dim),
        )
        nn.init.zeros_(self.filter.weight)
        nn.init.zeros_(self.filter.bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, desc):
        if desc.dim() != 3:
            raise AssertionError('LightDimensionFilterLite expects desc [B,K,D]')
        b, k, d = desc.shape
        if d != self.descriptor_dim:
            raise AssertionError('LDF-lite descriptor dimension mismatch')
        if k == 1:
            return desc
        delta = self.filter(desc.transpose(1, 2)).transpose(1, 2)
        gate = torch.sigmoid(self.gate(desc))
        return desc + gate * delta


class LightSpikeAttentionLite(nn.Module):
    """Multi-head attention over light descriptors.

    Input/output: [B, K, D]. K=1 returns identity.
    """

    def __init__(self, descriptor_dim=64, num_heads=2):
        super().__init__()
        self.descriptor_dim = descriptor_dim
        self.attn = nn.MultiheadAttention(embed_dim=descriptor_dim, num_heads=num_heads,
                                          batch_first=True)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, desc):
        if desc.dim() != 3:
            raise AssertionError('LightSpikeAttentionLite expects desc [B,K,D]')
        if desc.shape[-1] != self.descriptor_dim:
            raise AssertionError('LSA-lite descriptor dimension mismatch')
        if desc.shape[1] == 1:
            return desc
        attn_out, _ = self.attn(desc, desc, desc, need_weights=False)
        return desc + self.scale * attn_out


class LightAdapter(nn.Module):
    """Zero-initialized FiLM adapter from light code to SwinSF feature channels."""

    def __init__(self, descriptor_dim, channels):
        super().__init__()
        self.to_gamma_beta = nn.Sequential(
            nn.Linear(descriptor_dim, descriptor_dim),
            nn.GELU(),
            nn.Linear(descriptor_dim, channels * 2),
        )
        nn.init.zeros_(self.to_gamma_beta[-1].weight)
        nn.init.zeros_(self.to_gamma_beta[-1].bias)

    def forward(self, z_target):
        gamma, beta = self.to_gamma_beta(z_target).chunk(2, dim=-1)
        return gamma[:, :, None, None], beta[:, :, None, None]


class LASwinSFLite(nn.Module):
    """Lightweight illumination-aware SwinSF.

    Only the target light enters the SwinSF backbone. The whole group is used
    only for [B,K,D] light descriptors.
    """

    def __init__(self, backbone=None, descriptor_dim=64, use_light_code=True,
                 use_ldf_lite=False, use_lsa_lite=False, **backbone_kwargs):
        super().__init__()
        self.backbone = backbone if backbone is not None else SwinSpikeFormer(**backbone_kwargs)
        self.descriptor_dim = descriptor_dim
        self.use_light_code = use_light_code
        self.use_ldf_lite = use_ldf_lite
        self.use_lsa_lite = use_lsa_lite
        self.descriptor_encoder = LightDescriptorEncoder(descriptor_dim=descriptor_dim)
        self.ldf_lite = LightDimensionFilterLite(descriptor_dim) if use_ldf_lite else None
        self.lsa_lite = LightSpikeAttentionLite(descriptor_dim, num_heads=2) if use_lsa_lite else None
        self.adapter = LightAdapter(descriptor_dim, self.backbone.embed_dim) if use_light_code else None

    def forward(self, spikes, etas=None, target_index=None, return_features=False):
        if spikes.dim() == 4:
            spikes = spikes.unsqueeze(1)
        elif spikes.dim() != 5:
            raise AssertionError('LASwinSFLite expects spikes [B,T,H,W] or [B,K,T,H,W]')

        b, k, t, h, w = spikes.shape
        if t < 41:
            raise AssertionError('SwinSF_1000 expects at least 41 spike frames')
        etas = normalize_etas_lite(etas, b, k, spikes.device)
        target_index = normalize_target_index(target_index, b, spikes.device)
        if torch.any(target_index < 0) or torch.any(target_index >= k):
            raise AssertionError('target_index is out of range for grouped input')

        gather_idx = target_index.view(b, 1, 1, 1, 1).expand(-1, 1, t, h, w)
        spikes_target = torch.gather(spikes, 1, gather_idx).squeeze(1)
        eta_target = torch.gather(etas, 1, target_index.view(b, 1)).squeeze(1)

        features = {'eta_target': eta_target}
        z_target = None
        if self.use_light_code or self.use_ldf_lite or self.use_lsa_lite:
            desc = self.descriptor_encoder(spikes, etas)
            features['desc_before_lite'] = desc
            if self.ldf_lite is not None:
                desc = self.ldf_lite(desc)
            features['desc_after_ldf_lite'] = desc
            if self.lsa_lite is not None:
                desc = self.lsa_lite(desc)
            features['desc_after_lsa_lite'] = desc
            z_target = torch.gather(
                desc, 1, target_index.view(b, 1, 1).expand(-1, 1, desc.shape[-1])
            ).squeeze(1)
            features['z_target'] = z_target

        out = self.forward_target_backbone(spikes_target, z_target)
        if return_features:
            return out[0], out[1], out[2], features
        return out

    def forward_target_backbone(self, x, z_target=None):
        x_size = (x.shape[2], x.shape[3])
        x_l = x[:, 0:28, :, :]
        x_m = x
        x_r = x[:, 13:41, :, :]
        x_l = self.backbone.patch_embed(self.backbone.conv_first_ref(x_l))
        x_m = self.backbone.patch_embed(self.backbone.conv_first(x_m))
        x_r = self.backbone.patch_embed(self.backbone.conv_first_ref(x_r))
        if self.backbone.ape:
            x_l = x_l + self.backbone.absolute_pos_embed
            x_m = x_m + self.backbone.absolute_pos_embed
            x_r = x_r + self.backbone.absolute_pos_embed
        x_l = self.backbone.pos_drop(x_l)
        x_m = self.backbone.pos_drop(x_m)
        x_r = self.backbone.pos_drop(x_r)

        if self.adapter is not None and z_target is not None:
            gamma, beta = self.adapter(z_target)
            gamma = gamma.flatten(2).transpose(1, 2)
            beta = beta.flatten(2).transpose(1, 2)
            x_l = x_l * (1.0 + gamma) + beta
            x_m = x_m * (1.0 + gamma) + beta
            x_r = x_r * (1.0 + gamma) + beta

        for layer in self.backbone.layers:
            x_l, x_m, x_r = layer(x_l, x_m, x_r, x_size)
        x_l = self.backbone.norm(x_l)
        x_m = self.backbone.norm(x_m)
        x_r = self.backbone.norm(x_r)
        x_l = self.backbone.patch_unembed(x_l, x_size)
        x_m = self.backbone.patch_unembed(x_m, x_size)
        x_r = self.backbone.patch_unembed(x_r, x_size)
        x_m = torch.concat([x_l, x_m, x_r], dim=1)
        x_m = self.backbone.conv_last(x_m)
        x_l = self.backbone.conv_lm(x_l)
        x_r = self.backbone.conv_lm(x_r)
        return x_m, x_l, x_r
