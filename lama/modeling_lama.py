import abc
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from .configuration_lama import LamaConfig


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def get_activation(kind="tanh"):
    if kind == "tanh":
        return nn.Tanh()
    if kind == "sigmoid":
        return nn.Sigmoid()
    if kind is False:
        return nn.Identity()
    raise ValueError(f"Unknown activation kind {kind}")


def make_constant_area_crop_params(
    img_height, img_width, min_size=128, max_size=512, area=256 * 256, round_to_mod=16
):
    min_size = min(img_height, img_width, min_size)
    max_size = min(img_height, img_width, max_size)
    if random.random() < 0.5:
        out_height = min(
            max_size, ceil_modulo(random.randint(min_size, max_size), round_to_mod)
        )
        out_width = min(max_size, ceil_modulo(area // out_height, round_to_mod))
    else:
        out_width = min(
            max_size, ceil_modulo(random.randint(min_size, max_size), round_to_mod)
        )
        out_height = min(max_size, ceil_modulo(area // out_width, round_to_mod))

    start_y = random.randint(0, img_height - out_height)
    start_x = random.randint(0, img_width - out_width)
    return (start_y, start_x, out_height, out_width)


def make_constant_area_crop_batch(batch, **kwargs):
    crop_y, crop_x, crop_height, crop_width = make_constant_area_crop_params(
        img_height=batch["image"].shape[2], img_width=batch["image"].shape[3], **kwargs
    )
    batch["image"] = batch["image"][
        :, :, crop_y : crop_y + crop_height, crop_x : crop_x + crop_width
    ]
    batch["mask"] = batch["mask"][
        :, :, crop_y : crop_y + crop_height, crop_x : crop_x + crop_width
    ]
    return batch


def make_multiscale_noise(base_tensor, scales=6, scale_mode="bilinear"):
    batch_size, _, height, width = base_tensor.shape
    cur_height, cur_width = height, width
    result = []
    align_corners = False if scale_mode in ("bilinear", "bicubic") else None
    for _ in range(scales):
        cur_sample = torch.randn(
            batch_size, 1, cur_height, cur_width, device=base_tensor.device
        )
        cur_sample_scaled = F.interpolate(
            cur_sample,
            size=(height, width),
            mode=scale_mode,
            align_corners=align_corners,
        )
        result.append(cur_sample_scaled)
        cur_height //= 2
        cur_width //= 2
    return torch.cat(result, dim=1)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        res = x * y.expand_as(x)
        return res


class FFCSE_block(nn.Module):
    def __init__(self, channels, ratio_g):
        super(FFCSE_block, self).__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(channels, channels // r, kernel_size=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_a2l = (
            None
            if in_cl == 0
            else nn.Conv2d(channels // r, in_cl, kernel_size=1, bias=True)
        )
        self.conv_a2g = (
            None
            if in_cg == 0
            else nn.Conv2d(channels // r, in_cg, kernel_size=1, bias=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x = self.avgpool(x)
        x = self.relu1(self.conv1(x))

        x_l = 0 if self.conv_a2l is None else id_l * self.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * self.sigmoid(self.conv_a2g(x))
        return x_l, x_g


class FourierUnit(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        groups=1,
        spatial_scale_factor=None,
        spatial_scale_mode="bilinear",
        spectral_pos_encoding=False,
        use_se=False,
        se_kwargs=None,
        ffc3d=False,
        fft_norm="ortho",
    ):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(
            in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.groups,
            bias=False,
        )
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

        # squeeze and excitation block
        self.use_se = use_se
        if use_se:
            if se_kwargs is None:
                se_kwargs = {}
            self.se = SELayer(self.conv_layer.in_channels, **se_kwargs)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(
                x,
                scale_factor=self.spatial_scale_factor,
                mode=self.spatial_scale_mode,
                align_corners=False,
            )

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view(
            (
                batch,
                -1,
            )
            + ffted.size()[3:]
        )

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = (
                torch.linspace(0, 1, height)[None, None, :, None]
                .expand(batch, 1, height, width)
                .to(ffted)
            )
            coords_hor = (
                torch.linspace(0, 1, width)[None, None, None, :]
                .expand(batch, 1, height, width)
                .to(ffted)
            )
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        if self.use_se:
            ffted = self.se(ffted)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = (
            ffted.view(
                (
                    batch,
                    -1,
                    2,
                )
                + ffted.size()[2:]
            )
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(
            ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm
        )

        if self.spatial_scale_factor is not None:
            output = F.interpolate(
                output,
                size=orig_size,
                mode=self.spatial_scale_mode,
                align_corners=False,
            )

        return output


class SpectralTransform(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        groups=1,
        enable_lfu=True,
        **fu_kwargs,
    ):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels // 2, kernel_size=1, groups=groups, bias=False
            ),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(
                torch.split(x[:, : c // 4], split_s, dim=-2), dim=1
            ).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1), dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


class FFC(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        ratio_gin,
        ratio_gout,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        enable_lfu=True,
        padding_type="reflect",
        gated=False,
        **spectral_kwargs,
    ):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        # groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        # groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(
            in_cl,
            out_cl,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode=padding_type,
        )
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(
            in_cl,
            out_cg,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode=padding_type,
        )
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(
            in_cg,
            out_cl,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode=padding_type,
        )
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg,
            out_cg,
            stride,
            1 if groups == 1 else groups // 2,
            enable_lfu,
            **spectral_kwargs,
        )

        self.gated = gated
        module = (
            nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        )
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        ratio_gin,
        ratio_gout,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        norm_layer=nn.BatchNorm2d,
        activation_layer=nn.Identity,
        padding_type="reflect",
        enable_lfu=True,
        **kwargs,
    ):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(
            in_channels,
            out_channels,
            kernel_size,
            ratio_gin,
            ratio_gout,
            stride,
            padding,
            dilation,
            groups,
            bias,
            enable_lfu,
            padding_type=padding_type,
            **kwargs,
        )
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class FFCResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        padding_type,
        norm_layer,
        activation_layer=nn.ReLU,
        dilation=1,
        spatial_transform_kwargs=None,
        inline=False,
        **conv_kwargs,
    ):
        super().__init__()
        self.conv1 = FFC_BN_ACT(
            dim,
            dim,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            padding_type=padding_type,
            **conv_kwargs,
        )
        self.conv2 = FFC_BN_ACT(
            dim,
            dim,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
            padding_type=padding_type,
            **conv_kwargs,
        )
        if spatial_transform_kwargs is not None:
            self.conv1 = LearnableSpatialTransformWrapper(
                self.conv1, **spatial_transform_kwargs
            )
            self.conv2 = LearnableSpatialTransformWrapper(
                self.conv2, **spatial_transform_kwargs
            )
        self.inline = inline

    def forward(self, x):
        if self.inline:
            x_l, x_g = (
                x[:, : -self.conv1.ffc.global_in_num],
                x[:, -self.conv1.ffc.global_in_num :],
            )
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out


class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        assert isinstance(x, tuple)
        x_l, x_g = x
        assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)


class FFCResNetGenerator(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        n_downsampling=3,
        n_blocks=9,
        norm_layer=nn.BatchNorm2d,
        padding_type="reflect",
        activation_layer=nn.ReLU,
        up_norm_layer=nn.BatchNorm2d,
        up_activation=nn.ReLU(True),
        init_conv_kwargs={},
        downsample_conv_kwargs={},
        resnet_conv_kwargs={},
        spatial_transform_layers=None,
        spatial_transform_kwargs={},
        add_out_act=True,
        max_features=1024,
        out_ffc=False,
        out_ffc_kwargs={},
        **kwargs,
    ):
        assert n_blocks >= 0
        super().__init__()

        model = [
            nn.ReflectionPad2d(3),
            FFC_BN_ACT(
                input_nc,
                ngf,
                kernel_size=7,
                padding=0,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                **init_conv_kwargs,
            ),
        ]

        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs["ratio_gout"] = resnet_conv_kwargs.get("ratio_gin", 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [
                FFC_BN_ACT(
                    min(max_features, ngf * mult),
                    min(max_features, ngf * mult * 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    **cur_conv_kwargs,
                )
            ]

        mult = 2**n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        ### resnet blocks
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(
                feats_num_bottleneck,
                padding_type=padding_type,
                activation_layer=activation_layer,
                norm_layer=norm_layer,
                **resnet_conv_kwargs,
            )
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(
                    cur_resblock, **spatial_transform_kwargs
                )
            model += [cur_resblock]

        model += [ConcatTupleLayer()]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    min(max_features, ngf * mult),
                    min(max_features, int(ngf * mult / 2)),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                up_norm_layer(min(max_features, int(ngf * mult / 2))),
                up_activation,
            ]

        if out_ffc:
            model += [
                FFCResnetBlock(
                    ngf,
                    padding_type=padding_type,
                    activation_layer=activation_layer,
                    norm_layer=norm_layer,
                    inline=True,
                    **out_ffc_kwargs,
                )
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
        ]
        if add_out_act:
            model.append(get_activation("tanh" if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class BaseDiscriminator(nn.Module):
    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Predict scores and get intermediate activations. Useful for feature matching loss
        :return tuple (scores, list of intermediate activations)
        """
        raise NotImplemented()


class NLayerDiscriminator(BaseDiscriminator):
    def __init__(
        self,
        input_nc,
        ndf=64,
        n_layers=3,
        norm_layer=nn.BatchNorm2d,
        **kwargs,
    ):
        super().__init__()
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [
            [
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True),
            ]
        ]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)

            cur_model = []
            cur_model += [
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True),
            ]
            sequence.append(cur_model)

        nf_prev = nf
        nf = min(nf * 2, 512)

        cur_model = []
        cur_model += [
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True),
        ]
        sequence.append(cur_model)

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        for n in range(len(sequence)):
            setattr(self, "model" + str(n), nn.Sequential(*sequence[n]))

    def get_all_activations(self, x):
        res = [x]
        for n in range(self.n_layers + 2):
            model = getattr(self, "model" + str(n))
            res.append(model(res[-1]))
        return res[1:]

    def forward(self, x):
        act = self.get_all_activations(x)
        return act[-1], act[:-1]


class FFCNLayerDiscriminator(BaseDiscriminator):
    def __init__(
        self,
        input_nc,
        ndf=64,
        n_layers=3,
        norm_layer=nn.BatchNorm2d,
        max_features=512,
        init_conv_kwargs={},
        conv_kwargs={},
    ):
        super().__init__()
        self.n_layers = n_layers

        def _act_ctor(inplace=True):
            return nn.LeakyReLU(negative_slope=0.2, inplace=inplace)

        kw = 3
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [
            [
                FFC_BN_ACT(
                    input_nc,
                    ndf,
                    kernel_size=kw,
                    padding=padw,
                    norm_layer=norm_layer,
                    activation_layer=_act_ctor,
                    **init_conv_kwargs,
                )
            ]
        ]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, max_features)

            cur_model = [
                FFC_BN_ACT(
                    nf_prev,
                    nf,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    norm_layer=norm_layer,
                    activation_layer=_act_ctor,
                    **conv_kwargs,
                )
            ]
            sequence.append(cur_model)

        nf_prev = nf
        nf = min(nf * 2, 512)

        cur_model = [
            FFC_BN_ACT(
                nf_prev,
                nf,
                kernel_size=kw,
                stride=1,
                padding=padw,
                norm_layer=norm_layer,
                activation_layer=lambda *args, **kwargs: nn.LeakyReLU(
                    *args, negative_slope=0.2, **kwargs
                ),
                **conv_kwargs,
            ),
            ConcatTupleLayer(),
        ]
        sequence.append(cur_model)

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        for n in range(len(sequence)):
            setattr(self, "model" + str(n), nn.Sequential(*sequence[n]))

    def get_all_activations(self, x):
        res = [x]
        for n in range(self.n_layers + 2):
            model = getattr(self, "model" + str(n))
            res.append(model(res[-1]))
        return res[1:]

    def forward(self, x):
        act = self.get_all_activations(x)
        feats = []
        for out in act[:-1]:
            if isinstance(out, tuple):
                if torch.is_tensor(out[1]):
                    out = torch.cat(out, dim=1)
                else:
                    out = out[0]
            feats.append(out)
        return act[-1], feats


class BaseAdversarialLoss:
    def pre_generator_step(
        self,
        real_batch: torch.Tensor,
        fake_batch: torch.Tensor,
        generator: nn.Module,
        discriminator: nn.Module,
    ):
        """
        Prepare for generator step
        :param real_batch: Tensor, a batch of real samples
        :param fake_batch: Tensor, a batch of samples produced by generator
        :param generator:
        :param discriminator:
        :return: None
        """

    def pre_discriminator_step(
        self,
        real_batch: torch.Tensor,
        fake_batch: torch.Tensor,
        generator: nn.Module,
        discriminator: nn.Module,
    ):
        """
        Prepare for discriminator step
        :param real_batch: Tensor, a batch of real samples
        :param fake_batch: Tensor, a batch of samples produced by generator
        :param generator:
        :param discriminator:
        :return: None
        """

    def generator_loss(
        self,
        real_batch: torch.Tensor,
        fake_batch: torch.Tensor,
        discr_real_pred: torch.Tensor,
        discr_fake_pred: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate generator loss
        :param real_batch: Tensor, a batch of real samples
        :param fake_batch: Tensor, a batch of samples produced by generator
        :param discr_real_pred: Tensor, discriminator output for real_batch
        :param discr_fake_pred: Tensor, discriminator output for fake_batch
        :param mask: Tensor, actual mask, which was at input of generator when making fake_batch
        :return: total generator loss along with some values that might be interesting to log
        """
        raise NotImplemented()

    def discriminator_loss(
        self,
        real_batch: torch.Tensor,
        fake_batch: torch.Tensor,
        discr_real_pred: torch.Tensor,
        discr_fake_pred: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate discriminator loss and call .backward() on it
        :param real_batch: Tensor, a batch of real samples
        :param fake_batch: Tensor, a batch of samples produced by generator
        :param discr_real_pred: Tensor, discriminator output for real_batch
        :param discr_fake_pred: Tensor, discriminator output for fake_batch
        :param mask: Tensor, actual mask, which was at input of generator when making fake_batch
        :return: total discriminator loss along with some values that might be interesting to log
        """
        raise NotImplemented()

    def interpolate_mask(self, mask, shape):
        assert mask is not None
        assert self.allow_scale_mask or shape == mask.shape[-2:]
        if shape != mask.shape[-2:] and self.allow_scale_mask:
            if self.mask_scale_mode == "maxpool":
                mask = F.adaptive_max_pool2d(mask, shape)
            else:
                mask = F.interpolate(mask, size=shape, mode=self.mask_scale_mode)
        return mask


def make_r1_gp(discr_real_pred, real_batch):
    if torch.is_grad_enabled():
        grad_real = torch.autograd.grad(
            outputs=discr_real_pred.sum(), inputs=real_batch, create_graph=True
        )[0]
        grad_penalty = (
            grad_real.view(grad_real.shape[0], -1).norm(2, dim=1) ** 2
        ).mean()
    else:
        grad_penalty = 0
    real_batch.requires_grad = False

    return grad_penalty


class NonSaturatingWithR1(BaseAdversarialLoss):
    def __init__(
        self,
        gp_coef=5,
        weight=1,
        mask_as_fake_target=False,
        allow_scale_mask=False,
        mask_scale_mode="nearest",
        extra_mask_weight_for_gen=0,
        use_unmasked_for_gen=True,
        use_unmasked_for_discr=True,
        **kwargs,
    ):
        self.gp_coef = gp_coef
        self.weight = weight
        # use for discr => use for gen;
        # otherwise we teach only the discr to pay attention to very small difference
        assert use_unmasked_for_gen or (not use_unmasked_for_discr)
        # mask as target => use unmasked for discr:
        # if we don't care about unmasked regions at all
        # then it doesn't matter if the value of mask_as_fake_target is true or false
        assert use_unmasked_for_discr or (not mask_as_fake_target)
        self.use_unmasked_for_gen = use_unmasked_for_gen
        self.use_unmasked_for_discr = use_unmasked_for_discr
        self.mask_as_fake_target = mask_as_fake_target
        self.allow_scale_mask = allow_scale_mask
        self.mask_scale_mode = mask_scale_mode
        self.extra_mask_weight_for_gen = extra_mask_weight_for_gen

    def generator_loss(
        self,
        real_batch: torch.Tensor,
        fake_batch: torch.Tensor,
        discr_real_pred: torch.Tensor,
        discr_fake_pred: torch.Tensor,
        mask=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        fake_loss = F.softplus(-discr_fake_pred)
        if (
            (self.mask_as_fake_target and self.extra_mask_weight_for_gen > 0)
            or not self.use_unmasked_for_gen
        ):  # == if masked region should be treated differently
            mask = self.interpolate_mask(mask, discr_fake_pred.shape[-2:])
            if not self.use_unmasked_for_gen:
                fake_loss = fake_loss * mask
            else:
                pixel_weights = 1 + mask * self.extra_mask_weight_for_gen
                fake_loss = fake_loss * pixel_weights

        return fake_loss.mean() * self.weight, dict()

    def pre_discriminator_step(
        self,
        real_batch: torch.Tensor,
        fake_batch: torch.Tensor,
        generator: nn.Module,
        discriminator: nn.Module,
    ):
        real_batch.requires_grad = True

    def discriminator_loss(
        self,
        real_batch: torch.Tensor,
        fake_batch: torch.Tensor,
        discr_real_pred: torch.Tensor,
        discr_fake_pred: torch.Tensor,
        mask=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        real_loss = F.softplus(-discr_real_pred)
        grad_penalty = make_r1_gp(discr_real_pred, real_batch) * self.gp_coef
        fake_loss = F.softplus(discr_fake_pred)

        if not self.use_unmasked_for_discr or self.mask_as_fake_target:
            # == if masked region should be treated differently
            mask = self.interpolate_mask(mask, discr_fake_pred.shape[-2:])
            # use_unmasked_for_discr=False only makes sense for fakes;
            # for reals there is no difference beetween two regions
            fake_loss = fake_loss * mask
            if self.mask_as_fake_target:
                fake_loss = fake_loss + (1 - mask) * F.softplus(-discr_fake_pred)

        sum_discr_loss = real_loss + grad_penalty + fake_loss
        metrics = dict(
            discr_real_out=discr_real_pred.mean(),
            discr_fake_out=discr_fake_pred.mean(),
            discr_real_gp=grad_penalty,
        )
        return sum_discr_loss.mean(), metrics


class BCELoss(BaseAdversarialLoss):
    def __init__(self, weight):
        self.weight = weight
        self.bce_loss = nn.BCEWithLogitsLoss()

    def generator_loss(
        self, discr_fake_pred: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        real_mask_gt = torch.zeros(discr_fake_pred.shape).to(discr_fake_pred.device)
        fake_loss = self.bce_loss(discr_fake_pred, real_mask_gt) * self.weight
        return fake_loss, dict()

    def pre_discriminator_step(
        self,
        real_batch: torch.Tensor,
        fake_batch: torch.Tensor,
        generator: nn.Module,
        discriminator: nn.Module,
    ):
        real_batch.requires_grad = True

    def discriminator_loss(
        self,
        mask: torch.Tensor,
        discr_real_pred: torch.Tensor,
        discr_fake_pred: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        real_mask_gt = torch.zeros(discr_real_pred.shape).to(discr_real_pred.device)
        sum_discr_loss = (
            self.bce_loss(discr_real_pred, real_mask_gt)
            + self.bce_loss(discr_fake_pred, mask)
        ) / 2
        metrics = dict(
            discr_real_out=discr_real_pred.mean(),
            discr_fake_out=discr_fake_pred.mean(),
            discr_real_gp=0,
        )
        return sum_discr_loss, metrics


class LamaPretrainedModel(PreTrainedModel):
    """https://github.com/advimman/lama/blob/main/saicinpainting/training/trainers/base.py#L57"""

    def __init__(self, config: LamaConfig) -> None:
        super().__init__(config)
        self.config = config
        self.generator = self.make_generator()

    def make_generator(self):
        """https://github.com/advimman/lama/blob/main/saicinpainting/training/modules/__init__.py#L7-L19"""
        if self.config.generator.kind == "pix2pixhd_multidilated":
            raise NotImplementedError

        elif self.config.generator.kind == "pix2pixhd_global":
            raise NotImplementedError

        elif self.config.generator.kind == "ffc_resnet":
            return FFCResNetGenerator(**self.config.generator.to_dict())

        else:
            raise ValueError(f"Unknown generator: {self.config.generator_kind}")


class LamaModel(LamaPretrainedModel):
    """https://github.com/advimman/lama/blob/main/saicinpainting/training/trainers/default.py#L26"""

    def __init__(self, config: LamaConfig) -> None:
        super().__init__(config)
        self.discriminator = self.make_discriminator()
        self.adversarial_loss = self.make_discrim_loss()

    def make_discriminator(self):
        """https://github.com/advimman/lama/blob/main/saicinpainting/training/modules/__init__.py#L22-L31"""
        if self.config.discriminator.kind == "pix2pixhd_nlayer_multidilated":
            raise NotImplementedError

        elif self.config.discriminator.kind == "pix2pixhd_nlayer":
            return NLayerDiscriminator(**self.config.discriminator.to_dict())

        else:
            raise ValueError(f"Unknown discriminator: {self.config.discriminator_kind}")

    def make_discrim_loss(self):
        """https://github.com/advimman/lama/blob/main/saicinpainting/training/losses/adversarial.py#L172-L177"""
        if self.config.discrim_loss.kind == "r1":
            return NonSaturatingWithR1(**self.config.discrim_loss.to_dict())

        elif self.config.discrim_loss.kind == "bce":
            return BCELoss(**self.config.discrim_loss.to_dict())

        else:
            raise ValueError(
                f"Unknown discriminator loss: {self.config.discrim_loss_kind}"
            )

    def forward(self, batch):
        if self.training and self.rescale_size_getter is not None:
            cur_size = self.rescale_size_getter(self.global_step)
            batch["image"] = F.interpolate(
                batch["image"], size=cur_size, mode="bilinear", align_corners=False
            )
            batch["mask"] = F.interpolate(batch["mask"], size=cur_size, mode="nearest")

        if self.training and self.const_area_crop_kwargs is not None:
            batch = make_constant_area_crop_batch(batch, **self.const_area_crop_kwargs)

        img = batch["image"]
        mask = batch["mask"]

        masked_img = img * (1 - mask)

        if self.add_noise_kwargs is not None:
            noise = make_multiscale_noise(masked_img, **self.add_noise_kwargs)
            if self.noise_fill_hole:
                masked_img = masked_img + mask * noise[:, : masked_img.shape[1]]
            masked_img = torch.cat([masked_img, noise], dim=1)

        if self.concat_mask:
            masked_img = torch.cat([masked_img, mask], dim=1)

        batch["predicted_image"] = self.generator(masked_img)
        batch["inpainted"] = (
            mask * batch["predicted_image"] + (1 - mask) * batch["image"]
        )

        if self.fake_fakes_proba > 1e-3:
            if self.training and torch.rand(1).item() < self.fake_fakes_proba:
                batch["fake_fakes"], batch["fake_fakes_masks"] = self.fake_fakes_gen(
                    img, mask
                )
                batch["use_fake_fakes"] = True
            else:
                batch["fake_fakes"] = torch.zeros_like(img)
                batch["fake_fakes_masks"] = torch.zeros_like(mask)
                batch["use_fake_fakes"] = False

        batch["mask_for_losses"] = (
            self.refine_mask_for_losses(img, batch["predicted_image"], mask)
            if self.refine_mask_for_losses is not None and self.training
            else mask
        )

        return batch
