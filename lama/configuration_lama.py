from typing import Any, Dict, Literal, Optional

from transformers import PretrainedConfig

GeneratorKind = Literal[
    "pix2pixhd_multidilated",
    "pix2pixhd_global",
    "ffc_resnet",
]
DiscriminatorKind = Literal[
    "pix2pixhd_nlayer_multidilated",
    "pix2pixhd_nlayer",
]
DiscrimLossKind = Literal["r1", "bce"]


class LamaGeneratorConfig(PretrainedConfig):
    def __init__(
        self,
        kind: GeneratorKind = "ffc_resnet",
        input_nc: int = 4,
        output_nc: int = 3,
        ngf: int = 64,
        n_downsampling: int = 3,
        n_blocks: int = 18,
        add_out_act: str = "sigmoid",
        init_conv_kwargs: Optional[Dict[str, Any]] = None,
        downsample_conv_kwargs: Optional[Dict[str, Any]] = None,
        resnet_conv_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.kind = kind
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_downsampling = n_downsampling
        self.n_blocks = n_blocks
        self.add_out_act = add_out_act

        self.init_conv_kwargs = init_conv_kwargs or {
            "ratio_gin": 0,
            "ratio_gout": 0,
            "enable_lfu": False,
        }
        self.downsample_conv_kwargs = downsample_conv_kwargs or {
            "ratio_gin": self.init_conv_kwargs["ratio_gout"],
            "ratio_gout": self.init_conv_kwargs["ratio_gout"],
            "enable_lfu": False,
        }
        self.resnet_conv_kwargs = resnet_conv_kwargs or {
            "ratio_gin": 0.75,
            "ratio_gout": 0.75,
            "enable_lfu": False,
        }


class LamaDiscriminatorConfig(PretrainedConfig):
    def __init__(
        self,
        kind: DiscriminatorKind = "pix2pixhd_nlayer",
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.kind = kind
        self.input_nc = input_nc
        self.ndf = ndf
        self.n_layers = n_layers


class LamaDiscrimLossConfig(PretrainedConfig):
    def __init__(
        self,
        kind: DiscrimLossKind = "r1",
        weight: int = 10,
        gp_coef: float = 0.001,
        mask_as_fake_target: bool = True,
        allow_scale_mask: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.kind = kind
        self.weight = weight
        self.gp_coef = gp_coef
        self.mask_as_fake_target = mask_as_fake_target
        self.allow_scale_mask = allow_scale_mask


class LamaConfig(PretrainedConfig):
    is_composition: bool = True

    def __init__(
        self,
        generator: LamaGeneratorConfig = LamaGeneratorConfig(),
        discriminator: LamaDiscriminatorConfig = LamaDiscriminatorConfig(),
        discrim_loss: LamaDiscrimLossConfig = LamaDiscrimLossConfig(),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.discrim_loss = discrim_loss
