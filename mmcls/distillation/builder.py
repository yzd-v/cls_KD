from mmcv.utils import Registry, build_from_cfg
from torch import nn

DISTILLER = Registry('distiller')
DISTILL_LOSSES = Registry('distill_loss')
DISRUNNERS = Registry('runner')

def build(cfg, registry, default_args=None):
    """Build a module.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """

    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)

def build_distill_loss(cfg):
    """Build distill loss."""
    return build(cfg, DISTILL_LOSSES)

def build_distiller(cfg,teacher_cfg=None,student_cfg=None):
    """Build distiller."""

    return build(cfg, DISTILLER, dict(teacher_cfg=teacher_cfg,student_cfg=student_cfg))


def build_runner(cfg, default_args=None):
    return build_from_cfg(cfg, DISRUNNERS, default_args=default_args)
