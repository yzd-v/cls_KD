from .mgd import  MGDLoss
from .dkd import  DKDLoss
from .nkd import  NKDLoss
from .srrl import  SRRLLoss
from .wsld import  WSLDLoss
from .kd import KDLoss
from .vitkd import ViTKDLoss
from .uskd import USKDLoss

__all__ = [
    'MGDLoss', 'DKDLoss', 'NKDLoss', 'SRRLLoss', 'WSLDLoss', 'KDLoss', 'ViTKDLoss', 'USKDLoss'
]
