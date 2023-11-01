from models.backbones.cnn.dn4.dn4_cnn import BaselineBackbone2d
from models.backbones.cnn.dn4.dn4am_cnn import DN4_AM
from models.backbones.cnn.dn7.dn7_cnn import SevenLayer_64F
from models.backbones.cnn.npdn4.npair_dn4 import DN4_MCNP
from models.backbones.cnn.npdn4.sn4am_fc import B_4L64F_AM_MCNP
from models.backbones.resnet.resnet12 import ResNetBackbone2d
from models.backbones.cnn.npdn4.sn4_fc import B_4L64F_MCNP

__all__ = {
    'DN4_CNN2d': BaselineBackbone2d,
    'DN7_CNN2d': SevenLayer_64F,
    'DN4_AM_CNN2d': DN4_AM,
    'ResNet2d': ResNetBackbone2d,
    'B_4L64F_MCNP': B_4L64F_MCNP,
    'DN4_MCNP': DN4_MCNP,
    'B_4L64F_AM_MCNP': B_4L64F_AM_MCNP
}
