import torch
from torchvision.models import resnet50, resnet18, resnet34, resnet101
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, BackboneWithFPN, \
    _validate_trainable_layers, _resnet_fpn_extractor
from torchvision.ops import  FrozenBatchNorm2d
from torchvision.ops.feature_pyramid_network import LastLevelP6P7






def get_retinanet_model(num_classes, backbone_name="resnet50"):
    backbone = extract_trainable_layers(backbone_name)

    model = RetinaNet(
        num_classes=num_classes,
        backbone=backbone
    )

    return model
def extract_trainable_layers(backbone_name):
    trainable_backbone_layers = None

    is_trained = True
    trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)
    norm_layer = FrozenBatchNorm2d if is_trained else torch.nn.BatchNorm2d

    progress = True

    # Load the ResNet-18 backbone with FPN
    if backbone_name == "resnet50":
        backbone = resnet50( pretrained=True, progress=progress, norm_layer=norm_layer)
    elif backbone_name == "resnet34":
        backbone = resnet34( pretrained=True, progress=progress, norm_layer=norm_layer)
    elif backbone_name == "resnet18":
        backbone = resnet18( pretrained=True, progress=progress, norm_layer=norm_layer)
    elif backbone_name == "resnet101":
        backbone = resnet101( pretrained=True, progress=progress, norm_layer=norm_layer)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    backbone = _resnet_fpn_extractor(
        backbone, trainable_backbone_layers, returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256)
    )

    return backbone

