from torchvision.models.detection import FCOS
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def get_fcos_model(num_classes, backbone_name="resnet50"):
    if backbone_name == "resnet50":
        backbone = resnet_fpn_backbone('resnet50', pretrained = True)
    elif backbone_name == "resnet34":
        backbone = resnet_fpn_backbone('resnet34', pretrained= True)
    elif backbone_name == "resnet18":
        backbone = resnet_fpn_backbone('resnet18', pretrained= True)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    model = FCOS(
        backbone,
        num_classes=num_classes,
    )

    return model