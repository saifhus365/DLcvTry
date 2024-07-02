# Updated get_model function
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models import resnet50, resnet34, resnet18, resnet101, resnext50_32x4d, resnext101_32x8d, mobilenet_v2
import torch

def get_model(num_classes, backbone_name):
    # Load a pre-trained backbone
    if backbone_name == "resnet50":
        backbone = resnet50(pretrained=True)
    elif backbone_name == "resnet34":
        backbone = resnet34(pretrained=True)
    elif backbone_name == "resnet18":
        backbone = resnet18(pretrained=True)
    elif backbone_name == "resnet101":
        backbone = resnet101(pretrained=True)
    elif backbone_name == "resnext50_32x4d":
        backbone = resnext50_32x4d(pretrained=True)
    elif backbone_name == "resnext101_32x8d":
        backbone = resnext101_32x8d(pretrained=True)
    elif backbone_name == "mobilenet_v2":
        backbone = mobilenet_v2(pretrained=True)


    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    backbone = torch.nn.Sequential(*(list(backbone.children())[:-2]))
    backbone.out_channels = 2048 if backbone_name in ["resnet50", "resnet101", "resnext50_32x4d", "resnext101_32x8d"] else 512

    # Create the anchor generator for the FPN
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    # Feature Pyramid Network (FPN) needs to know the number of output channels in each feature map.
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )

    return model

def prepare_targets(self, annotations, img_id):
        boxes = []
        labels = []

        for ann in annotations:
            bbox = ann['bbox']
            if bbox:
                x_min, y_min, width, height = bbox
                x_max, y_max = x_min + width, y_min + height
                # Ensure the coordinates are within the image bounds and have positive area
                if width > 0 and height > 0:
                    x_min = max(x_min, 0)
                    y_min = max(y_min, 0)
                    x_max = min(x_max, self.images[img_id]['width'])
                    y_max = min(y_max, self.images[img_id]['height'])
                    if x_max > x_min and y_max > y_min:
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(ann['category_id'])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        targets = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id])
        }

        return targets