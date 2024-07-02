
import os
import torch
from models.faster_rcnn import get_model
from models.FCOS import get_fcos_model
from models.retina_net import get_retinanet_model
from training import train_and_evaluate_model, evaluate_one_epoch
from utils import write_results_to_csv, write_results_to_json, collate_fn, get_transform, save_model
from torch.utils.data import DataLoader
from data.cisol_dataset import CISOLDataset
from pycocotools.coco import COCO

def train_model(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.NO_CUDA else "cpu")
    print(device)
    data_root = cfg.DATA_ROOT
    train_transform = get_transform(train=True)
    test_transform = get_transform(train=False)
    coco = COCO(os.path.join(data_root, 'annotations/train.json'))

    dataset = CISOLDataset(root=data_root,
                           split='train', transform=train_transform)
    dataset_test = CISOLDataset(root=data_root,
                                split='test', transform=test_transform)
    dataset_val = CISOLDataset(root=data_root,
                                split='val', transform=test_transform)
    # Initialize data loaders

    train_loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(dataset_test, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(dataset_val, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    if cfg.MODEL == 'faster_rcnn':
        model = get_model(num_classes=cfg.NUM_CLASSES, backbone_name=cfg.BACKBONE).to(device)
    elif cfg.MODEL == 'fcos':
        model = get_fcos_model(num_classes=cfg.NUM_CLASSES, backbone_name=cfg.BACKBONE).to(device)
    elif cfg.MODEL == 'retina_net':
        model = get_retinanet_model(num_classes=cfg.NUM_CLASSES, backbone_name=cfg.BACKBONE).to(device)




    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.BASE_LR,
        weight_decay=cfg.WEIGHT_DECAY
    )

    steps_per_epoch = len(train_loader)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.BASE_LR,  # Max learning rate
        steps_per_epoch=steps_per_epoch,
        epochs=cfg.EPOCHS
    )

    train_losses, test_losses = train_and_evaluate_model(
        model, train_loader, val_loader, optimizer, cfg.EPOCHS, device, coco,num_classes=cfg.NUM_CLASSES , scheduler=scheduler,
        early_stopping=cfg.DO_EARLY_STOPPING
    )
    json_obj = evaluate_one_epoch(model, test_loader, device, coco)
    write_results_to_json(cfg.RUN_NAME, json_obj)

    write_results_to_csv(cfg.RUN_NAME, train_losses)
    save_model(model, cfg.RUN_NAME)