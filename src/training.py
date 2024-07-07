import torch
import tqdm
from visualize import visualize_top_boxes
def train_one_epoch(model, data_loader, optimizer, device, num_classes):
    model.train()

    epoch_loss = 0.0
    for imgs, targets in tqdm.tqdm(data_loader):
        imgs = [img.to(device) for img in imgs]
        targets = [move_to_device(t, device) for t in targets]

        # Ensure bounding boxes are valid

        optimizer.zero_grad()
        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())

        losses.backward()
        optimizer.step()
        epoch_loss += losses.item()
    return epoch_loss / len(data_loader)

def evaluate_one_epoch(model, data_loader, device, coco):
    model.eval()
    results = []
    img_names = []
    with torch.no_grad():
        for imgs, targets in tqdm.tqdm(data_loader):
            imgs = [img.to(device) for img in imgs]
            outputs = model(imgs)
            for target, output in zip(targets, outputs):
                img_name = target['image_name']
                img_names.append(img_name)
                for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
                    score = score.item()
                    label = label.item()
                    result = {
                        'file_name': img_name,
                        'category_id': label,
                        'bbox': [box[0].item(), box[1].item(), box[2].item() - box[0].item(), box[3].item() - box[1].item()],
                        'score': score
                    }
                    results.append(result)

    return results

def train_and_evaluate_model(model, train_loader,val_loader, optimizer, num_epochs, device, cocoval, num_classes, scheduler=None, early_stopping=False):
    train_l = []
    test_l = []
    for epoch in range(num_epochs):
        print("\nepoch: " + str(epoch))
        train_loss = train_one_epoch(model, train_loader, optimizer, device, num_classes)
        train_l.append(train_loss)

        result = evaluate_one_epoch(model, val_loader, device, cocoval)
        test_l.append(result)

        if scheduler:
            scheduler.step()
        visualize_top_boxes('src/Images', epoch, result)

    return train_l, test_l

def move_to_device(target, device):
    if isinstance(target, dict):
        return {k: move_to_device(v, device) for k, v in target.items()}
    elif isinstance(target, list):
        return [move_to_device(v, device) for v in target]
    elif isinstance(target, torch.Tensor):
        return target.to(device)
    else:
        return target