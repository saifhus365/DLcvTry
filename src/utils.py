import csv
import torchvision
import json
import torchvision.transforms as T
import torch




def write_results_to_csv(file_path, train_losses):
    """
    Writes the training results to a CSV file.

    Args:
        file_path (str): Path to the CSV file where results will be saved. Without the postfix .csv
        train_losses (list): List of training losses.
    """
    with open(file_path + ".csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "loss"])  # Write the header
        for epoch, loss in enumerate(train_losses, start=1):
            writer.writerow([epoch, loss])


def get_transform(train=True):
    """
    Creates a torchvision transform pipeline for training and testing datasets. For training, augmentations
    such as horizontal flipping and random rotation can be included. For testing, only essential transformations
    like normalization and converting the image to a tensor are applied.

    Args:
        train (bool): Indicates whether the transform is for training or testing. If True, augmentations are applied.
        horizontal_flip_prob (float): Probability of applying a horizontal flip to the images. Effective only if train=True.
        rotation_degrees (float): The range of degrees for random rotation. Effective only if train=True.

    Returns:
        torchvision.transforms.Compose: Composed torchvision transforms for data preprocessing.
    """
    transforms = []
    transforms.append(T.ToTensor())

    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    transforms.append(
            T.Resize((3509, 2480)))

        # Example values for ImageNet normalization

    return T.Compose(transforms)


def write_results_to_json(file_path, dict):
    """
    Writes the training results to a csv file.

    Args:
        file_path (str): Path to the csv file where results will be saved. Without the postfix .json
        train_losses (list): List of training losses.

    """

    with open(file_path + ".json", mode='w') as file:
        json.dump(dict, file, indent=4)


def save_model(model, file_path):
    """
    Saves the trained model to a file.

    Args:
        model (torch.nn.Module): The trained PyTorch model to be saved.
        file_path (str): Path to the file where the model will be saved.
    """
    torch.save(model.state_dict(), file_path + ".pth")

def collate_fn(batch):
    return tuple(zip(*batch))






