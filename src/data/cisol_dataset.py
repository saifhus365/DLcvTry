import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

class CISOLDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        self.images, self.annotations = self.load_data()



    def load_data(self):
        annotations_file = os.path.join(self.root, 'annotations', f'{self.split}.json')


        # Check if the annotations file exists
        if not os.path.exists(annotations_file):
            image_fldr = os.path.join(self.root, 'images', self.split)
            images = {}
            annotations = {}

            # Assuming images are in a directory and you want to iterate through them
            for filename in os.listdir(image_fldr):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    images[filename] = {
                        'file_name': filename,
                    }
                    annotations[filename] = []  # Placeholder for annotations, since none are provided

            return images, annotations

        with open(annotations_file, 'r') as f:
            data = json.load(f)

        images = {img['file_name']: img for img in data['images']}
        annotations = {img['file_name']: [] for img in data['images']}

        for ann in data['annotations']:
            image_name = [img['file_name'] for img in data['images'] if img['id'] == ann['image_id']][0]
            annotations[image_name].append(ann)

        return images, annotations

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = list(self.images.keys())[idx]
        img_data = self.images[image_name]
        annotations = self.annotations.get(image_name, [])  # Get annotations or empty list if not available

        image_path = os.path.join(self.root, 'images', self.split, image_name)

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")  # Debugging statement
            return self.__getitem__((idx + 1) % len(self))  # Skip the missing image and try the next one

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        targets = self.prepare_targets(annotations, image_name)

        return image, targets

    def prepare_targets(self, annotations, image_name):
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
                    x_max = min(x_max, self.images[image_name]['width'])
                    y_max = min(y_max, self.images[image_name]['height'])
                    if x_max > x_min and y_max > y_min:
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(ann['category_id'])

        if len(boxes) == 0:  # No annotations found
            # Return default targets (empty tensors or None)
            targets = {
                'boxes': torch.empty((0, 4), dtype=torch.float32),
                'labels': torch.empty((0,), dtype=torch.int64),
                'image_name': image_name
            }
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

            targets = {
                'boxes': boxes,
                'labels': labels,
                'image_name': image_name
            }

        return targets
