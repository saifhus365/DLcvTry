import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import pandas as pd


def visualize_top_boxes(image_root, epoch, annotations, num_images=10, save_directory='.'):
    # Create the save directory if it does not exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Group annotations by image
    annotations_by_image = {}
    for ann in annotations:
        if ann['file_name'] not in annotations_by_image:
            annotations_by_image[ann['file_name']] = []
        annotations_by_image[ann['file_name']].append(ann)

    # Take the first num_images images
    selected_images = list(annotations_by_image.keys())[:num_images]

    # Iterate through the selected images and annotations
    for idx, image_name in enumerate(selected_images):
        # Construct full image path
        image_path = os.path.join(image_root, image_name)

        # Open the image
        img = Image.open(image_path)

        # Create figure and axes
        fig, ax = plt.subplots(1)

        # Display the image
        ax.imshow(img)

        # Sort annotations by score
        annotations_sorted = sorted(annotations_by_image[image_name], key=lambda x: x['score'], reverse=True)

        # Add bounding boxes for top 20 scores
        for ann in annotations_sorted[:10]:
            bbox = ann['bbox']
            x, y, w, h = bbox

            # Convert x, y, w, h to x_min, y_min, x_max, y_max
            x_min = x
            y_min = y
            x_max = x + w
            y_max = y + h

            # Create a Rectangle patch with yellow color
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='yellow', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)

            # Add label in red color
            category_id = ann['category_id']
            score = ann['score']
            ax.text(x_min, y_min - 5, f'Category: {category_id}, Score: {score:.2f}', fontsize=8, color='red')

        # Show plot for the current image
        plt.axis('off')  # Turn off axis
        plt.title(f"Image {idx + 1}/{num_images}")

        # Save the plot
        save_path = os.path.join(save_directory, f"{os.path.splitext(image_name)[0]}_preds_{epoch}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()  # Close the plot to free memory
def plot_training_losses(csv_files, labels, title='Training Losses', xlabel='Epoch', ylabel='Loss'):
    """
    Plots training losses from multiple CSV files.

    Parameters:
    csv_files (list of str): List of paths to the CSV files.
    labels (list of str): List of labels for the different models.
    title (str): Title of the plot.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    save_path (str): Path to save the plot image. If None, the plot will be displayed.
    """
    plt.figure(figsize=(10, 6))

    for csv_file, label in zip(csv_files, labels):
        # Read CSV file
        data = pd.read_csv(csv_file)

        # Ensure the CSV contains 'epoch' and 'loss' columns
        if 'epoch' not in data.columns or 'loss' not in data.columns:
            raise ValueError(f"CSV file {csv_file} must contain 'epoch' and 'loss' columns")

        # Plot the training losses
        plt.plot(data['epoch'], data['loss'], label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)


    plt.savefig('Images/TrainLosses.png', dpi=300)

'''
import os
import matplotlib.pyplot as plt
from PIL import Image

# Define directories for each model
directories = {
    'FasterRCNN': 'Images/FasterRCNN',
    'FCOS': 'Images/FCOS',
    'RetinaNet': 'Images/RetinaNet'
}

# Initialize list to hold image paths
image_paths = {'FasterRCNN': [], 'FCOS': [], 'RetinaNet': []}

# Load image paths from each directory
for model, dir_path in directories.items():
    for file_name in os.listdir(dir_path):
        if file_name.endswith('.png'):
            image_paths[model].append(os.path.join(dir_path, file_name))
    # Sort image paths to ensure consistent order
    image_paths[model] = sorted(image_paths[model])

# Set up the plot
fig, axes = plt.subplots(10, 3, figsize=(15, 50))
fig.tight_layout(pad=3.0)

# Plot images
for idx in range(10):
    for col, model in enumerate(['FasterRCNN', 'FCOS', 'RetinaNet']):
        ax = axes[idx, col]
        img_path = image_paths[model][idx]
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(f'{model} - Image {idx + 1}')
        ax.axis('off')

# Save the plot as a high-resolution image
plt.savefig('Images/combined_plot.png', dpi=300)
plt.show()


# Define the mAP values and labels
map_list = [54.81280263925944, 29.388947867224775, 42.12001653932782]
labels = ['FasterRCNN', 'FCOS', 'RetinaNet']

# Create the bar chart
plt.figure(figsize=(10, 6))
plt.bar(labels, map_list, color=['blue', 'green', 'red'])
plt.xlabel('Model')
plt.ylabel('mAP')
plt.title('mAP values for different models')
plt.ylim(0, 60)  # Adjust the y-axis limit if needed
plt.grid(axis='y')

# Save the plot to the Images directory
output_dir = 'Images'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'mAP_values.png')
plt.savefig(output_path)
plt.close()'''