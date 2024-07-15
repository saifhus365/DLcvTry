from PIL import Image, ImageDraw, ImageFont
import imageio
import os
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
# Define the directory containing the images
image_directory = 'Images/vid_images'

# Collect all image file names in the directory
images = [os.path.join(image_directory, file) for file in os.listdir(image_directory) if file.lower().endswith(('png', 'jpg', 'jpeg'))]

# Sort the images if necessary (assuming they are named in a way that sorting by name puts them in the correct order)
images.sort(key=lambda x: natural_sort_key(os.path.basename(x)))

# Define the font and size (adjust path and size if necessary)

font = ImageFont.truetype("Roboto-Medium.ttf", 30)  # Increase font size for larger text


# List to store frames
frames = []

# Iterate through images, adding filenames
for image_path in images:
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    filename = os.path.basename(image_path)
    match = re.search(r'\d+', filename)

    epoch_number = match.group()
    text_to_display = f"Epoch {str(int(epoch_number) + 1)}"

    text_position = (340, 10)  # Adjust position as needed

    # Get the bounding box of the text to be added
    text_bbox = draw.textbbox(text_position, filename, font=font)
    text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])

    # Draw a rectangle behind the text for better visibility
    draw.rectangle([text_position, (text_position[0] + text_size[0], text_position[1] + text_size[1])], fill="white")

    # Add text on top of the rectangle
    draw.text(text_position, text_to_display, font=font, fill="black")

    frames.append(image)

# Save frames as a gif with a 1-second delay
output_path = 'output.gif'
frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=1000, loop=0)  # 1000 ms = 1 second

print("GIF created successfully")
