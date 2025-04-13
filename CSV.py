import cv2
import os
import pandas as pd

# Initialize the list to store annotations
annotations = []

# Global variables for mouse callback
drawing = False  # True if the mouse is pressed
ix, iy = -1, -1  # Initial mouse coordinates
image_path = ''  # To hold the current image path
current_image = None  # To hold the current image being annotated

# Mouse callback function for drawing rectangles
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, annotations, current_image, image_path

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = current_image.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow('image', img_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(current_image, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow('image', current_image)
        # Store the annotation
        label = 'Object'  # You can change this to any label or ask for user input
        annotations.append({
            'image_id': len(annotations) + 1,  # Image ID starts from 1 and increases sequentially
            'filename': image_path,
            'label': label,
            'x': ix,
            'y': iy,
            'width': x - ix,
            'height': y - iy
        })
        print(f"Annotation saved: {image_path}, Label: {label}, x: {ix}, y: {iy}, width: {x - ix}, height: {y - iy}")

# Function to read all images from a directory and sort them by number
def read_images_from_directory(directory_path):
    image_paths = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if file_path.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # Add more formats if needed
            image_paths.append(file_path)
    # Sort the images numerically based on filename
    image_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    return image_paths

# Main function to annotate images one by one
def annotate_images(directory_path):
    global annotations, image_path, current_image

    # Read all image paths from the directory
    image_paths = read_images_from_directory(directory_path)

    for image_index, image_path in enumerate(image_paths, 1):  # Start numbering from 1
        # Load the current image
        current_image = cv2.imread(image_path)
        cv2.imshow('image', current_image)

        # Set the mouse callback function to draw rectangles
        cv2.setMouseCallback('image', draw_rectangle)

        # Wait for the user to annotate the image
        print(f"Annotating image {image_index}: {image_path}")
        cv2.waitKey(0)  # Wait for a key press to continue to the next image
        cv2.destroyAllWindows()  # Close the window after annotation

    # After annotating all images, save the annotations to CSV
    annotations_df = pd.DataFrame(annotations)
    annotations_df.to_csv('annotations.csv', index=False)
    print("Annotations saved to annotations.csv")

# Specify the directory path containing images
directory_path = 'E:/Project_Yelloskye'  # Replace with your directory path

# Start the annotation process
annotate_images(directory_path)
