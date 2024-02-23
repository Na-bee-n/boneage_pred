# This Program Resize the image into shape (299,299)

from PIL import Image
import os

def resize_images(input_folder, output_folder, new_size=(224, 224)):
    # check whether the output folder exists ?
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        # Check if the file is a valid image file
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Open and resize the image
            with Image.open(input_path) as img:
                resized_img = img.resize(new_size, Image.BICUBIC)

                # Save the resized image to the output folder
                resized_img.save(output_path)


if __name__ == "__main__":
    # Specify the input and output folders
    input_folder = "D:/dataset/boneage-test-dataset/boneage-validation-dataset/"
    output_folder = "D:/dataset/resize_image/boneage-validation-dataset/"

    # Specify the new size for the images
    new_size = (299,299)

    # Resize the images in the input folder and save them to the output folder
    resize_images(input_folder, output_folder, new_size)