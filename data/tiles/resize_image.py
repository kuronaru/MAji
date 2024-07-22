import os

from PIL import Image

input_directory = "./"
output_directory = input_directory
target_resolution = (40, 60)


def resize_images(input_dir, output_dir, target_res):
    if (not os.path.exists(output_dir)):
        os.makedirs(output_dir)

    for item in os.listdir(input_dir):
        directory = os.path.join(input_dir, item)
        if (os.path.isdir(directory)):
            for filename in os.listdir(directory):
                if filename.endswith((".png", ".jpg", ".jpeg")):
                    input_path = os.path.join(input_dir, directory, filename)
                    output_path = os.path.join(output_dir, directory, filename)

                    with Image.open(input_path) as img:
                        resized_img = img.resize(target_res, Image.BILINEAR)
                        resized_img.save(output_path)


if __name__ == '__main__':
    resize_images(input_directory, output_directory, target_resolution)
