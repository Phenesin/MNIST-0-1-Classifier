from torchvision import transforms
import numpy as np
from PIL import Image


def preprocess_image(image, device):

    image_array = np.array(image)
    coords = np.argwhere(image_array > 0)
    if(len(coords) > 0):
        y_min, x_min = coords.min(axis = 0)
        y_max, x_max = coords.max(axis = 0)

        image_array = image_array[y_min : y_max + 1, x_min : x_max + 1]

    cropped_image = Image.fromarray(image_array)
    width, height = cropped_image.size

    if width > height:
        new_width = 20
        new_height = int((height / width) *20)
    else:
        new_height = 20
        new_width = int((height / width) *20)

    final_image = Image.new("L", (28, 28), 0)
    paste_x = (28 - new_width) // 2
    paste_y = (28 - new_height) // 2
    transform = transforms.ToTensor()
    tensor = transform(final_image)
    tensor = tensor.view(1, 784)
    tensor = tensor.to(device)
    return tensor, final_image