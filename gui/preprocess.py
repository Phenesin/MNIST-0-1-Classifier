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

    image = Image.fromarray(image_array)
    image = image.resize((28, 28))
    final_image = Image.new("L", (28, 28), 0)
    final_image.paste(image, (4, 4))
    transform = transforms.ToTensor()
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)
    return tensor, final_image