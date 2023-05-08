import sys
from PIL import Image
import os
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt




def main():

    tensors = []

    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    names = sys.argv[1]

    # Open the file for reading
    with open(names, 'r') as f:
        # Read the contents of the file into a list of strings
        filenames = f.read().splitlines()

        for filename in filenames:
            with Image.open(os.path.join("UTKFace_test", filename)) as img:
                tens = transform(img)
                tensors.append(tens)
    
    grid = torchvision.utils.make_grid(tensors[64:80], nrow=4)
    plt.imshow(grid.permute(1, 2, 0))
    plt.savefig("first16.jpg")

    grid = torchvision.utils.make_grid(tensors[128:144], nrow=4)
    plt.imshow(grid.permute(1, 2, 0))
    plt.savefig("second16.jpg")

    grid = torchvision.utils.make_grid(tensors[32:48], nrow=4)
    plt.imshow(grid.permute(1, 2, 0))
    plt.savefig("third16.jpg")

    grid = torchvision.utils.make_grid(tensors[48:64], nrow=4)
    plt.imshow(grid.permute(1, 2, 0))
    plt.savefig("fourth16.jpg")

    grid = torchvision.utils.make_grid(tensors, nrow=28)
    plt.imshow(grid.permute(1, 2, 0))
    plt.savefig("all.jpg")
            

   
# -----------------------------------
if __name__ == '__main__':
    main()