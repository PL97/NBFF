from torchvision.utils import make_grid
import torch
import numpy as np
import matplotlib.pyplot as plt

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

def visualize_stn(name, model, test_loader, device):
    model = model.to(device)
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0][:64].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = convert_image_np(
            make_grid(input_tensor))

        out_grid = convert_image_np(
            make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')
        
        plt.savefig(f"figs/{name}")