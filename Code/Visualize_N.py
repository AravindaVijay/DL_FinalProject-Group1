import matplotlib.pyplot as plt
import numpy as np

def display_image_with_caption(image_tensor, caption, image_info=None):
    """
    Display an image with its corresponding caption and optional image information.

    Args:
        image_tensor (torch.Tensor): Tensor of the image to be displayed.
        caption (str): Caption text for the image.
        image_info (dict, optional): Additional info about the image, e.g., filename or ID. Default is None.
    """
    # Denormalize image
    image = image_tensor[0].cpu().permute(1, 2, 0).numpy()
    image = (image * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]  # De-normalize
    image = np.clip(image, 0, 1)

    # Format the title
    title = f"Caption: {caption}"
    if image_info:
        if 'file_name' in image_info:
            title += f"\nFilename: {image_info['file_name']}"
        if 'id' in image_info:
            title += f"\nImage ID: {image_info['id']}"

    # Display image with caption
    plt.imshow(image)
    plt.axis('off')
    plt.title(title, fontsize=10)
    plt.show()
