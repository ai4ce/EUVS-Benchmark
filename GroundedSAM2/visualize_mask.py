import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# Set the paths for the input image and mask
img_path = "original_img.jpg"
mask_path = "mask.png"

# Load the original image
image = np.array(Image.open(img_path))

# Load the mask
mask = np.array(Image.open(mask_path))
mask = (mask > 0).astype(np.uint8) 

# Expand the dimensions of the mask
if mask.ndim == 2:
    mask = np.expand_dims(mask, axis=-1)

# Apply the mask to the image
masked_image = image * mask

# Convert the masked image back
masked_image_pil = Image.fromarray((masked_image.astype(np.uint8)))

# Visualize masked image
plt.figure(figsize=(50,15))

# Show the original image
plt.subplot(1,3,1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

# Show the mask
plt.subplot(1, 3, 2)
plt.imshow(mask.squeeze(), cmap="gray")
plt.title("Mask")
plt.axis("off")

# Show the masked image
plt.subplot(1, 3, 3)
plt.imshow(masked_image)
plt.title("Masked Image")
plt.axis("off")

# Display the plots
plt.tight_layout()
plt.show()