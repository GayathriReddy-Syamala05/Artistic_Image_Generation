import matplotlib.pyplot as plt
import numpy as np
from API import transfer_style

if __name__ == "__main__":  # Correcting this line
    model_path = r"C:\GAN\Hi\Hlo\model"
    content_image_path = r"C:\GAN\Hi\Hlo\Imgs\content1.jpg"
    style_image_path = r"C:\GAN\Hi\Hlo\Imgs\art3.png"

    # Transfer the style
    img = transfer_style(content_image_path, style_image_path, model_path)

    # Ensure correct shape for saving and display
    img = np.clip(img, 0, 1)  # Ensure values are in range [0, 1]

    output_path = r"C:\GAN\Hi\Hlo\stylized_image.jpg"
    plt.imsave(output_path, img)

    print(f"Stylized image saved at: {output_path}")

    # Display the image
    plt.imshow(img)
    plt.axis("off")  # Hide axes
    plt.show()
