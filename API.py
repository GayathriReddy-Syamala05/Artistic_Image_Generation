from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


def load_and_process_image(image_path, resize_shape=None):
    """
    Loads an image using PIL, converts it to float32, normalizes, and optionally resizes.
    """
    try:
        # Open the image and ensure it's RGB
        image = Image.open(image_path).convert("RGB")

        # Convert to NumPy array and normalize to [0, 1]
        image = np.array(image).astype(np.float32) / 255.0

        # Expand dimensions to match model input (batch size of 1)
        image = np.expand_dims(image, axis=0)

        # Resize if specified
        if resize_shape:
            image = tf.image.resize(image, resize_shape)

        return image

    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def transfer_style(content_image_path, style_image_path, model_path):
    """
    Performs Neural Style Transfer using a pre-trained TF model.
    """
    print("Loading images...")
    
    # Resize content image to reduce memory usage
    content_image = load_and_process_image(content_image_path, resize_shape=(512, 512))
    style_image = load_and_process_image(style_image_path, resize_shape=(256, 256))

    # Ensure images are loaded correctly
    if content_image is None or style_image is None:
        raise ValueError("Error: One of the images failed to load. Check file paths and formats.")

    print("Loading pre-trained model...")
    hub_module = hub.load(model_path)

    print("Generating stylized image now... please wait.")
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    
    stylized_image = outputs[0].numpy()
    print("Stylizing completed.")

    return np.squeeze(stylized_image, axis=0)  # Remove batch dimension
