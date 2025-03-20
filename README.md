**Creating Digital Art with the Power of Artificial Intelligence!🎨🖌**
Consider yourself an artist struggling for inspiration. You admire Van Gogh's distinctive style but can't quite capture his magic. That's where our technology steps in.
Using Generative Adversarial Networks, we blend your creative vision with Van Gogh's artistic genius. It's simple:
Your idea (content image) + Van Gogh's style (style image) = Your painting in Van Gogh's style
Our design analyzes both images - capturing what makes your composition unique while learning Van Gogh's brushstrokes, color palette, and textures. The result? Your artistic vision expressed through Van Gogh's eyes.

<div align="center">
  <img src="/Imgs/website.gif" width="90%"/>
</div>
</br>

Image style transfer is a computer vision technique that involves applying the artistic style of one image (the style image) to another image (the content image) while preserving the structure of the content image. This technique is based on deep learning and leverages convolutional neural networks (CNNs), particularly pre-trained models like VGG-19.

<br> <!-- line break -->

<div align="center">
<img src="C:\Users\gayat\Downloads\A Couple Walking in an Autumn Forest.jpg"/>
</div>

<br> <!-- line break -->


🎯 Objective 
This project aims to explore Neural Style Transfer through hands-on implementation. We will create a Neural Style Transfer model using TensorFlow and Keras. At the end of this project, we will deploy the model as a web application so that users can easily create digital artwork, which could even be used as NFTs.

📝 Summary of Neural Style Transfer
Style Transfer is a computer vision approach where two images—one representing the content and the other representing the style—are combined. The output image retains the fundamental content of the first image but appears in the style of the second image. The process typically involves training a neural network with two essential components:
1) A pre-trained feature extractor
2) A style transfer network


<div align="center">
<img src="C:\Users\gayat\Downloads\art-4.jpg" width="75%"/>
</div>

<br> <!-- line break -->


The key idea behind Neural Style Transfer is to optimize the output image to minimize both content loss and style loss. Content loss ensures that the essential features of the content image are preserved, while style loss ensures the output mimics the style of the reference image.


<div align="center">
<img src="/Imgs/final_oss.png" width="50%" />
</div>

<br> <!-- line break -->




## 👨‍💻 Implementation

In the past, Neural Style Transfer required an extensive number of iterations to apply a style to a single image. However, researchers have developed an optimized approach called Fast Neural Style Transfer. This approach uses a trained model to transform any image in a single, fast feed-forward pass, dramatically reducing the time and computational resources required.

In this project, we use a pre-trained "Generative Adversarial Network"—a Fast-NST architecture that was trained on a collection of over 80,000 paintings. The model generalizes well, even for paintings that weren’t in the training set.

How to Run Locally

1. Download the pre-trained TF model.

2. Clone the Repository:
```
git clone https://github.com/deepeshdm/Neural-Style-Transfer.git
```
3. Install all the required dependencies inside a virtual environment
```
pip install -r requirements.txt
```
4. Copy the below code snippet and pass the required variable values
 ~~~ python

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
~~~


Web Interface & API

In order to make it easy for anyone to interact with the model,we created a clean web interface using Streamlit and deployed it on their official cloud space.

- Checkout Official Website : https://share.streamlit.io/deepeshdm/pixelmix/main/App.py
- Website Repository : [here](https://github.com/deepeshdm/PixelMix)

<div align="center">
  <img src="/Imgs/website.gif" width="90%"/>
</div>


## 🖼🖌 Some of the art we created in this project

<div align="center">
  <img src="/Imgs/content1.jpg" width="35%"/>
<img src="/Imgs/art1.png" width="35%"/>
</div>

<div align="center">
<img src="/Imgs/content2.jpg" width="35%"/>
<img src="/Imgs/art2.png" width="35%"/>
</div>

<div align="center">
<img src="/Imgs/content3.jpg" width="35%"/>
<img src="/Imgs/art3.png" width="35%"/>
</div>

<div align="center">
<img src="/Imgs/content4.jpg" width="35%"/>
<img src="/Imgs/art4.png" width="35%"/>
</div>

References
Neural Style Transfer Paper
Keras Neural Style Transfer Example
Generative Adversarial Networks














