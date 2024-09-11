"""Create an Image Classification Web App using PyTorch and Streamlit."""

# import libraries
from PIL import Image
from torchvision import models, transforms
import torch
import streamlit as st
import re

# set title of app
st.title("Artifact Explorer")
st.info("Discover Stories Behind Every Artifact!")

# enable users to upload images for the model to make predictions
file_up = st.file_uploader("Upload an image of an artifact", type=["jpg", "jpeg", "png"])

def predict(image):
    """Return the top prediction ranked by highest probability.

    Parameters
    ----------
    :param image: uploaded image
    :type image: file-like object
    :rtype: tuple
    :return: top prediction ranked by highest probability
    """
    # create a ResNet model
    resnet = models.resnet101(pretrained=True)

    # transform the input image through resizing, normalization
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    # load the image, pre-process it, and make predictions
    img = Image.open(image).convert('RGB')
    batch_t = torch.unsqueeze(transform(img), 0)
    resnet.eval()
    with torch.no_grad():
        out = resnet(batch_t)

    # load ImageNet class names
    try:
        with open('imagenet_classes.txt') as f:
            classes = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        st.error("Class file 'imagenet_classes.txt' not found.")
        return None

    # return the top prediction ranked by highest probability
    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    top_index = indices[0][0].item()
    top_class = classes[top_index]
    top_prob = prob[top_index].item()

    # Remove numerical tags if they are present
    top_class = re.sub(r'^\d+\s*', '', top_class).strip()

    return top_class, top_prob

if file_up is not None:
    try:
        # display image that user uploaded
        image = Image.open(file_up).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("We are scanning your artifact!")
        result = predict(file_up)

        if result:
            # Remove leading comma or unwanted characters
            artifact_name = result[0].lstrip(',').strip()
            st.write(f"Your artifact is: {artifact_name} with a confidence of {result[1]:.2f}%")
    except Exception as e:
        st.error(f"An error occurred: {e}")
