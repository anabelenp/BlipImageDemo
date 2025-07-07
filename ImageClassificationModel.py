import torch
import requests
from PIL import Image
from torchvision import transforms

import gradio as gr

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).eval()

# Download human-readable labels for ImageNet.
response=requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

# After processing the tensor through the model,
# it returns the predictions in the form of a dictionary named confidences.
def predict(inp):
    #Converts the input image to a PIL image and then to a PyTorch tensor and adds a batch dimension.
    #inp: the input image as a PIL image
    inp = transforms.ToTensor()(inp).unsqueeze(0)

    # Disables gradient calculation for inference (saves memory and computation).
    with torch.no_grad():
        #Passes the image through the model, applies softmax to get class probabilities.
        # softmax function is crucial because it converts the raw output logits from the model,
        # which can be any real number, into probabilities that sum up to 1
        prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)

        #Creates a dictionary mapping each label to its predicted probability.
        confidences = {labels[i]: float(prediction[i]) for i in range(1000)}

    #Returns the dictionary of class probabilities.
    return confidences

gr.Interface(fn=predict,
       inputs=gr.Image(type="pil"),
       outputs=gr.Label(num_top_classes=3),
       examples=["ana-desk.jpeg", "choclo.jpg"]).launch()


