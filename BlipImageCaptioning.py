# Install the transformers library
#!pip install transformers Pillow torch torchvision torchaudio (brew install)
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

#Initilize the processor and model from Hugging Face
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", use_fast=True)
model =BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

#load am image
image = Image.open("ana-desk.jpeg")

#Prepare the image
#This line processes the input image using the BLIP processor,
# converting it into a format suitable for the model.
# The parameter return_tensors="pt" specifies that the output should be PyTorch tensors.
# The result, inputs, contains the processed image data ready for model inference.
inputs = processor(image, return_tensors="pt")

#Generate captions
outputs = model.generate(**inputs)
caption = processor.decode(outputs[0],skip_special_tokens=True)

print("Generated caption:", caption)