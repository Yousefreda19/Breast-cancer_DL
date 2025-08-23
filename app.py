import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

# ---- Model Setup ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)  # مش عايزين pretrained هنا لو أنت حافظت weights
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)      # عدد الكلاسات = 2
model.load_state_dict(torch.load("resnet_weights1.pth", map_location=device))
model = model.to(device)
model.eval()

# ---- Streamlit UI ----
st.title("Breast Cancer  Classifier")
st.write("Upload an image to predict if it's benign or malignant or normal.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)

    
    # ---- Preprocessing ----
    transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

    img = transform(image).unsqueeze(0).to(device)
   
    with torch.no_grad():
      outputs = model(img)
      _, preds = torch.max(outputs, 1)

    classes = ['benign', 'malignant', 'normal']
    st.write(f"Prediction : {classes[preds.item()]}")
#http://localhost:8501/