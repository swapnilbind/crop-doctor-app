import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define your class names directly here
class_names = [
    'Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy',
    'Blueberry_healthy', "Cherry(including_sour)Powdery_mildew", "Cherry(including_sour)healthy",
    'Corn(maize)Cercospora_leaf_spot Gray_leaf_spot', 'Corn(maize)Common_rust', 'Corn_(maize)Northern_Leaf_Blight',
    'Corn(maize)healthy', 'Grape_Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)',
    'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot', 'Peach_healthy',
    'Pepper,_bell_Bacterial_spot', 'Pepper,_bell_healthy', 'Potato_Early_blight', 'Potato_Late_blight',
    'Potato_healthy', 'Raspberry_healthy', 'Soybean_healthy', 'Squash_Powdery_mildew', 'Strawberry_Leaf_scorch',
    'Strawberry_healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites Two-spotted_spider_mite',
    'Tomato_Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus', 'Tomato__healthy'
]

@st.cache_resource
def load_model():
    import torchvision.models as models
    import torch.nn as nn
    num_classes = len(class_names)
    model = models.vit_b_16(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    for param in model.heads.head.parameters():
        param.requires_grad = True
    model.load_state_dict(torch.load('vit_plantdisease.pt', map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.title("Crop Doctor - Disease Prediction")
uploaded_file = st.file_uploader("Upload a crop image", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        result = class_names[predicted.item()]
    st.success(f"Prediction: {result}")
