import streamlit as st
st.set_page_config(
    page_title="ARGUS",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Apply dark styling via markdown CSS hack
st.markdown("""
    <style>
        body {
            background-color: black;
            color: white;
        }
        .stApp {
            background-color: black;
            color: white;
        }
        h1, h2, h3, h4, h5, h6 {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18

# Load ResNet model
resnet = resnet18(pretrained=True)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_features(img):
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = resnet(img_tensor).squeeze()
    return features

st.title("ARGUS - Target Verification Demo")

st.markdown("Upload a **Target Image** (Reference Object) and a **Search Image** (Scene with multiple objects). ARGUS will return the most similar object with a verification score.")

uploaded_ref = st.file_uploader("Upload Reference Image", type=["jpg", "png"], key="ref")
uploaded_search = st.file_uploader("Upload Search Image", type=["jpg", "png"], key="search")

if uploaded_ref and uploaded_search:
    ref_image = Image.open(uploaded_ref).convert("RGB")
    search_image = Image.open(uploaded_search).convert("RGB")

    st.image(ref_image, caption="Reference Image", use_column_width=True)
    st.image(search_image, caption="Search Image", use_column_width=True)

    if st.button("Run Verification"):
        st.info("Running object matching (demo logic only)...")

        # Extract features (placeholder logic)
        ref_vec = extract_features(ref_image)
        search_vec = extract_features(search_image)

        # Cosine similarity
        score = torch.nn.functional.cosine_similarity(ref_vec, search_vec, dim=0).item()

        st.success(f"Verification Score: {score:.4f}")
        st.markdown("(This demo does not yet detect or crop multiple objects. That will come next.)")
