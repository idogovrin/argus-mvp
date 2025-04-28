import streamlit as st
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from ultralytics import YOLO
import numpy as np
import os
import uuid
import csv

st.set_page_config(page_title="Try ARGUS", page_icon="🔍")

# === Initialize session state keys ===
for key in ["latest_crop", "latest_score", "show_dropdown", "feedback_type"]:
    if key not in st.session_state:
        st.session_state[key] = None

# === Constants ===
FEEDBACK_DIR = "feedback"
FEEDBACK_CSV = os.path.join(FEEDBACK_DIR, "feedback_log.csv")
os.makedirs(FEEDBACK_DIR, exist_ok=True)
if not os.path.exists(FEEDBACK_CSV):
    with open(FEEDBACK_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "ref_image", "search_image", "match_crop", "score", "feedback", "explanation"])

# === Load models ===
resnet = resnet18(pretrained=True)
resnet.eval()
yolo = YOLO("yolov8m.pt")

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === Feature extraction ===
def extract_features(img):
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = resnet(img_tensor).squeeze()
    return features

def crop_box(image, box):
    x1, y1, x2, y2 = map(int, box)
    return image.crop((x1, y1, x2, y2))

def find_best_match(ref_vec, search_image, search_boxes):
    best_score = -1
    best_box = None
    best_crop = None
    for box in search_boxes:
        crop = crop_box(search_image, box)
        crop_vec = extract_features(crop)
        score = torch.nn.functional.cosine_similarity(ref_vec, crop_vec, dim=0).item()
        if score > best_score:
            best_score = score
            best_box = box
            best_crop = crop
    return best_box, best_score, best_crop

def draw_box_on_image(image, box, score):
    x1, y1, x2, y2 = map(int, box)
    draw = ImageDraw.Draw(image)
    if score >= 0.95:
        color = "green"
    elif score >= 0.85:
        color = "yellow"
    else:
        color = "red"
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    draw.text((x1, max(y1 - 10, 0)), f"{int(score * 100)}%", fill=color)

# === UI ===
st.title("Try ARGUS")
st.markdown("Upload a **Reference Image** and a **Search Image**. ARGUS will analyze and return the most similar object and a verification score.")

uploaded_ref = st.file_uploader("Upload Reference Image", type=["jpg", "png"], key="ref")
uploaded_search = st.file_uploader("Upload Search Image", type=["jpg", "png"], key="search")

if uploaded_ref and uploaded_search:
    ref_image = Image.open(uploaded_ref).convert("RGB")
    search_image_original = Image.open(uploaded_search).convert("RGB")
    search_image = search_image_original.copy()

    st.image(ref_image, caption="Reference Image", use_container_width=True)
    image_placeholder = st.empty()
    image_placeholder.image(search_image_original, caption="Search Image (before verification)", use_container_width=True)

    if st.button("Run Verification"):
        st.session_state["show_dropdown"] = False
        st.session_state["feedback_type"] = None

        ref_results = yolo(ref_image)
        ref_boxes = ref_results[0].boxes.xyxy.cpu().numpy()
        if len(ref_boxes) == 0:
            st.error("No object detected in reference image.")
            st.stop()
        ref_crop = crop_box(ref_image, ref_boxes[0])
        ref_vec = extract_features(ref_crop)

        search_results = yolo(search_image)
        search_boxes = search_results[0].boxes.xyxy.cpu().numpy()
        if len(search_boxes) == 0:
            st.error("No objects detected in search image.")
            st.stop()

        best_box, best_score, best_crop = find_best_match(ref_vec, search_image, search_boxes)
        st.session_state["latest_crop"] = best_crop
        st.session_state["latest_score"] = best_score
        draw_box_on_image(search_image, best_box, best_score)

        st.success(f"Best Match Verification Score: {int(best_score * 100)}%")
        image_placeholder.image(search_image, caption="Search Image (with match highlighted)", use_container_width=True)

    if st.session_state["latest_crop"] is not None and st.session_state["latest_score"] is not None:
        st.markdown("### Was this match correct?")

        feedback_type = st.radio("Choose feedback type:", ["✅ Correct", "⚠️ Borderline", "❌ Wrong"])
        if st.button("Next"):
            if "Correct" in feedback_type:
                st.session_state["feedback_type"] = "correct"
                st.session_state["show_dropdown"] = False
            elif "Borderline" in feedback_type:
                st.session_state["feedback_type"] = "borderline"
                st.session_state["show_dropdown"] = True
            elif "Wrong" in feedback_type:
                st.session_state["feedback_type"] = "wrong"
                st.session_state["show_dropdown"] = True

    if st.session_state["feedback_type"] == "correct" and st.button("Submit Feedback"):
        feedback_id = str(uuid.uuid4())
        ref_path = os.path.join(FEEDBACK_DIR, f"{feedback_id}_ref.jpg")
        search_path = os.path.join(FEEDBACK_DIR, f"{feedback_id}_search.jpg")
        match_path = os.path.join(FEEDBACK_DIR, f"{feedback_id}_match.jpg")
        ref_image.save(ref_path)
        search_image_original.save(search_path)
        st.session_state["latest_crop"].save(match_path)
        with open(FEEDBACK_CSV, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                feedback_id,
                ref_path,
                search_path,
                match_path,
                st.session_state["latest_score"],
                "correct",
                ""
            ])
        st.success("📝 Feedback saved as 'CORRECT'")
        st.session_state["feedback_type"] = None

    if st.session_state["show_dropdown"] and st.session_state["feedback_type"] in ["borderline", "wrong"]:
        if st.session_state["feedback_type"] == "borderline":
            explanation = st.selectbox("Why is this borderline?", [
                "Right match but low score",
                "Partial overlap",
                "Wrong match but very low score",
                "Other"])
        elif st.session_state["feedback_type"] == "wrong":
            explanation = st.selectbox("Why is this wrong?", [
                "Wrong object",
                "Too similar visually",
                "Score misleading",
                "Different object type",
                "Other"])

        if st.button("Submit Feedback"):
            feedback_id = str(uuid.uuid4())
            ref_path = os.path.join(FEEDBACK_DIR, f"{feedback_id}_ref.jpg")
            search_path = os.path.join(FEEDBACK_DIR, f"{feedback_id}_search.jpg")
            match_path = os.path.join(FEEDBACK_DIR, f"{feedback_id}_match.jpg")
            ref_image.save(ref_path)
            search_image_original.save(search_path)
            st.session_state["latest_crop"].save(match_path)
            with open(FEEDBACK_CSV, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    feedback_id,
                    ref_path,
                    search_path,
                    match_path,
                    st.session_state["latest_score"],
                    st.session_state["feedback_type"],
                    explanation
                ])
            st.success(f"📝 Feedback saved as '{st.session_state['feedback_type'].upper()}' — Reason: {explanation}")
            st.session_state["show_dropdown"] = False
            st.session_state["feedback_type"] = None
