import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="ARGUS | AI Target Verification",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Apply dark styling and tactical font
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
    html, body, [class*="css"] {
        font-family: 'Share Tech Mono', monospace;
        background-color: black;
        color: white;
    }
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("ARGUS")
st.subheader("AI-Powered Target Verification for VISINT Operations")

st.markdown("""
ARGUS is an AI-powered web platform designed to help visual platform operators make faster, safer, and more accurate real-time decisions. During high-pressure visual intelligence (VISINT) missions, one wrong identification can cost lives. ARGUS reduces human error by comparing a known reference image of a target (e.g., a vehicle or person) to live drone or sensor images and delivers a **verification score** - showing how likely two objects are a match.
""")

st.image("demo_flow.png", caption="Example: Reference image → Live VISINT frame → ARGUS verification output", use_column_width=True)

st.markdown("### 🔍 How It Works")
st.markdown("""
1. **Detect** the object in the reference image  
2. **Search** for matching objects in the live VISINT frame  
3. **Compare** the visual features using deep learning  
4. **Score** the best match and visualize it  
""")

st.markdown("### ⚡ MVP Capabilities")
st.markdown("""
- Upload target image + live sensor image  
- Automatic detection & matching  
- Returns verification score of top match  
""")

st.markdown("### 🚀 Try it now")
st.page_link("pages/2_Verification_Tool.py", label="Go to ARGUS Verification Tool")

st.markdown("---")
st.caption("Developed by Ido Govrin | idogovrin2@gmail.com")
