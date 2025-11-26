import streamlit as st
from PIL import Image
import os
import sys
from pathlib import Path

# Add src to path so imports work cleanly
PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

# Import from pipelines package
from pipelines.inference_pipeline import inference_pipeline

# --- Page Configuration ---
st.set_page_config(
    page_title="Car Damage Estimator",
    page_icon="🚗",
    layout="wide"
)


# --- Load Model (Cached) ---
# We use @st.cache_resource so we don't reload the model on every click
@st.cache_resource
def get_model():
    # return load_model()
    return 0


model = get_model()

# --- Title and Header ---
st.title("🚗 AI Car Damage Cost Estimator")
st.markdown("""
Welcome! Upload a photo of the damaged vehicle and provide car details 
to get an instant repair cost estimate.
""")

st.write("---")

# --- Sidebar: Car Details ---
st.sidebar.header("Step 1: Car Details")

# Define available options (You can expand these lists)
car_makes = ["Toyota", "Ford", "BMW", "Honda", "Mercedes"]
car_years = list(range(2000, 2025))

selected_make = st.sidebar.selectbox("Select Make", car_makes)

# Dynamic models based on make (Simple logic for demo)
if selected_make == "Toyota":
    available_models = ["Corolla", "Camry", "RAV4"]
elif selected_make == "BMW":
    available_models = ["X5", "3 Series", "M3"]
else:
    available_models = ["Generic Model A", "Generic Model B"]

selected_model = st.sidebar.selectbox("Select Model", available_models)
selected_year = st.sidebar.selectbox("Select Year", car_years, index=len(car_years) - 1)

# --- Main Area: Image Upload ---
st.header("Step 2: Upload Image")
uploaded_file = st.file_uploader("Choose an image of the damage...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Display the image
    image = Image.open(uploaded_file)

    # Create two columns: one for image, one for results
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)

    with col2:
        st.subheader("Analysis & Report")

        # Button to trigger prediction
        if st.button("Generate Damage Report", type="primary"):
            # Organize input data
            car_info_str = f"{selected_make} {selected_model} {selected_year}"

            # Save uploaded image temporarily
            temp_dir = "temp_uploads"
            os.makedirs(temp_dir, exist_ok=True)
            temp_image_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner('🔍 Analyzing damage... This may take a few moments...'):
                try:
                    # Run the full inference pipeline
                    pdf_path = inference_pipeline([temp_image_path], car_info_str)
                    
                    if pdf_path and os.path.exists(pdf_path):
                        st.success("✅ Analysis Complete!")
                        
                        # Provide download button for PDF
                        with open(pdf_path, "rb") as pdf_file:
                            pdf_bytes = pdf_file.read()
                            
                        st.download_button(
                            label="📄 Download Full Damage Report (PDF)",
                            data=pdf_bytes,
                            file_name=f"damage_report_{selected_make}_{selected_model}_{selected_year}.pdf",
                            mime="application/pdf"
                        )
                        
                        st.info(f"Report generated for: {car_info_str}")
                        st.balloons()
                    else:
                        st.error("❌ No damage detected or report generation failed.")
                        
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.exception(e)
                
                finally:
                    # Cleanup temp file
                    if os.path.exists(temp_image_path):
                        try:
                            os.remove(temp_image_path)
                        except:
                            pass