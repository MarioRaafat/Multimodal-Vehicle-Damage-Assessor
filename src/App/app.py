import streamlit as st
from PIL import Image
# from model_inference import load_model, predict_damage_cost

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
        st.subheader("Estimation")

        # Button to trigger prediction
        if st.button("Calculate Repair Cost", type="primary"):
            # Organize input data
            car_info = {
                "make": selected_make,
                "model": selected_model,
                "year": selected_year
            }

            with st.spinner('Analyzing damage patterns...'):
                # Call the function from our other file
                # cost = predict_damage_cost(model, image, car_info)
                cost = 500

            # Display Result
            st.success("Analysis Complete!")
            st.metric(label="Estimated Repair Cost", value=f"${cost}")

            st.info(f"Note: This estimate is for a {selected_year} {selected_make} {selected_model}.")