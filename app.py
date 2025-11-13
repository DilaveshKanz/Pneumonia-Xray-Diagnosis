import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import os
import warnings
import sys
import kagglehub

# --- 1. Configuration ---
MODEL_PATH = 'pneumonia_VGG16_best.keras'
IMG_HEIGHT = 150
IMG_WIDTH = 150
CONFIDENCE_THRESHOLD = 0.90 # 90%
COLOR_THRESHOLD = 15 

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# --- 2. Find Dataset Path (for examples) ---
print("Finding dataset path for examples...")
try:
    dataset_root_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    base_dir = os.path.join(dataset_root_path, "chest_xray")
    print(f"Dataset found at: {base_dir}")
except Exception:
    base_dir = 'chest_xray' # Fallback
    print(f"Warning: Could not use kagglehub. Assuming dataset is in local folder: '{base_dir}'")
    
# --- 3. Load the Trained Model ---
print(f"Loading model from {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    print("Please run 'train_and_evaluate.py' first to generate the model file.")
    sys.exit(1)
    
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# --- 4. Define Helper and Prediction Functions ---
def is_color_image(img_bgr, threshold=COLOR_THRESHOLD):
    """Checks if an image is color."""
    gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    rebuilt_bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    diff = np.abs(img_bgr.astype("float") - rebuilt_bgr.astype("float"))
    mean_diff = np.mean(diff)
    print(f"Image color difference: {mean_diff}") # For debugging
    return mean_diff > threshold

def predict_image(img):
    """Takes a NumPy image, validates, and predicts."""
    try:
        # 1. Validate the image
        if is_color_image(img):
            return "Error: This is a color image. Please upload a grayscale X-ray."
        
        # 2. Preprocess the image
        img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        if len(img_resized.shape) == 2:
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        img_normalized = img_rgb / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)

        # 3. Make prediction
        prediction = model.predict(img_batch, verbose=0) 
        probability = prediction[0][0] # Probability of PNEUMONIA

        # 4. Filter the prediction
        if probability > CONFIDENCE_THRESHOLD:
            return f"Prediction: PNEUMONIA (Confidence: {probability * 100:.2f}%)"
        elif probability < (1.0 - CONFIDENCE_THRESHOLD):
            return f"Prediction: NORMAL (Confidence: {(1.0 - probability) * 100:.2f}%)"
        else:
            return "Error: The model is not confident. The image may be low quality or invalid."

    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error: Could not process image."

# --- 5. Find Example Images ---
example_list = []
try:
    test_normal_dir = os.path.join(base_dir, 'test', 'NORMAL')
    test_pneumonia_dir = os.path.join(base_dir, 'test', 'PNEUMONIA')
    example_normal = os.path.join(test_normal_dir, os.listdir(test_normal_dir)[5])
    example_pneumonia = os.path.join(test_pneumonia_dir, os.listdir(test_pneumonia_dir)[5])
    example_list = [example_normal, example_pneumonia]
    print(f"Found example images: {example_list}")
except Exception as e:
    print(f"Warning: Could not load example images. {e}")

# --- 6. Create the Gradio Interface ---
print("Creating Gradio interface...")
with gr.Blocks(theme=gr.themes.Monochrome()) as app:
    
    gr.HTML(
        """
        <div style="text-align: center; max-width: 800px; margin: 0 auto;">
            <h1 style="font-size: 2.5em; font-weight: 600;">🩻 Pneumonia X-Ray Diagnosis</h1>
            <p style="font-size: 1.1em; color: #555;">
                This AI model (VGG16) was trained to detect signs of pneumonia,
                achieving <strong>89.90% accuracy</strong>.
            </p>
        </div>
        """
    )
    
    with gr.Row(variant="panel"):
        with gr.Column(scale=1):
            gr.Markdown("### 1. Upload Your Image")
            img_input = gr.Image(type="numpy", label="Drag & Drop X-Ray Here", height=300)
            predict_btn = gr.Button("Diagnose", variant="primary", scale=1)
        with gr.Column(scale=1):
            gr.Markdown("### 2. View Diagnosis")
            output_text = gr.Textbox(label="Prediction", scale=2)
            gr.Markdown(
                """
                **Disclaimer:** This tool is for educational purposes only and is not a 
                substitute for professional medical advice.
                """
            )

    if example_list:
        gr.Examples(
            examples=example_list,
            inputs=img_input,
            outputs=output_text, 
            fn=predict_image, 
            cache_examples=False 
        )

    predict_btn.click(
        fn=predict_image,
        inputs=img_input,
        outputs=output_text
    )

# --- 7. Launch the App ---
print("Launching Gradio app...")
app.launch(share=True) # share=True is good for local testing
print("Launching Gradio app...")
# On Hugging Face, 'launch()' is all you need. 
# 'share=True' is not necessary.

app.launch()
