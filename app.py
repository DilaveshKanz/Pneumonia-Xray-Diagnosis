import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import os
import warnings
import sys
import subprocess

# --- 0. Install Dependencies ---
def install_package(package):
    """Installs a package if it's not already installed."""
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

print("--- Checking dependencies ---")
install_package("tensorflow")
install_package("gradio")
install_package("opencv-python-headless") # Use headless for servers
install_package("numpy")
install_package("kagglehub") # Keep this if you want auto-downloading examples

# --- 1. Imports (NOW they are safe to import) ---
import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import kagglehub

# --- 2. Configuration ---
MODEL_PATH = 'pneumonia_VGG16_best.keras'
IMG_HEIGHT = 150
IMG_WIDTH = 150
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# --- 3. Find Dataset Path (for examples) ---
print("Finding dataset path for examples...")
try:
    dataset_root_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    base_dir = os.path.join(dataset_root_path, "chest_xray")
    print(f"Dataset found at: {base_dir}")
except Exception:
    base_dir = 'chest_xray' # Fallback
    print(f"Warning: Could not use kagglehub. Assuming dataset is in local folder: '{base_dir}'")
    
# --- 4. Load the Trained Model ---
print(f"Loading model from {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    print("Please make sure 'pneumonia_VGG16_best.keras' is in the same folder.")
    sys.exit(1)
    
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# --- 5. Define the Prediction Function ---
def predict_image(img):
    """
    Takes a NumPy image, preprocesses it, and returns
    a dictionary of {class_name: probability}.
    """
    try:
        img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        
        if len(img_resized.shape) == 2: # if grayscale
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        img_normalized = img_rgb / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)

        prediction = model.predict(img_batch, verbose=0) 
        probability = prediction[0][0] 

        return {'NORMAL': 1.0 - probability, 'PNEUMONIA': float(probability)}

    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"Error": 0.0}

# --- 6. Find Example Images ---
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
    print(f"Make sure this path is correct: {os.path.abspath(base_dir)}")

# --- 7. Create the Gradio Interface ---
print("Creating Gradio interface...")

with gr.Blocks(theme=gr.themes.Monochrome()) as app:
    
    gr.HTML(
        """
        <div style="text-align: center; max-width: 800px; margin: 0 auto;">
            <h1 style="font-size: 2.5em; font-weight: 600;">🩻 Pneumonia X-Ray Diagnosis</h1>
            <p style="font-size: 1.1em; color: #555;">
                This AI model (VGG16) was trained to detect signs of pneumonia from chest X-ray images,
                achieving <strong>89.90% accuracy</strong> with a <strong>96% recall for pneumonia</strong>.
            </p>
        </div>
        """
    )
    
    with gr.Row(variant="panel"):
        # Column 1: Inputs
        with gr.Column(scale=1):
            gr.Markdown("### 1. Upload Your Image")
            img_input = gr.Image(
                type="numpy", 
                label="Drag & Drop X-Ray Here",
                height=300
            )
            predict_btn = gr.Button("Diagnose", variant="primary", scale=1)

        # Column 2: Outputs
        with gr.Column(scale=1):
            gr.Markdown("### 2. View Diagnosis")
            output_label = gr.Label(
                num_top_classes=2, 
                label="Prediction",
                scale=2
            )
            
            gr.Markdown(
                """
                **Disclaimer:** This tool is for educational purposes only and is not a substitute for
                professional medical advice. The model's predictions are not 100% accurate.
                """
            )

    if example_list:
        gr.Examples(
            examples=example_list,
            inputs=img_input,
            outputs=output_label, 
            fn=predict_image, 
            # *** THIS IS THE FIX ***
            cache_examples=False # Changed to False
            # *** END OF FIX ***
        )

    predict_btn.click(
        fn=predict_image,
        inputs=img_input,
        outputs=output_label
    )

# --- 8. Launch the App ---
print("Launching Gradio app...")
# On Hugging Face, 'launch()' is all you need. 
# 'share=True' is not necessary.
app.launch()