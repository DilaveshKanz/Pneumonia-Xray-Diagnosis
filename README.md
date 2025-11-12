<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pneumonia X-Ray Diagnosis</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            padding: 20px;
            max-width: 900px;
            margin: 0 auto;
            background-color: #f6f8fa;
        }
        h1, h2, h3 {
            border-bottom: 2px solid #eaecef;
            padding-bottom: 10px;
        }
        h1 { font-size: 2.2em; }
        h2 { font-size: 1.8em; }
        h3 { font-size: 1.4em; }
        pre {
            background-color: #ffffff;
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 16px;
            overflow: auto;
        }
        code {
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            font-size: 0.9em;
        }
        pre code {
            font-size: 1em;
        }
        img {
            max-width: 100%;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 16px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: center;
        }
        th {
            background-color: #f6f8fa;
        }
    </style>
</head>
<body>

    <h1>🩻 Pneumonia X-Ray Diagnosis (VGG16 CNN)</h1>

    <p>
        <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" alt="Python">
        <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow" alt="TensorFlow">
        <img src="https://img.shields.io/badge/Gradio-WebApp-yellow?style=for-the-badge&logo=gradio" alt="Gradio">
        <img src="https://img.shields.io/badge/Test_Accuracy-89.90%25-brightgreen?style=for-the-badge" alt="Accuracy">
    </p>

    <p>A high-performance, VGG16-based Convolutional Neural Network built with TensorFlow to diagnose pneumonia from chest X-ray images. This model achieves <strong>89.90% test accuracy</strong> using a robust, trustworthy validation strategy.</p>
    
    <p>This project was built as part of my portfolio for application to MBZUAI.</p>

    <hr>

    <h2>🏆 Final Results: 89.90% Accuracy</h2>

    <p>This model excels at its most critical task: <strong>identifying pneumonia (96% Recall)</strong>, while maintaining a strong balance in identifying normal cases (79% Recall).</p>

    <table>
        <thead>
            <tr>
                <th style="text-align: center;">Training History</th>
                <th style="text-align: center;">Confusion Matrix</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="text-align: center;"><img src="training_history_VGG16.png" alt="Training History"></td>
                <td style="text-align: center;"><img src="confusion_matrix_VGG16.png" alt="Final Confusion Matrix"></td>
            </tr>
        </tbody>
    </table>

    <hr>

    <h2>🔬 The Strategy: Why This Model is Robust</h2>

    <p>Many public notebooks on this dataset achieve 92-95% accuracy by making a critical error: they validate against the tiny, 16-image <code>val</code> folder. These scores are "flukes" and are not reproducible.</p>

    <p>My approach was to build a <strong>scientifically sound and reliable model</strong>.</p>

    <ol>
        <li><strong>A Robust Validation Set:</strong> I ignored the 16-image <code>val</code> folder and created a proper <strong>80/20 validation split</strong> (1,043 images) from the main 5,216-image <code>train</code> set.</li>
        <li><strong>State-of-the-Art Architecture:</strong> I used <strong>Transfer Learning</strong> with the proven VGG16 model, freezing the base and adding a custom classification head with <code>Dropout(0.5)</code> to prevent overfitting.</li>
        <li><strong>Handling Imbalance:</strong> The model was trained using <code>class_weight</code> to teach it to pay closer attention to the minority "NORMAL" class.</li>
        <li><strong>Finding the Best Model:</strong> <code>EarlyStopping</code> (monitoring <code>val_loss</code>) was used to automatically find the model's peak performance and prevent overfitting.</li>
    </ol>

    <p>This <strong>89.90%</strong> score is a trustworthy, reproducible benchmark of what this architecture can achieve.</p>

    <hr>

    <h2>🚀 Quickstart</h2>

    <p>This repository is fully self-contained. The scripts will automatically download the dataset and install dependencies.</p>

    <h3>1. Clone the Repository</h3>
    <pre><code>git clone https://github.com/your-username/Pneumonia-Xray-Diagnosis-CNN.git
cd Pneumonia-Xray-Diagnosis-CNN</code></pre>

    <h3>2. Install Dependencies</h3>
    <p>This will install all required libraries.</p>
    <pre><code>pip install -r requirements.txt</code></pre>

    <h3>3. Train & Evaluate the Model</h3>
    <p>This all-in-one script runs the entire experiment:</p>
    <ul>
        <li>Downloads the Kaggle dataset</li>
        <li>Runs the training with <code>EarlyStopping</code></li>
        <li>Saves the best model as <code>pneumonia_VGG16_best.keras</code></li>
        <li>Prints the final accuracy report and saves the graphs.</li>
    </ul>
    <pre><code>python train_and_evaluate.py</code></pre>

    <h3>4. Run the Interactive Web App</h3>
    <p>Once the model is trained, you can launch the Gradio web demo.</p>
    <pre><code>python app.py</code></pre>
    <p>Open the local URL (e.g., <code>http://127.0.0.1:7860</code>) or the public Gradio link in your browser to use the app.</p>

    <hr>

    <h2>🛠️ Technology Stack</h2>
    <ul>
        <li><strong>Python 3.10</strong></li>
        <li><strong>TensorFlow & Keras:</strong> For building and training the CNN.</li>
        <li><strong>Gradio:</strong> For creating the interactive web demo.</li>
        <li><strong>Scikit-learn:</strong> For calculating class weights and metrics.</li>
        <li><strong>KaggleHub:</strong> For dynamically downloading the dataset.</li>
        <li><strong>Seaborn & Matplotlib:</strong> For plotting the results.</li>
        <li><strong>OpenCV:</strong> For image preprocessing in the app.</li>
    </ul>

    <hr>

    <h2>📂 File Structure</h2>
    <pre><code>.
├── 🚀 app.py                 # The Gradio Web App
├── 📊 train_and_evaluate.py   # All-in-one training & evaluation script
├── 📜 README.md                # You are here!
├── 📦 requirements.txt         # All Python dependencies
├── .gitignore               # Ignores the dataset folder
│
├── 🖼️ confusion_matrix_VGG16.png  (Generated by script)
├── 📈 training_history_VGG16.png (Generated by script)
└── 🤖 pneumonia_VGG16_best.keras (Generated by script)
</code></pre>

</body>
</html>
