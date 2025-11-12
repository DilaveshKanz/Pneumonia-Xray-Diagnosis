import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
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
install_package("seaborn")
install_package("sklearn")
install_package("matplotlib")
install_package("kagglehub") # Make sure kagglehub is installed

import kagglehub

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


# --- 1. Configuration & DYNAMIC PATH FINDING ---
print("--- Starting Project: Pneumonia Detection (VGG16) ---")

# Automatically find the dataset path
print("Downloading/finding dataset path...")
try:
    dataset_root_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    base_dir = os.path.join(dataset_root_path, "chest_xray")
    print(f"Dataset found at: {base_dir}")
except Exception as e:
    print(f"Error downloading dataset from Kaggle Hub: {e}")
    # Fallback for local-only environments
    base_dir = 'chest_xray'
    print(f"Warning: Could not use kagglehub. Assuming dataset is in local folder: '{base_dir}'")

# Check if the path is valid
if not os.path.isdir(base_dir):
    print(f"Error: The directory '{base_dir}' does not exist.")
    print("Please make sure your data is in a folder named 'chest_xray' or that kagglehub can access it.")
    sys.exit(1)

train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Model parameters
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS = 25  # Max epochs (EarlyStopping will find the best)
LEARNING_RATE = 0.0001
PATIENCE = 3 # Patience for EarlyStopping

# File paths
MODEL_SAVE_PATH = 'pneumonia_VGG16_best.keras'
HISTORY_SAVE_PATH = 'training_history_VGG16.pkl'


# --- 2. Setup Data Pipeline ---
print(f"\n--- Setting up Data Generators (using 80/20 split from {train_dir}) ---")

try:
    train_val_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # Use 20% of training data for validation
    )

    train_generator = train_val_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='rgb',
        subset='training'  # 80%
    )

    validation_generator = train_val_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='rgb',
        subset='validation', # 20%
        shuffle=False
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=1, # 1 image at a time for evaluation
        class_mode='binary',
        color_mode='rgb',
        shuffle=False
    )
except FileNotFoundError as e:
    print(f"Data directory error: {e}")
    print("Could not find 'train' or 'test' folders inside your base_dir.")
    print(f"Please check the folder structure at: {base_dir}")
    sys.exit(1)

print(f"Training images: {train_generator.samples}")
print(f"Validation images: {validation_generator.samples}")
print(f"Test images: {test_generator.samples}")


# --- 3. Build Model ---
print("\n--- Building VGG16 Model ---")

def build_model(input_shape, lr):
    # Load VGG16 base
    base_model = VGG16(
        input_shape=input_shape,
        include_top=False, 
        weights='imagenet'
    )
    
    # Freeze the base
    print("Freezing VGG16 base layers...")
    base_model.trainable = False
    
    # Add our custom head
    x = base_model.output
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x) # Key layer for preventing overfitting
    output = layers.Dense(1, activation='sigmoid')(x)
    
    # Create and compile
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=lr), 
        metrics=['accuracy']
    )
    return model

model = build_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), lr=LEARNING_RATE)
model.summary()


# --- 4. Train Model ---
print("\n--- Calculating Class Weights ---")
training_labels = train_generator.classes
class_labels = np.unique(training_labels)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=class_labels,
    y=training_labels
)
class_weight_dict = dict(enumerate(class_labels, class_weights))
print(f"Class indices: {train_generator.class_indices}")
print(f"Class weights computed: {class_weight_dict}")

print(f"\nSetting up EarlyStopping callback (Patience={PATIENCE})...")
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE, 
    verbose=1,
    restore_best_weights=True # Automatically restore the best model
)

print(f"\n--- Starting VGG16 Model Training (Max {EPOCHS} Epochs) ---")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=[early_stopping]
)

print("\n--- Training Complete ---")

# Save the final model and history
print(f"Saving best model to {MODEL_SAVE_PATH}...")
model.save(MODEL_SAVE_PATH)
print("Model saved successfully.")

with open(HISTORY_SAVE_PATH, 'wb') as f:
    pickle.dump(history.history, f)
print(f"Training history saved to {HISTORY_SAVE_PATH}")


# --- 5. Evaluate Model ---
print("\n--- Evaluating Final Model on Test Set ---")

# Plot Training History
print("Plotting training history...")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_ran = range(1, len(acc) + 1)

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_ran, acc, 'bo-', label='Training Accuracy')
plt.plot(epochs_ran, val_acc, 'ro-', label='Validation Accuracy')
plt.title('VGG16 Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(epochs_ran, loss, 'bo-', label='Training Loss')
plt.plot(epochs_ran, val_loss, 'ro-', label='Validation Loss')
plt.title('VGG16 Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('training_history_VGG16.png')
print("Saved 'training_history_VGG16.png'")
plt.show(block=False) 

# Get Test Set Predictions
test_generator.reset()
true_labels = test_generator.classes
steps_needed = int(np.ceil(test_generator.samples / test_generator.batch_size))
raw_predictions = model.predict(test_generator, steps=steps_needed)
predicted_labels = (raw_predictions[:len(true_labels)] > 0.5).astype(int)

# Show Final Metrics
class_names = list(test_generator.class_indices.keys())
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"\n--- FINAL VGG16 ACCURACY ---")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("\n--- Classification Report ---")
print(classification_report(true_labels, predicted_labels, target_names=class_names))

# Show Confusion Matrix
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix (VGG16 Model)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix_VGG16.png')
print("Saved 'confusion_matrix_VGG16.png'")
plt.show(block=False) 

print("\n--- Project Finished Successfully ---")
print("You can now run app.py")