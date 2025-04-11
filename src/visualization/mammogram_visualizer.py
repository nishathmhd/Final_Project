import os
import sys
import numpy as np
import matplotlib
# Use non-interactive backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import cv2

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config import DL_MODELS_DIR, RESULTS_DIR, IMAGE_SIZE

def gradcam(model, img_array, layer_name='conv5_block3_out'):
    """
    Generate Grad-CAM visualization for a mammogram.
    
    Args:
        model: Trained Keras model
        img_array: Preprocessed image (normalized, correct shape)
        layer_name: Name of the layer to use for Grad-CAM
        
    Returns:
        tuple: (heatmap, superimposed_img)
    """
    # Get the desired layer
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs], 
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    # Get gradients
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    
    # Extract gradients
    grads = tape.gradient(loss, conv_output)
    
    # Pool gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Use pooled gradients to weight the output feature map
    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert original image from normalized to 0-255
    orig_img = img_array[0] * 255
    orig_img = np.uint8(orig_img)
    
    # Superimpose heatmap on original image
    superimposed_img = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)
    
    return heatmap, superimposed_img

def visualize_augmentations(img_path, output_path=None):
    """
    Visualize different data augmentation techniques applied to a mammogram.
    
    Args:
        img_path: Path to mammogram image
        output_path: Path to save visualization
        
    Returns:
        None
    """
    try:
        # Load image
        img = tf.keras.preprocessing.image.load_img(img_path)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Create augmentation generators
    rotation_datagen = ImageDataGenerator(rotation_range=20, fill_mode='nearest')
    shift_datagen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2, fill_mode='nearest')
    shear_datagen = ImageDataGenerator(shear_range=0.2, fill_mode='nearest')
    zoom_datagen = ImageDataGenerator(zoom_range=0.2, fill_mode='nearest')
    flip_datagen = ImageDataGenerator(horizontal_flip=True)
    
    # Apply augmentations
    rotation_img = next(rotation_datagen.flow(img_array, batch_size=1))[0]
    shift_img = next(shift_datagen.flow(img_array, batch_size=1))[0]
    shear_img = next(shear_datagen.flow(img_array, batch_size=1))[0]
    zoom_img = next(zoom_datagen.flow(img_array, batch_size=1))[0]
    flip_img = next(flip_datagen.flow(img_array, batch_size=1))[0]
    
    try:
        # Create augmentation generators
        rotation_datagen = ImageDataGenerator(rotation_range=20, fill_mode='nearest')
        shift_datagen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2, fill_mode='nearest')
        shear_datagen = ImageDataGenerator(shear_range=0.2, fill_mode='nearest')
        zoom_datagen = ImageDataGenerator(zoom_range=0.2, fill_mode='nearest')
        flip_datagen = ImageDataGenerator(horizontal_flip=True)
        
        # Apply augmentations
        rotation_img = next(rotation_datagen.flow(img_array, batch_size=1))[0]
        shift_img = next(shift_datagen.flow(img_array, batch_size=1))[0]
        shear_img = next(shear_datagen.flow(img_array, batch_size=1))[0]
        zoom_img = next(zoom_datagen.flow(img_array, batch_size=1))[0]
        flip_img = next(flip_datagen.flow(img_array, batch_size=1))[0]
        
        # Create visualization
        plt.figure(figsize=(15, 8))
        
        # Original image
        plt.subplot(2, 3, 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis('off')
        
        # Rotation
        plt.subplot(2, 3, 2)
        plt.imshow(rotation_img)
        plt.title('Rotation (20°)')
        plt.axis('off')
        
        # Shift
        plt.subplot(2, 3, 3)
        plt.imshow(shift_img)
        plt.title('Width/Height Shift (20%)')
        plt.axis('off')
        
        # Shear
        plt.subplot(2, 3, 4)
        plt.imshow(shear_img)
        plt.title('Shear (20%)')
        plt.axis('off')
        
        # Zoom
        plt.subplot(2, 3, 5)
        plt.imshow(zoom_img)
        plt.title('Zoom (20%)')
        plt.axis('off')
        
        # Flip
        plt.subplot(2, 3, 6)
        plt.imshow(flip_img)
        plt.title('Horizontal Flip')
        plt.axis('off')
        
        plt.suptitle('Data Augmentation Techniques for Mammograms', fontsize=16)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            print(f"Visualization saved to {output_path}")
        else:
            # Use non-interactive backend if showing fails
            import matplotlib
            matplotlib.use('Agg')
            plt.show()
    except Exception as e:
        print(f"Error during visualization: {e}")

def visualize_preprocessing(img_path, output_path=None):
    """Visualize mammogram preprocessing steps including CLAHE enhancement"""
    # Load original image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error loading image: {img_path}")
        return

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(img)

    # Resize image
    resized_img = cv2.resize(img, (300, 300))
    resized_clahe = cv2.resize(clahe_img, (300, 300))

    # Create visualization
    plt.figure(figsize=(12, 4))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # CLAHE enhanced
    plt.subplot(1, 3, 2)
    plt.imshow(clahe_img, cmap='gray')
    plt.title('After CLAHE')
    plt.axis('off')
    
    # Final resized
    plt.subplot(1, 3, 3)
    plt.imshow(resized_clahe, cmap='gray')
    plt.title('Final (300x300)')
    plt.axis('off')
    
    plt.suptitle('Mammogram Preprocessing Steps', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

def visualize_prediction(model, img_path, output_path=None):
    """
    Make a prediction and visualize it with Grad-CAM.
    
    Args:
        model: Trained Keras model
        img_path: Path to mammogram image
        output_path: Path to save visualization
        
    Returns:
        tuple: (prediction, confidence)
    """
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Make prediction
    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])
    class_label = "Malignant" if confidence > 0.5 else "Benign"
    
    # Generate Grad-CAM visualization
    heatmap, superimposed_img = gradcam(model, img_array)
    
    # Create visualization
    plt.figure(figsize=(12, 4))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    
    # Heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    plt.title('Class Activation Map')
    plt.axis('off')
    
    # Superimposed
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Prediction: {class_label} ({confidence:.2%})')
    plt.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
    
    return class_label, confidence

if __name__ == "__main__":
    # Visualize augmentations on a sample image
    test_img_path = os.path.join("/Users/nishathmhd/Desktop/Nishathmhd", "breast_cancer_project", "data", "images", "test", "benign", "Mass-Test_P_00032_RIGHT_CC_FULL_PRE.png")
    
    # Visualize data augmentations
    aug_output = os.path.join(RESULTS_DIR, 'data_augmentation_visualization.png')
    print(f"Visualizing augmentations for: {test_img_path}")
    visualize_augmentations(test_img_path, aug_output)
    
    # Visualize preprocessing steps
    preprocess_output = os.path.join(RESULTS_DIR, 'preprocessing_steps.png')
    print(f"\nVisualizing preprocessing for: {test_img_path}")
    visualize_preprocessing(test_img_path, preprocess_output)
