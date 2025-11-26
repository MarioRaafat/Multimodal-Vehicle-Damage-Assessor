import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import shutil

from config import (
    TRAIN_DIR, VAL_DIR, IMG_SIZE, BATCH_SIZE,
    ROTATION_RANGE, WIDTH_SHIFT_RANGE, HEIGHT_SHIFT_RANGE,
    HORIZONTAL_FLIP, ZOOM_RANGE, CLASS_NAMES, RANDOM_SEED
)


def create_data_generators(img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=ROTATION_RANGE,
        width_shift_range=WIDTH_SHIFT_RANGE,
        height_shift_range=HEIGHT_SHIFT_RANGE,
        horizontal_flip=HORIZONTAL_FLIP,
        zoom_range=ZOOM_RANGE,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=RANDOM_SEED
    )
    
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, train_generator.class_indices


def load_dataset_for_yolo(output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Create directory structure
    for split in ['train', 'val']:
        for class_name in CLASS_NAMES:
            (output_path / split / class_name).mkdir(exist_ok=True, parents=True)
    
    # Copy images to YOLO format
    for split, source_dir in [('train', TRAIN_DIR), ('val', VAL_DIR)]:
        for folder in source_dir.iterdir():
            if folder.is_dir():
                class_name = CLASS_NAMES[int(folder.name.split('-')[0]) - 1]
                dest_dir = output_path / split / class_name
                
                for img_file in folder.glob('*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        shutil.copy2(img_file, dest_dir / img_file.name)
    
    # Create dataset.yaml
    yaml_content = f"""# Vehicle Damage Severity Dataset
path: {output_path.absolute().as_posix()}
train: train
val: val

# Classes
names:
  0: minor
  1: moderate
  2: severe
"""
    
    yaml_path = output_path / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    return str(yaml_path)


def get_class_distribution(data_dir):
    distribution = {}
    
    for folder in Path(data_dir).iterdir():
        if folder.is_dir():
            class_name = folder.name
            count = len(list(folder.glob('*')))
            distribution[class_name] = count
    
    return distribution


def visualize_sample_images(data_dir, num_samples=3):
    folders = sorted([f for f in Path(data_dir).iterdir() if f.is_dir()])
    
    fig, axes = plt.subplots(len(folders), num_samples, figsize=(15, 5*len(folders)))
    
    for i, folder in enumerate(folders):
        images = list(folder.glob('*'))[:num_samples]
        class_name = folder.name
        
        for j, img_path in enumerate(images):
            img = Image.open(img_path)
            if len(folders) == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]
            ax.imshow(img)
            ax.axis('off')
            if j == 0:
                ax.set_title(f'{class_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def load_image(img_path, img_size=IMG_SIZE):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_size, img_size))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array
