import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import config


def get_preprocessing_function(model_name):
    """Get the appropriate preprocessing function for each model"""
    from tensorflow.keras.applications import (
        efficientnet,
        resnet_v2,
        mobilenet_v2,
        densenet,
        inception_v3
    )
    
    preprocessing_functions = {
        'efficientnet_b4': efficientnet.preprocess_input,
        'resnet50v2': resnet_v2.preprocess_input,
        'mobilenet_v2': mobilenet_v2.preprocess_input,
        'densenet121': densenet.preprocess_input,
        'inceptionv3': inception_v3.preprocess_input
    }
    
    return preprocessing_functions.get(model_name, lambda x: x / 255.0)


def create_data_generators(model_name='efficientnet_b4', use_augmentation=True):
    """
    Create data generators with model-specific preprocessing
    
    Args:
        model_name: Name of the model (for preprocessing)
        use_augmentation: Whether to use data augmentation
        
    Returns:
        train_generator, val_generator
    """
    # Get model-specific input size
    target_size = config.PRETRAINED_MODELS[model_name]['input_size']
    
    # Get preprocessing function
    preprocess_func = get_preprocessing_function(model_name)
    
    # Training data augmentation
    if use_augmentation:
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_func,
            rotation_range=config.ROTATION_RANGE,
            width_shift_range=config.WIDTH_SHIFT_RANGE,
            height_shift_range=config.HEIGHT_SHIFT_RANGE,
            shear_range=config.SHEAR_RANGE,
            zoom_range=config.ZOOM_RANGE,
            horizontal_flip=config.HORIZONTAL_FLIP,
            fill_mode=config.FILL_MODE,
            brightness_range=[0.8, 1.2]  # Additional augmentation for vehicle images
        )
    else:
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_func)
    
    # Validation data - only preprocessing
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_func)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        config.SEVERITY_TRAIN_DIR,
        target_size=target_size,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=config.RANDOM_SEED
    )
    
    val_generator = val_datagen.flow_from_directory(
        config.SEVERITY_VAL_DIR,
        target_size=target_size,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        seed=config.RANDOM_SEED
    )
    
    print(f"\nData generators created for {model_name}")
    print(f"Target size: {target_size}")
    print(f"Train samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Batch size: {config.BATCH_SIZE}")
    
    return train_generator, val_generator


def create_tf_dataset(model_name='efficientnet_b4'):
    """
    Create TensorFlow datasets with better performance using tf.data API
    
    Args:
        model_name: Name of the model (for preprocessing)
        
    Returns:
        train_dataset, val_dataset, dataset_info
    """
    target_size = config.PRETRAINED_MODELS[model_name]['input_size']
    preprocess_func = get_preprocessing_function(model_name)
    
    # Training dataset
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        config.SEVERITY_TRAIN_DIR,
        labels='inferred',
        label_mode='categorical',
        image_size=target_size,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        seed=config.RANDOM_SEED
    )
    
    # Validation dataset
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        config.SEVERITY_VAL_DIR,
        labels='inferred',
        label_mode='categorical',
        image_size=target_size,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        seed=config.RANDOM_SEED
    )
    
    # Apply preprocessing
    train_dataset = train_dataset.map(
        lambda x, y: (preprocess_func(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    val_dataset = val_dataset.map(
        lambda x, y: (preprocess_func(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Apply data augmentation to training set
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ])
    
    train_dataset = train_dataset.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Performance optimization
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Get dataset info
    dataset_info = {
        'train_samples': len(train_dataset) * config.BATCH_SIZE,
        'val_samples': len(val_dataset) * config.BATCH_SIZE,
        'num_classes': config.SEVERITY_NUM_CLASSES,
        'class_names': config.SEVERITY_CLASS_NAMES,
        'target_size': target_size,
        'batch_size': config.BATCH_SIZE
    }
    
    print(f"\nTensorFlow datasets created for {model_name}")
    print(f"Target size: {target_size}")
    print(f"Train batches: {len(train_dataset)}")
    print(f"Validation batches: {len(val_dataset)}")
    
    return train_dataset, val_dataset, dataset_info


def get_dataset_info(train_generator, val_generator):
    info = {
        'train_samples': train_generator.samples,
        'val_samples': val_generator.samples,
        'total_samples': train_generator.samples + val_generator.samples,
        'num_classes': train_generator.num_classes,
        'class_indices': train_generator.class_indices,
        'batch_size': config.BATCH_SIZE,
        'image_shape': config.INPUT_SHAPE
    }
    
    return info


def plot_sample_images(dataset, num_images=16, denormalize=True):
    """
    Plot sample images from dataset
    
    Args:
        dataset: TensorFlow dataset or generator
        num_images: Number of images to plot
        denormalize: Whether to denormalize images for display
    """
    # Get a batch
    for images, labels in dataset.take(1):
        images_np = images.numpy()
        labels_np = labels.numpy()
        
        # Denormalize if needed (for display purposes)
        if denormalize:
            # Simple denormalization (works for most models)
            images_np = (images_np - images_np.min()) / (images_np.max() - images_np.min())
        
        fig, axes = plt.subplots(4, 4, figsize=(15, 15))
        fig.suptitle('Sample Images from Dataset', fontsize=16)
        
        for i, ax in enumerate(axes.flat):
            if i < num_images and i < len(images_np):
                ax.imshow(images_np[i])
                class_idx = np.argmax(labels_np[i])
                class_name = config.SEVERITY_CLASS_NAMES[class_idx]
                ax.set_title(f'Class: {class_name}', fontsize=10)
                ax.axis('off')
        
        plt.tight_layout()
        return fig


def visualize_augmentation(generator, num_augmentations=5):
    # Get one batch
    images, labels = next(generator)
    image = images[0]
    class_idx = np.argmax(labels[0])
    class_name = config.SEVERITY_CLASS_NAMES[class_idx]
    
    fig, axes = plt.subplots(1, num_augmentations, figsize=(20, 4))
    fig.suptitle(f'Data Augmentation Examples - Class: {class_name}', fontsize=14)
    
    for i in range(num_augmentations):
        if i == 0:
            axes[i].imshow(image)
            axes[i].set_title('Original')
        else:
            # Get another augmented version
            images_aug, _ = next(generator)
            axes[i].imshow(images_aug[0])
            axes[i].set_title(f'Augmented {i}')
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig
