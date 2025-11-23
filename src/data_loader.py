import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import config

def create_data_generators(workers=1, use_multiprocessing=False, max_queue_size=10):
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=config.ROTATION_RANGE,
        width_shift_range=config.WIDTH_SHIFT_RANGE,
        height_shift_range=config.HEIGHT_SHIFT_RANGE,
        shear_range=config.SHEAR_RANGE,
        zoom_range=config.ZOOM_RANGE,
        horizontal_flip=config.HORIZONTAL_FLIP,
        fill_mode=config.FILL_MODE
    )
    
    # Validation data - only rescaling
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        config.SEVERITY_TRAIN_DIR,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=config.RANDOM_SEED
    )
    
    val_generator = val_datagen.flow_from_directory(
        config.SEVERITY_VAL_DIR,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        seed=config.RANDOM_SEED
    )
    
    return train_generator, val_generator


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


def plot_sample_images(generator, num_images=16):
    images, labels = next(generator)

    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    fig.suptitle('Sample Images from Dataset', fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        if i < num_images and i < len(images):
            ax.imshow(images[i])
            class_idx = np.argmax(labels[i])
            class_name = config.SEVERITY_CLASS_NAMES[class_idx]
            ax.set_title(f'Class: {class_name}')
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
