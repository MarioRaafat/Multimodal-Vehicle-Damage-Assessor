
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import config

directories = [
    config.MODELS_DIR,
    config.RESULTS_DIR,
    config.LOGS_DIR
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)

train_exists = os.path.exists(config.SEVERITY_TRAIN_DIR)
val_exists = os.path.exists(config.SEVERITY_VAL_DIR)

if train_exists and val_exists:
    for split, split_dir in [('Training', config.SEVERITY_TRAIN_DIR), ('Validation', config.SEVERITY_VAL_DIR)]:
        print(f"\n{split} Set:")
        total = 0
        for class_folder in ['01-minor', '02-moderate', '03-severe']:
            class_dir = os.path.join(split_dir, class_folder)
            if os.path.exists(class_dir):
                count = len([f for f in os.listdir(class_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"  {class_folder}: {count} images")
                total += count
        print(f"  Total: {total} images")

else:
    print(f"  Expected training dir: {config.SEVERITY_TRAIN_DIR}")
    print(f"  Expected validation dir: {config.SEVERITY_VAL_DIR}")
    print("\nPlease check your dataset location.")
