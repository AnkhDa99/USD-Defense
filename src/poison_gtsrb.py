import numpy as np
from copy import deepcopy
from PIL import Image
import torch

# MODIFIED: Trigger generation now creates a standalone pattern, not tied to a specific image size.
def generate_trigger_pattern(trigger_type, trigger_size=(3, 3)):
    """
    Generates the trigger pattern and mask as small numpy arrays.
    """
    h, w = trigger_size
    if trigger_type == 'white_square':
        # A simple white square trigger
        pattern = np.ones((h, w, 3), dtype=np.uint8) * 255
        mask = np.ones((h, w, 3), dtype=np.uint8)
    elif trigger_type == 'checkerboard_3x3':
        # A 3x3 checkerboard
        pattern = np.zeros(shape=(h, w, 3), dtype=np.uint8)
        mask = np.ones(shape=(h, w, 3), dtype=np.uint8)
        trigger_value = [[0, 255, 0], [255, 0, 255], [0, 255, 0]]
        for y in range(h):
            for x in range(w):
                # Apply to all 3 color channels
                pattern[y, x, :] = trigger_value[y][x]
    else:
        raise ValueError(f"Trigger type '{trigger_type}' not implemented for GTSRB.")
    
    return pattern, mask

def add_predefined_trigger_gtsrb(data_set, trigger_info, exclude_target=True):
    """将同样的触发器加到测试集中所有非目标类样本上，用于计算 ASR。"""
    if trigger_info is None:
        return data_set
    poison_set = deepcopy(data_set)
    trig_h, trig_w = trigger_info['trigger_size']
    alpha = trigger_info['trigger_alpha']
    ttype = trigger_info['trigger_type']
    pattern, mask = generate_trigger_pattern(ttype, (trig_h, trig_w))

    for i in range(len(poison_set.images)):
        img_np = np.array(poison_set.images[i])
        H, W, _ = img_np.shape
        sy, sx = H - trig_h, W - trig_w
        full_mask = np.zeros_like(img_np, dtype=np.uint8); full_mask[sy:H, sx:W, :] = mask
        full_pat  = np.zeros_like(img_np, dtype=np.uint8); full_pat[sy:H, sx:W, :]  = pattern
        out = np.clip((1-full_mask)*img_np + full_mask*((1-alpha)*img_np + alpha*full_pat), 0, 255).astype(np.uint8)
        poison_set.images[i] = Image.fromarray(out)
        poison_set.labels[i] = int(trigger_info['poison_target'][0])

    if exclude_target:
        keep = [j for j, y in enumerate(data_set.labels) if y != int(trigger_info['poison_target'][0])]
        poison_set.images = [poison_set.images[j] for j in keep]
        poison_set.labels = [poison_set.labels[j] for j in keep]
    return poison_set

# NEW: The main poisoning function for variable-sized GTSRB images
def add_trigger_gtsrb(data_set, trigger_type, poison_rate, poison_target, trigger_alpha=1.0):
    """
    Adds a trigger to variable-sized images in a GTSRB-style dataset.
    The trigger is applied at the bottom-right corner.
    """
    # Generate a small, fixed-size trigger pattern. Let's use 3x3 for GTSRB.
    trigger_pattern, trigger_mask = generate_trigger_pattern(trigger_type, trigger_size=(3, 3))
    trig_h, trig_w, _ = trigger_pattern.shape
    
    poison_set = deepcopy(data_set)

    # GTSRB uses ._samples which is a list of (image_path, label) tuples
    # and is loaded into memory. We assume `data_set` has been loaded
    # and has attributes like `images` and `labels` (as from your split function).
    labels_array = np.array(poison_set.labels)
    
    poison_cand = [i for i in range(len(labels_array)) if labels_array[i] != poison_target]
    poison_num = int(poison_rate * len(poison_cand))
    choices = np.random.choice(poison_cand, poison_num, replace=False)

    print(f"[GTSRB Poison] Poisoning {len(choices)} images with a {trig_h}x{trig_w} '{trigger_type}' trigger.")

    for idx in choices:
        # Get the original PIL Image
        original_pil_image = poison_set.images[idx]
        
        # Convert to numpy array for manipulation
        img_np = np.array(original_pil_image)
        img_h, img_w, _ = img_np.shape

        # Ensure trigger fits on the image (GTSRB images can be small)
        if img_h < trig_h or img_w < trig_w:
            print(f"Skipping image {idx} as it's too small ({img_h}x{img_w}) for the trigger.")
            continue
        
        # Define the placement at the bottom-right corner
        start_y, start_x = img_h - trig_h, img_w - trig_w
        
        # Create a full-size mask for this specific image
        full_mask = np.zeros_like(img_np, dtype=np.uint8)
        full_pattern = np.zeros_like(img_np, dtype=np.uint8)
        
        # Place the small trigger pattern and mask onto the full-size placeholders
        full_mask[start_y:img_h, start_x:img_w, :] = trigger_mask
        full_pattern[start_y:img_h, start_x:img_w, :] = trigger_pattern
        
        # Apply the trigger via blending
        poisoned_img_np = np.clip(
            (1 - full_mask) * img_np + full_mask * ((1 - trigger_alpha) * img_np + trigger_alpha * full_pattern),
            0, 255
        ).astype(np.uint8)
        
        # Convert back to PIL Image and update the dataset
        poison_set.images[idx] = Image.fromarray(poisoned_img_np)
        poison_set.labels[idx] = poison_target

    # Note: Trigger info here is symbolic as the trigger's placement is dynamic.
    trigger_info = {'trigger_type': trigger_type, 'trigger_size': (trig_h, trig_w),
                    'trigger_alpha': trigger_alpha, 'poison_target': np.array([poison_target]),
                    'description': f'{trig_h}x{trig_w} trigger at bottom-right corner.'}
    
    return poison_set, trigger_info


# It's good practice to move the GTSRB split function here as well.
def split_dataset_gtsrb(dataset, val_frac=0.1, perm=None):
    """
    Splits the GTSRB dataset.
    :param dataset: The whole dataset which will be split.
    :param val_frac: the fraction of validation set.
    :param perm: A predefined permutation for sampling. If perm is None, generate one.
    :return: A training set + a validation set
    """
    if perm is None:
        perm = np.arange(len(dataset.images))
        np.random.shuffle(perm)
    nb_val = int(val_frac * len(dataset.images))

    train_set = deepcopy(dataset)
    train_set.images = [dataset.images[i] for i in perm[nb_val:]]
    train_set.labels = [dataset.labels[i] for i in perm[nb_val:]]

    val_set = deepcopy(dataset)
    val_set.images = [dataset.images[i] for i in perm[:nb_val]]
    val_set.labels = [dataset.labels[i] for i in perm[:nb_val]]
    return train_set, val_set