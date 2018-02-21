"""Tutorial for the CS7GV1 Computer Vision 17/18 lecture at Trinity College Dublin.

This script prepares a hdf5 image database file from the tiny-imagenet dataset."""

import os

def save_name_dict_to_file(data_dir, class_to_name_dict):
    """This function saves a dictionary that maps class labels to descriptions in a text file.

    Args:
        data_dir: Main directory for the datasets.
        class_to_name_dict: Dictionary that will be saved.

    Returns:
        Nothing."""

    if not os.path.exists(os.path.join(data_dir, 'cache')):
        os.makedirs(os.path.join(data_dir, 'cache'))

    # Read the Tiny-Imagenet words.txt that maps the folder names to actual class descriptions
    with open(os.path.join(data_dir, 'words.txt')) as f:
        content = f.readlines()
    # Remove whitespace characters
    content = [x.strip() for x in content] 
    # Create a dict with folder names to class descriptions
    name_to_desc = {}
    # And fill it
    for c in content:
        c_ = c.split('\t')
        name_to_desc[c_[0]] = c_[1]

    # Save to file
    save_file = os.path.join(data_dir, 'cache', 'class_to_name.txt')
    with open(save_file, 'w') as text_file:
        for key, value in class_to_name_dict.iteritems():
            text_file.write('{:03d}\t{}\n'.format(key, name_to_desc[value]))
    text_file.close()

def save_filename_list(filenames, data_dir, name):
    """This function takes a list of filepaths/label tuples and saves them in .txt file.

    Args:
        data_dir: Main directory for the datasets.
        filenames: List of tuples (image_path, label).

    Returns:
        Path to the saved file."""

    if not os.path.exists(os.path.join(data_dir, 'cache')):
        os.makedirs(os.path.join(data_dir, 'cache'))

    save_file = os.path.join(data_dir, 'cache', name)
    with open(save_file, 'w') as text_file:
        for path, label in filenames:
            text_file.write('{} {}\n'.format(path, label))
    text_file.close()
    return save_file

def build_dataset_index(data_dir):
    train_file, class_dict = build_train_index(data_dir)
    val_file = build_val_index(data_dir, class_dict)

    return train_file, val_file

def build_val_index(data_dir, class_dict):
    """This functions lists all image paths and their labels for the validation set and
    saves them in a file.

    Args:
        data_dir: Main directory of the datasets.

    Returns:
        Path to file that contains all validation paths and labels."""

    filenames = []

    # Open val_annotations.txt to find the labels for the validation images
    with open(os.path.join(data_dir, 'val', 'val_annotations.txt'), 'r') as text_file:
        content = text_file.readlines()
    content = [x.strip() for x in content]

    # Build dict to map image name to label
    label_dict = {}
    for line in content:
        line_split = line.split('\t')
        label_dict[line_split[0]] = class_dict[line_split[1]]

    # Travel through the val folder
    for subdir, dirs, files in os.walk(os.path.join(data_dir, 'val')):
        for file in files:
            if file.endswith('.JPEG'):
                file_path = os.path.join(subdir, file)
                filenames.append((file_path, label_dict[file]))

    save_file = save_filename_list(filenames, data_dir, 'val_image_paths.txt')
    return save_file

def build_train_index(data_dir):
    """This function lists all image paths and their labels. It also saves a .txt file
    that maps class labels to descriptions and a .txt file that contains all image paths
    and their corresponding labels.

    Args: 
        data_dir: Main directory for the datasets.
    
    Returns:
        Path to file that contains all image paths and labels.
        Dictionary mapping folder names to class labels."""

    # Build a dict to map from folder names in the datasets to class indices
    next_class = 0
    name_to_class = {}
    class_to_name = {}
    filenames = []

    # Travel through the train folder and build the dicts
    for subdir, dirs, files in os.walk(os.path.join(data_dir, 'train')):
        # Fill in the dict
        if subdir.endswith('tiny-imagenet-200/train'):
            folder_names = dirs
            for class_name in dirs:
                name_to_class[class_name] = next_class
                class_to_name[next_class] = class_name
                next_class += 1
        # Add all .jpeg files to the list
        for file in files:
            if file.endswith('.JPEG'):
                class_label = name_to_class[subdir.split('/')[-2]]
                file_path = os.path.join(subdir, file)
                filenames.append((file_path, class_label))

    # Write data to disk
    save_name_dict_to_file(data_dir, class_to_name)
    save_file = save_filename_list(filenames, data_dir, 'train_image_paths.txt')
    return save_file, name_to_class
