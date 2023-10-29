import audio
import image
import protein
import video

import os
import numpy as np
import pickle


def load_training_set(train_dir, 
                      dataset, 
                      seed,
                      number_of_entire_training_instances,
                      feature_size,
                      patch, 
                      patch_sizes):
    if dataset in ['cifar', 'kodak']:
        # load training dataset
        training_images = sorted(os.listdir(train_dir))
        training_images_path = []
        for img_name in training_images:
            image_path = (train_dir + img_name) if (train_dir[-1] == "/") else (train_dir + "/" + img_name)
            training_images_path.append(image_path)

        # randomly select train_size instances from the training set 
        np.random.seed(seed)
        number_of_entire_training_instances = min(len(training_images_path), number_of_entire_training_instances)
        idx = np.random.choice(len(training_images_path), number_of_entire_training_instances, False)
        np.random.seed(None)
        training_images_path = [training_images_path[i] for i in idx]
        X, Y = image.load_image(training_images_path, 
                                feature_size,
                                patch, 
                                patch_sizes)
    if dataset == 'audio':
        # load training dataset
        with open(train_dir + '/train_dataset.pkl', 'wb') as f:
            train_tensor = pickle.load(f)
        np.random.seed(seed)
        number_of_entire_training_instances = min(len(train_tensor), number_of_entire_training_instances)
        idx = np.random.choice(len(train_tensor), number_of_entire_training_instances, False)
        np.random.seed(None)
        train_tensor = [train_tensor[i] for i in idx]
        X, Y = audio.load_audio(train_tensor, 
                                feature_size,
                                patch, 
                                patch_sizes)
    if dataset == 'video':
        # load training dataset
        with open(train_dir + '/train_dataset.pkl', 'wb') as f:
            train_tensor = pickle.load(f)
        np.random.seed(seed)
        number_of_entire_training_instances = min(len(train_tensor), number_of_entire_training_instances)
        idx = np.random.choice(len(train_tensor), number_of_entire_training_instances, False)
        np.random.seed(None)
        train_tensor = [train_tensor[i] for i in idx]
        X, Y = video.load_video(train_tensor, 
                                feature_size,
                                patch, 
                                patch_sizes)
    if dataset == 'protein':
        # load training dataset
        with open(train_dir + '/train_dataset.pkl', 'wb') as f:
            train_tensor = pickle.load(f)
        np.random.seed(seed)
        number_of_entire_training_instances = min(len(train_tensor), number_of_entire_training_instances)
        idx = np.random.choice(len(train_tensor), number_of_entire_training_instances, False)
        np.random.seed(None)
        train_tensor = [train_tensor[i] for i in idx]
        X, Y = protein.load_protein(train_tensor, 
                                    feature_size,
                                    patch, 
                                    patch_sizes)

    return X, Y