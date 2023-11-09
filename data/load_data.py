import data.audio as audio
import data.image as image
import data.protein as protein
import data.video as video

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



def load_test_set(test_dir, 
                  test_idx,
                  dataset, 
                  feature_size,
                  patch, 
                  patch_sizes):
    if dataset == 'cifar':
        images = sorted(os.listdir(test_dir))
        data_paths = []
        for img_name in images:
            image_path = test_dir + "/" + img_name
            data_paths.append(image_path)
            test_start_idx = test_idx * 500
            test_end_idx = test_idx * 500 + 500
        test_images_path = data_paths[test_start_idx: test_end_idx]
        X, Y = image.load_image(test_images_path, 
                                feature_size,
                                patch, 
                                patch_sizes)
    if dataset == 'kodak':
        images = sorted(os.listdir(test_dir))
        data_paths = []
        for img_name in images:
            image_path = test_dir + "/" + img_name
            data_paths.append(image_path)
            test_start_idx = test_idx * 1
            test_end_idx = test_idx * 1 + 1
        test_images_path = data_paths[test_start_idx: test_end_idx]
        X, Y = image.load_image(test_images_path, 
                                feature_size,
                                patch, 
                                patch_sizes)
    if dataset == 'audio':
        with open(test_dir + '/test_dataset.pkl', 'wb') as f:
            test_tensor = pickle.load(f)
        X, Y = audio.load_audio([test_tensor[test_idx]], 
                                feature_size,
                                patch, 
                                patch_sizes)
    if dataset == 'video':
        with open(test_dir + '/test_dataset.pkl', 'wb') as f:
            test_tensor = pickle.load(f)
        X, Y = video.load_video([test_tensor[test_idx]], 
                                feature_size,
                                patch, 
                                patch_sizes)
    if dataset == 'protein':
        with open(test_dir + '/test_dataset.pkl', 'wb') as f:
            test_tensor = pickle.load(f)
        test_start_idx = test_idx * 1000
        test_end_idx = test_idx * 1000 + 1000
        test_tensor = test_tensor[test_start_idx: test_end_idx]
        X, Y = protein.load_protein(test_tensor, 
                                    feature_size,
                                    patch, 
                                    patch_sizes)

    return X, Y