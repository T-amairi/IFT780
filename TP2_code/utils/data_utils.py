# Code adapté de projets académiques de la professeur Fei Fei Li et
# de ses étudiants Andrej Karpathy, Justin Johnson et autres.
# Version finale rédigée par Carl Lemaire, Vincent Ducharme et Pierre-Marc Jodoin
# Modifications subséquentes par Antoine Théberge


import pickle as pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread


def load_CIFAR_batch_file(filename):
    """ charge une batch de cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        data = datadict['data']
        labels = datadict['labels']
        data = data.reshape(10000, 3, 32, 32).transpose(
            0, 2, 3, 1).astype("float")
        labels = np.array(labels)
        return data, labels


def load_CIFAR10(ROOT):
    """ charge la totalité de cifar """
    all_data = []
    all_labels = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        data, labels = load_CIFAR_batch_file(f)
        all_data.append(data)
        all_labels.append(labels)
    concat_data = np.concatenate(all_data)
    concat_labels = np.concatenate(all_labels)
    del data, labels
    data_test, labels_test = load_CIFAR_batch_file(
        os.path.join(ROOT, 'test_batch'))

    with open(os.path.join(ROOT, 'batches.meta'), 'rb') as f:
        label_names = pickle.load(f, encoding='latin1')['label_names']

    return concat_data, concat_labels, data_test, labels_test, label_names


def load_tiny_imagenet(path, dtype=np.float32):
    """
    Charge TinyImageNet. TinyImageNet-100-A, TinyImageNet-100-B, and
    TinyImageNet-200 ont la même structure de répertoires, cette fonction peut
    donc être utilisée pour charger n'importe lequel d'entre eux.

    Inputs:
    - path: String du path vers le répertoire à charger.
    - dtype: numpy datatype utilisé pour charger les données.

    Returns: Un tuple de
    - class_names: list, class_names[i] étant une liste de string donnant les
      noms WordNet pour classe i dans le dataset.
    - X_train: (N_tr, 3, 64, 64) array, contient les images d'entraînement
    - y_train: (N_tr,) array, contient les labels d'entraînement
    - X_val: (N_val, 3, 64, 64) array, contient les images de validation
    - y_val: (N_val,) array, contient les labels de validation
    - X_test: (N_test, 3, 64, 64) array, contient le images de test.
    - y_test: (N_test,) array, contient les labels de test; si les labels ne
    sont pas disponibles, y_test = None
    """
    # Charge wnids
    with open(os.path.join(path, 'wnids.txt'), 'r') as f:
        wnids = [x.strip() for x in f]

    # Map wnids aux labels
    wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

    # Utilise words.txt pour obtenir les noms de chaque classe
    with open(os.path.join(path, 'words.txt'), 'r') as f:
        wnid_to_words = dict(line.split('\t') for line in f)
        for wnid, words in wnid_to_words.iteritems():
            wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
    class_names = [wnid_to_words[wnid] for wnid in wnids]

    # Charge les données d'entraînement.
    X_train = []
    y_train = []
    for i, wnid in enumerate(wnids):
        if (i + 1) % 20 == 0:
            print('loading training data for synset %d / %d' %
                  (i + 1, len(wnids)))

        boxes_file = os.path.join(path, 'train', wnid, '%s_boxes.txt' % wnid)
        with open(boxes_file, 'r') as f:
            filenames = [x.split('\t')[0] for x in f]
        num_images = len(filenames)

        X_train_block = np.zeros((num_images, 3, 64, 64), dtype=dtype)
        y_train_block = wnid_to_label[wnid] * \
            np.ones(num_images, dtype=int64)
        for j, img_file in enumerate(filenames):
            img_file = os.path.join(path, 'train', wnid, 'images', img_file)
            img = imread(img_file)
            if img.ndim == 2:
                # image grayscale
                img.shape = (64, 64, 1)
            X_train_block[j] = img.transpose(2, 0, 1)
        X_train.append(X_train_block)
        y_train.append(y_train_block)

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # Charge les données de validation
    with open(os.path.join(path, 'val', 'val_annotations.txt'), 'r') as f:
        img_files = []
        val_wnids = []
        for line in f:
            img_file, wnid = line.split('\t')[:2]
            img_files.append(img_file)
            val_wnids.append(wnid)
        num_val = len(img_files)
        y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])
        X_val = np.zeros((num_val, 3, 64, 64), dtype=dtype)
        for i, img_file in enumerate(img_files):
            img_file = os.path.join(path, 'val', 'images', img_file)
            img = imread(img_file)
            if img.ndim == 2:
                img.shape = (64, 64, 1)
            X_val[i] = img.transpose(2, 0, 1)

    # Charge les données de test
    img_files = os.listdir(os.path.join(path, 'test', 'images'))
    X_test = np.zeros((len(img_files), 3, 64, 64), dtype=dtype)
    for i, img_file in enumerate(img_files):
        img_file = os.path.join(path, 'test', 'images', img_file)
        img = imread(img_file)
        if img.ndim == 2:
            img.shape = (64, 64, 1)
        X_test[i] = img.transpose(2, 0, 1)

    y_test = None
    y_test_file = os.path.join(path, 'test', 'test_annotations.txt')
    if os.path.isfile(y_test_file):
        with open(y_test_file, 'r') as f:
            img_file_to_wnid = {}
            for line in f:
                line = line.split('\t')
                img_file_to_wnid[line[0]] = line[1]
        y_test = [wnid_to_label[img_file_to_wnid[img_file]]
                  for img_file in img_files]
        y_test = np.array(y_test)

    return class_names, X_train, y_train, X_val, y_val, X_test, y_test


def load_models(models_dir):
    """
    Charge les modèles sauvegardés sur disque. Va tenter de unpickle tous les
    fichier d'un répertoire, ceux qui causent une erreur seront ignorés.

    Inputs:
    - models_dir: String, path vers le répertoire qui contient les modèles.
      Chaque fichier de modèle est un "pickled dictionnary" avec le champ
      'model'.

    Outputs:
    Un dictionnaire qui map les noms de fichiers de modèles aux modèles.
    """
    models = {}
    for model_file in os.listdir(models_dir):
        with open(os.path.join(models_dir, model_file), 'rb') as f:
            try:
                models[model_file] = pickle.load(f)['model']
            except pickle.UnpicklingError:
                continue
    return models


def display_images(data, preds, label_names):
    nb_images = (4, 12)

    indices = np.random.randint(0, len(data), size=nb_images)

    plt.axis('off')
    f, axarr = plt.subplots(*(nb_images))

    for x in range(nb_images[0]):
        for y in range(nb_images[1]):
            im = (data[indices[x, y]].transpose(1, 2, 0)).astype(int)
            lb = label_names[preds[indices[x, y]].astype(int)]
            axarr[x][y].imshow(im)
            axarr[x][y].set_title(lb)
            axarr[x][y].axis('off')


def display_points(X, y):
    # Centrer et réduire les données (moyenne = 0, écart-type = 1)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    x_ = (X - mean) / std

    # Afficher
    plt.figure(figsize=(8, 6))
    plt.scatter(x_[:, 0], x_[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.show()
