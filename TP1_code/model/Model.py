import numpy as np
from collections import OrderedDict


class Model:
    def __init__(self):
        self.layers = OrderedDict()
        self.loss_function = None

    def add(self, layer, name=None):
        layer_name = 'L' + str(len(self.layers)) if name is None else name
        self.layers[layer_name] = layer

    def add_loss(self, loss_function):
        self.loss_function = loss_function

    def forward(self, X, **kwargs):
        """Effectue la propagation avant pour l'ensemble des couches du modèle.
        Cette fonction retourne le score du réseau, c'est-à-dire la sortie avant
        le SOFTMAX.

        Arguments:
            X {ndarray} -- Entrée du réseau. Shape (N, dim_input)
            mode {str} -- Indique si le model doit s'exécuter en mode train
                          ou test. N'affecte que les couches batchnorm et
                          dropout. (default: {'train'})

        Returns:
            ndarray -- Scores du réseau (sortie du forward de la dernière couche).
                       Shape (N, C)
        """

        previous_output = X
        for layer in self.layers.values():
            previous_output = layer.forward(previous_output, **kwargs)

        return previous_output

    def backward(self, dOutput, **kwargs):
        """Effectue la rétro-propagation pour l'ensemble des couches du modèle.
           NOTE: les gradients dW et db de chaque couche sont calculés dans la 
           fonction *backward()* et stockés comme variables membres de la classe
           layer.
           
        Arguments:
            dOutput {ndarray} -- dérivée de la loss par rapport au score du modèle.
                                 Shape (N, C)
        Returns:
            dA {ndarray} -- Dérivée de la loss par rapport à la couche d'entrée du 
                            réseau
        """

        dA = dOutput
        for layer in reversed(list(self.layers.values())):
            dA = layer.backward(dA, **kwargs)

        return dA

    def calculate_loss(self, model_output, targets, reg):
        """Calcule la loss du modèle.

        Arguments:
            model_output {ndarray} -- Scores calculés par la propagation avant.
                                      Shape (N, C)
            targets {ndarray} -- Labels attendus pour nos échantillons d'entraînement.
                                 Shape (N, )
            reg {float} -- Terme de régularisation.

        Returns:
            tuple -- Tuple contenant la loss et la dérivée de la loss par rapport
                     aux scores.
        """

        # On change la valeur de reg de chaque couche pour qu'elle soit la même
        # dans tout le modèle.
        for layer in self.layers.values():
            layer.reg = reg

        # Retourne un tuple contenant la loss et les gradients de la loss par
        # rapport au output de la derniere couche.
        return self.loss_function(model_output, targets, reg, self.parameters())

    def parameters(self):
        """Permet d'obtenir les paramètres W et b de chaque couche du modèle.

        Returns:
            dict -- Paramètres du modèle, regroupés par couche.
        """

        params = {}
        for name, layer in self.layers.items():
            params[name] = layer.get_params()

        return params

    def gradients(self):
        """Permet d'obtenir les gradients dW et db de chaque couche du modèle.

        Returns:
            dict -- Gradients du modèle, regroupés par couche.
        """

        gradients = {}
        for name, layer in self.layers.items():
            gradients[name] = layer.get_gradients()

        return gradients

    def predict(self, X):
        scores = self.forward(X, mode='test')
        return np.argmax(scores, axis=1)

    def reset(self):
        for layer in self.layers.values():
            layer.reset()
