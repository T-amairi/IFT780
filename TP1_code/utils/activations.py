import numpy as np


def relu_forward(Z):
    return np.maximum(0, Z)


def relu_backward(Z):
    return (Z > 0).astype(int)


def identity_forward(Z):
    return Z


def identity_backward(Z):
    return 1


def get_activation(activation):
    """Permet d'obtenir l'activation associée au paramètre activation
       et sa fonction de dérivation.

    Arguments:
        activation {str} -- Identifiant de l'activation demandée.

    Returns:
        dict -- Dictionnaire contenant l'activation et sa fonction de 
                dérivation.
    """

    if activation == 'identity':
        return {'forward': identity_forward, 'backward': identity_backward}
    elif activation == 'relu':
        return {'forward': relu_forward, 'backward': relu_backward}
    else:
        raise Exception("Not a valid activation function")
