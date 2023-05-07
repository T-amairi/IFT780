import numpy as np


class Dropout:
    def __init__(self, drop_rate=0.2):
        """
        Keyword Arguments:
            drop_rate {float} -- pourcentage de neurones qui ne sont pas activés
                                 à l'entrainement (default: {0.2})
        """
        self.drop_rate = drop_rate

        self.cache = None

    def forward(self, X, **kwargs):
        """Application du dropout inversé lors de la propagation avant.

        Arguments:
            X {ndarray} -- Outputs de la couche précédente.

        Keyword Arguments:
            **kwargs -- Utilisé pour indiquer si le forward
                        s'applique à l'entrainement ou au test
                        et pour inclure un seed (default: {'train', None})
        Returns:
            ndarray -- Scores de la couche
        """

        mode = kwargs.get('mode', 'train')
        seed = kwargs.get('seed', None)

        np.random.seed(seed)
        drop_mask = None  
        A = None

        if mode == 'train' and self.drop_rate:
            drop_mask = np.random.binomial(1,1 - self.drop_rate,size=X.shape) 
            A = X * drop_mask / (1 - self.drop_rate)
        elif mode == 'test' or not self.drop_rate:
            A = X
        else:
            raise Exception("Invalid forward mode %s" % mode)

        self.cache = drop_mask

        return A

    def backward(self, dA, **kwargs):
        """Rétro-propagation pour la couche de dropout inversé.

        Arguments:
            dA {ndarray} -- Gradients de la loss par rapport aux sorties.

        Keyword Arguments:
            **kwargs -- Utilisé pour indiquer si le forward
                        s'applique à l'entrainement ou au test (default: {'train'})
        Returns:
            ndarray -- Dérivée de la loss par rapport au input de la couche.
        """

        mode = kwargs.get('mode', 'train')

        if mode == 'train' and self.drop_rate:
            drop_mask = self.cache
            dX = dA * drop_mask / (1 - self.drop_rate)
        elif mode == 'test' or not self.drop_rate:
            dX = dA
        else:
            raise Exception("Invalid forward mode %s" % mode)

        return dX

    def get_params(self):
        return {}

    def get_gradients(self):
        return {}

    def reset(self):
        self.__init__(drop_rate=self.drop_rate)
