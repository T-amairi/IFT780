import numpy as np
from abc import abstractmethod
from utils.cython.im2col import col2im, im2col
from utils.activations import get_activation


class Conv2D:
    def __init__(self, num_filters,
                 filter_size=3,
                 channels=1,
                 stride=1,
                 padding=0,
                 weight_scale=1e-3,
                 activation='identity'):
        """
        Keyword Arguments:
            num_filters {int} -- nombre de cartes d'activation.
            filter_size {int, tuple} -- taille des filtres. (default: {3})
            channels {int} -- nombre de canaux. Doit être égal au nombre
                              de canaux des données en entrée. (default: {1})
            stride {int, tuple} -- taille de la translation des filtres. (default: {1})
            padding {int, tuple} -- nombre de zéros à rajouter avant et
                                    après les données. La valeur représente
                                    seulement les zéros d'un côté. (default: {0})
            weight_scale {float} -- écart type de la distribution normale utilisée
                                    pour l'initialisation des weights. (default: {1e-4})
            activation {str} -- identifiant de la fonction d'activation de la couche
                                (default: {'identite'})
        """

        self.num_filters = num_filters
        self.filter_size = filter_size
        self.channels = channels
        self.weight_scale = weight_scale
        self.activation_id = activation
        
        if isinstance(stride, tuple):
            self.stride = stride
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            raise Exception("Invalid stride format, must be tuple or integer")

        if isinstance(padding, tuple):
            self.pad = padding
        elif isinstance(padding, int):
            self.pad = (padding, padding)
        else:
            raise Exception("Invalid padding format, must be tuple or integer")

        if not isinstance(channels, int):
            raise Exception("Invalid channels format, must be integer")

        if isinstance(filter_size, tuple):
            self.W = np.random.normal(loc=0.0, scale=weight_scale, size=(num_filters, channels, filter_size[0],
                                                                         filter_size[1]))
        elif isinstance(filter_size, int):
            self.W = np.random.normal(loc=0.0, scale=weight_scale, size=(num_filters, channels, filter_size,
                                                                         filter_size))
        else:
            raise Exception("Invalid filter format, must be tuple or integer")

        self.b = np.zeros(num_filters)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)
        self.reg = 0.0
        self.cache = None

        self.activation = get_activation(activation)

    @abstractmethod
    def forward(self, X, **kwargs):
        pass

    @abstractmethod
    def backward(self, dA, **kwargs):
        pass

    def get_params(self):
        return {'W': self.W, 'b': self.b}

    def get_gradients(self):
        return {'W': self.dW, 'b': self.db}


class Conv2DNaive(Conv2D):

    def forward(self, X, **kwargs):
        """Effectue la propagation avant naïvement.

        Arguments:
            X {ndarray} -- Input de la couche. Shape (N, C, H, W)

        Returns:
            ndarray -- Scores de la couche
        """

        XPad = np.pad(X, ((0,0),(0,0),self.pad,self.pad), 'constant')

        N, channel, height, width = XPad.shape
        F, Fchannel, Fheight, Fwidth = self.W.shape

        ConvH, ConvW = (height - Fheight) // self.stride[0] + 1, (width - Fwidth) // self.stride[1] + 1
        H = np.zeros(shape=(N, F, ConvH, ConvW))

        # per input 
        for n in range(N):
            # per filter
            for f in range(F):
                # per channel
                for c in range(channel): 
                    x = XPad[n,c]
                    w = self.W[f,c]
                    # 2D convolution
                    for i in range(ConvH):
                        startH = i * self.stride[0] 
                        endH = startH + Fheight
                        for j in range(ConvW):
                            startW = j * self.stride[1] 
                            endW = startW + Fwidth
                            H[n,f,i,j] += w.ravel() @ x[startH:endH,startW:endW].ravel()

                # add bias
                H[n,f] += self.b[f]
                
        A = self.activation['forward'](H)
        self.cache = XPad, H, height, width
        return A

    def backward(self, dA, **kwargs):
        """Effectue la rétropropagation

        Arguments:
            dA {ndarray} -- Dérivée de la loss par rapport au output de la couche.
                            Shape (N, F, out_height, out_width)

        Returns:
            ndarray -- Dérivée de la loss par rapport au input de la couche.
        """
        X_col, out, height, width = self.cache
        N, F, out_height, out_width = dA.shape
        _, Fchannel, Fheight, Fwidth = self.W.shape

        dX = np.zeros((N, Fchannel, height, width))
        dH = dA * self.activation['backward'](out)
        self.dW = np.zeros(self.W.shape)

        # per input
        for n in range(N):
            # per filter
            for f in range(F):
                # per channel
                for c in range(Fchannel):
                    # 2D convolution for dW
                    x = X_col[n,c]
                    for i in range(Fheight):
                        startH = i * self.stride[0]
                        endH = startH + out_height
                        for j in range(Fwidth):
                            startW = j * self.stride[1]
                            endW = startW + out_width
                            self.dW[f,c,i,j] += dH[n,f].ravel() @ x[startH:endH,startW:endW].ravel()

                    # 2D full convolution for dX
                    w = np.rot90(self.W[f,c],2)
                    dhPad = np.pad(dH[n,f], ((Fheight - 1,Fheight - 1), (Fwidth - 1,Fwidth - 1)), 'constant')
                    height, width = (dhPad.shape[0] - Fheight) // self.stride[0] + 1, (dhPad.shape[1] - Fwidth) // self.stride[1] + 1
                    for i in range(height):
                        startH = i * self.stride[0]
                        endH = startH + Fheight
                        for j in range(width):
                            startW = j * self.stride[1]
                            endW = startW + Fwidth
                            dX[n,c,i,j] += dhPad[startH:endH,startW:endW].ravel() @ w.ravel()

        self.db = np.sum(dH,axis=(0,2,3))
        return dX


class Conv2DMat(Conv2D):

    def forward(self, X, **kwargs):
        """Effectue la propagation en vectorisant.

        Arguments:
            X {ndarray} -- entrée de la couche. Shape (N, C, H, W)

        Returns:
            ndarray -- Scores de la couche
        """

        N, channel, height, width = X.shape
        F, Fchannel, Fheight, Fwidth = self.W.shape

        assert channel == Fchannel
        assert (height - Fheight + 2 * self.pad[0]) % self.stride[0] == 0
        assert (width - Fwidth + 2 * self.pad[1]) % self.stride[1] == 0

        out_height = np.uint32(1 + (height - Fheight + 2 * self.pad[0]) / self.stride[0])
        out_width = np.uint32(1 + (width - Fwidth + 2 * self.pad[1]) / self.stride[1])
        out = np.zeros((N, F, out_height, out_width))

        X_padded = np.pad(X, ((0, 0), (0, 0), (self.pad[0], self.pad[0]), (self.pad[1], self.pad[1])), 'constant')
        
        W_row = self.W.reshape(F, Fchannel*Fheight*Fwidth)

        X_col = np.zeros((N, Fchannel*Fheight*Fwidth, out_height*out_width))
        for index in range(N):
            col = 0
            for i in range(0, height + 2 * self.pad[0] - Fheight + 1, self.stride[0]):
                for j in range(0, width + 2 * self.pad[1] - Fwidth + 1, self.stride[1]):
                    X_col[index, :, col] = X_padded[index, :, i:i+Fheight, j:j+Fwidth]\
                        .reshape(Fchannel*Fheight*Fwidth)
                    col += 1
            out[index] = (W_row.dot(X_col[index]) + self.b.reshape(F, 1)).reshape(F, out_height, out_width)

        self.cache = (X_col, out, height, width)        

        A = self.activation['forward'](out)
        
        return A

    def backward(self, dA, **kwargs):
        """Effectue la rétropropagation en vectorisant.

        Arguments:
            dA {ndarray} -- Dérivée de la loss par rapport à la sortie de la couche.
                            Shape (N, F, out_height, out_width)

        Returns:
            ndarray -- Dérivée de la loss par rapport au input de la couche.
        """

        X_col, out, height, width = self.cache

        N, F, out_height, out_width = dA.shape
        _, Fchannel, Fheight, Fwidth = self.W.shape
        
        pad_height = height + 2 * self.pad[0]
        pad_width = width + 2 * self.pad[1]

        # initialiser dW et db avec le facteur de régularisation
        self.dW = self.reg * self.W
        self.db = self.reg * self.b

        dX = np.zeros((N, Fchannel, height, width))

        W_row = self.W.reshape(F, Fchannel * Fheight * Fwidth)

        dOut = self.activation['backward'](out) * dA

        for index in range(N):
            dOut_row = dOut[index].reshape(F, out_height * out_width)
            dX_col = W_row.T.dot(dOut_row)
            dX_block = np.zeros((Fchannel, pad_height, pad_width))

            col = 0
            for i in range(0, pad_height - Fheight + 1, self.stride[0]):
                for j in range(0, pad_width - Fwidth + 1, self.stride[1]):
                    dX_block[:, i:i+Fheight, j:j+Fwidth] += dX_col[:, col].reshape(Fchannel, Fheight, Fwidth)
                    col += 1

            if self.pad[0] > 0 and self.pad[1] > 0:
                dX[index] = dX_block[:, self.pad[0]:-self.pad[0], self.pad[1]:-self.pad[1]]
            elif self.pad[0] > 0:
                dX[index] = dX_block[:, self.pad[0]:-self.pad[0], :]
            elif self.pad[1] > 0:
                dX[index] = dX_block[:, :, self.pad[1]:-self.pad[1]]
            else:
                dX[index] = dX_block

            self.dW += dOut_row.dot(X_col[index].T).reshape(F, Fchannel, Fheight, Fwidth)
            self.db += dOut_row.sum(axis=1)

        return dX


class Conv2DCython(Conv2D):

    def forward(self, X, **kwargs):
        """Effectue la propagation avant cythonisée.

        Arguments:
            X {ndarray} -- Input de la couche. Shape (N, C, H, W)

        Returns:
            ndarray -- Scores de la couche
        """

        N, channel, height, width = X.shape
        F, Fchannel, Fheight, Fwidth = self.W.shape

        assert channel == Fchannel
        assert (height - Fheight + 2 * self.pad[0]) % self.stride[0] == 0
        assert (width - Fwidth + 2 * self.pad[1]) % self.stride[1] == 0

        out_height = np.uint32(1 + (height - Fheight + 2 * self.pad[0]) / self.stride[0])
        out_width = np.uint32(1 + (width - Fwidth + 2 * self.pad[1]) / self.stride[1])

        W_row = self.W.reshape(F, Fchannel*Fheight*Fwidth)

        X_col = np.asarray(im2col(X, N, channel, height, width,
                                  Fheight, Fwidth, 
                                  self.pad[0], self.pad[1], 
                                  self.stride[0], self.stride[1]))

        out = (W_row.dot(X_col) + self.b.reshape(F, 1))
        out = out.reshape(F, N, out_height, out_width).transpose(1, 0, 2, 3)

        self.cache = (X_col, out, height, width)

        A = self.activation['forward'](out)

        return A

    def backward(self, dA, **kwargs):
        """Effectue la rétropropagation cythonisée.

        Arguments:
            dA {ndarray} -- Dérivée de la loss par rapport au output de la couche.
                            Shape (N, F, out_height, out_width)

        Returns:
            ndarray -- Dérivée de la loss par rapport au input de la couche.
        """

        X_col, out, height, width = self.cache
        N, F, out_height, out_width = dA.shape
        _, Fchannel, Fheight, Fwidth = self.W.shape

        W_row = self.W.reshape(F, Fchannel * Fheight * Fwidth)

        dOut = self.activation['backward'](out) * dA
        dOut_mat = dOut.transpose(1, 0, 2, 3).reshape(F, N * out_height * out_width)

        self.dW = dOut_mat.dot(X_col.T).reshape(self.W.shape)
        self.dW += self.reg * self.W

        self.db = dOut_mat.sum(axis=1) 
        self.db += self.reg * self.b

        dX_col = W_row.T.dot(dOut_mat)
        dX = col2im(dX_col, N, Fchannel, height, width, Fheight, Fwidth, 
                    self.pad[0], self.pad[1], self.stride[0], self.stride[1])

        return np.asarray(dX)

    def reset(self):
        self.__init__(self.num_filters,
                      filter_size=self.filter_size,
                      channels=self.channels,
                      stride=self.stride,
                      padding=self.pad,
                      weight_scale=self.weight_scale,
                      activation=self.activation_id)

