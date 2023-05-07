import numpy as np


def solver(X_train, y_train, X_val, y_val, reg, optimizer, lr_decay=1.0, num_iter=100, batch_size=200, verbose=True):
    """Boucle d'entraînement générale d'un modèle. La méthode de descente de gradient varie
       en fonction de l'optimizer.

    Arguments:
        X_train {ndarray} -- Données d'entraînement.
                             Shape (N, D)
        y_train {ndarray} -- Labels d'entraînement.
                             Shape (N, )
        X_val {ndarray} -- Données de validation.
                           Shape (N_val, D)
        y_val {ndarray} -- Labels de validation
                           Shape (N_val, )
        reg {float} -- Terme de régularisation
        optimizer {Optimizer} -- Objet permettant d'optimizer les paramètres du modèle.

    Keyword Arguments:
        lr_decay {float} -- Paramètre de contrôle pour la diminution du taux d'apprentissage.
                            (default: {1.0})
        num_iter {int} -- Nombre d'itérations d'entraînement (default: {100})
        batch_size {int} -- Nombre de données d'entraînement passées au modèle par itération.
                            (default: {200})
        verbose {bool} -- (default: {True})

    Returns:
        tuple -- Tuple contenant l'historique de la loss et de l'accuracy d'entraînement et de
                 validation
    """

    model = optimizer.model

    N = len(y_train)
    iterations_per_epoch = max(N / batch_size, 1)

    loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []
    list_batches = sample(X_train,y_train,batch_size)
    batches_number = len(list_batches)
    
    for it in range(num_iter):
       
        # All batch version (it works) but too long to execute 
        # # 1. Create batch
        # batch_loss = []
        # for X_batch, y_batch in list_batches :
        #     scores = model.forward(X_batch)

        #     # 2. Compute loss and gradient
        #     loss, dScores, _ = model.calculate_loss(scores, y_batch, reg)
        #     batch_loss.append(loss)
        #     model.backward(dScores)

        #     # 3. do a gradient step
        #     optimizer.step()

        # loss = np.mean(batch_loss)
        # loss_history.append(loss)

        # second version : get a random batch for each iteration
        idx = np.random.randint(0, batches_number)
        X_batch = list_batches[idx][0]
        y_batch = list_batches[idx][1]
            
        scores = model.forward(X_batch)
        loss, dScores, _ = model.calculate_loss(scores, y_batch, reg)
        model.backward(dScores)
        optimizer.step()

        loss_history.append(loss)

        if verbose and it % 500 == 0:
            print('iteration %d / %d: loss %f' % (it, num_iter, loss))

        # Périodiquement, on évalue l'accuracy d'entraînement et de
        # validation, et on diminue le learning rate (lr_decay = 1.0
        # veut dire que le learning rate reste fixe)
        if it % iterations_per_epoch == 0:
            train_accuracy = (model.predict(X_train) == y_train).mean()
            val_accuracy = (model.predict(X_val) == y_val).mean()
            train_accuracy_history.append(train_accuracy)
            val_accuracy_history.append(val_accuracy)

            optimizer.lr *= lr_decay

    return loss_history, train_accuracy_history, val_accuracy_history



def check_accuracy(X, y, batch_size, model):
    batches = sample(X, y, batch_size)

    accuracy = np.array([])

    for X_batch, y_batch in batches:
        accuracy = np.append(accuracy, (model.predict(X_batch) == y_batch))

    return accuracy.mean()


def sample(X, y, batch_size):
    """Permet de générer des batch d'entraînement pour une epoch.
        Cette fonction "découpe" les données dans X et y en une série
        de batches.
    Arguments:
        X {ndarray} -- Données d'entraînement.
                       Shape (N, D)
        y {ndarray} -- Labels d'entraînement.
                       Shape (N, )
        batch_size {int} -- Taille d'une batch d'entraînement.

    Returns:
        list -- Liste contenant des tuples (X_batch, y_batch).
    """

    N = len(y)
    mod = N % batch_size

    if N <= batch_size:
        return [(X, y)]

    shuffle_indices = np.random.permutation(N)

    X_shuffle = X[shuffle_indices]
    y_shuffle = y[shuffle_indices]

    X_split = np.split(X_shuffle[:N - mod], int(N) // int(batch_size))
    y_split = np.split(y_shuffle[:N - mod], int(N) // int(batch_size))

    if N - mod < N:
        X_split.append(X_shuffle[N - mod:])
        y_split.append(y_shuffle[N - mod:])

    return [(X_split[i], y_split[i]) for i in range(len(X_split))]


class SGD:
    def __init__(self, lr, model, momentum=0):
        self.initial_lr = lr

        self.lr = lr
        self.model = model
        self.momentum = momentum
        self.velocity = {}

        params = self.model.parameters()
        for layer_name, layer_params in params.items():
            tmp = {}

            for param_name, param in layer_params.items():
                tmp[param_name] = np.zeros(param.shape)

            self.velocity[layer_name] = tmp

    def step(self):
        """Applique l'algorithme de descente de gradient par batch
           à l'ensemble des paramètres du modèle.
        """

        params = self.model.parameters()
        grads = self.model.gradients()

        for layer_name, layer_params in params.items():
            for param_name, param in layer_params.items():
                self.velocity[layer_name][param_name] *= self.momentum
                self.velocity[layer_name][param_name] += self.lr * grads[layer_name][param_name]
                param -= self.velocity[layer_name][param_name]

    def set_lr(self, lr):
        self.initial_lr = lr
        self.lr = lr

    def reset(self):
        self.__init__(self.initial_lr,
                      self.model,
                      momentum=self.momentum)


class Adam:
    def __init__(self, lr, model, beta1=0.9, beta2=0.999, eps=1e-8):
        self.initial_lr = lr

        self.lr = lr
        self.model = model
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = {}
        self.v = {}
        self.t = 1

        params = self.model.parameters()
        for layer_name, layer_params in params.items():
            tmp_v = {}
            tmp_m = {}

            for param_name, param in layer_params.items():
                tmp_v[param_name] = np.zeros(param.shape)
                tmp_m[param_name] = np.zeros(param.shape)

            self.v[layer_name] = tmp_v
            self.m[layer_name] = tmp_m

    def step(self):
        """Applique l'algorithme de descente de gradient Adam
           à l'ensemble des paramètres du modèle.
        """

        params = self.model.parameters()
        grads = self.model.gradients()

        for layer_name, layer_params in params.items():
            for param_name, param in layer_params.items():
                self.m[layer_name][param_name] = self.beta1 * self.m[layer_name][param_name] + \
                                                 (1 - self.beta1) * \
                                                 grads[layer_name][param_name]
                self.v[layer_name][param_name] = self.beta2 * self.v[layer_name][param_name] + \
                                                 (1 - self.beta2) * \
                                                 (grads[layer_name][param_name] ** 2)

                m_corrected = self.m[layer_name][param_name] / (1 - self.beta1 ** self.t)
                v_corrected = self.v[layer_name][param_name] / (1 - self.beta2 ** self.t)

                param -= self.lr * m_corrected / (np.sqrt(v_corrected) + self.eps)

        self.t += 1

    def set_lr(self, lr):
        self.initial_lr = lr
        self.lr = lr

    def reset(self):
        self.__init__(self.initial_lr,
                      self.model,
                      beta1=self.beta1,
                      beta2=self.beta2,
                      eps=self.eps)


