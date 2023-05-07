import numpy as np

def softmax_ce_naive_forward_backward(X, W, y, reg):
    """Implémentation naive qui calcule la propagation avant, puis la
       propagation arrière pour finalement retourner la perte entropie croisée
       (ce) + une régularisation L2 et le gradient des poids. Utilise une 
       activation softmax en sortie.
       
       NOTE : la fonction codée est : EntropieCroisée + 0.5*reg*||W||^2
       
       N'oubliez pas le 0.5!

    Inputs:
    - X: Numpy array, shape (N, D). N représente le nombre d'exemple d'entrainement
        dans X, et D représente la dimension des exemples de X.
    - W: Numpy array, shape (D, C)
    - y: Numpy array, shape (N,). y[i] = c veut dire que X[i] appartient à la
        classe c, 0 <= c < C
    - reg: float. Terme de regularisation L2

    Outputs:
    - loss: float. Perte du classifieur linéaire softmax
    - dW: Numpy array, shape (D, C). Gradients des poids W
    """
    loss = 0.0
    dW = np.zeros(W.shape)
    N = X.shape[0]
    D = X.shape[1]
    C = W.shape[1]

   # compute (very) naive softmax
    for n in range(N):
        f = np.zeros(C)
        for c in range(C):
            sum = 0.0
            for d in range(D):
                sum += X[n,d] * W[d,c]
            f[c] = sum
        
        f -= np.max(f)
        e = np.exp(f)
        fSum = np.sum(e)
        S = e / fSum

        # compute loss
        loss += np.log(S[y[n]])

        # compute gradient with respect to W
        for c in range(C):
            if c == y[n]:
                dW[:,c] += (S[c] - 1) * X[n]
            else:
                dW[:,c] += S[c] * X[n]

    loss /= -N
    loss += 0.5*reg*np.linalg.norm(W)**2
    dW = dW/N + reg*W

    return loss, dW


def softmax_ce_forward_backward(X, W, y, reg):
    """Implémentation vectorisée qui calcule la propagation avant, puis la
       propagation arrière pour finalement retourner la perte entropie croisée
       (ce) et le gradient des poids. Utilise une activation softmax en sortie.
        
       NOTE : la fonction codée est : EntropieCroisée + 0.5*reg*||W||^2      
       N'oubliez pas le 0.5!

    Inputs:
    - X: Numpy array, shape (N, D). N représente le nombre d'exemples d'entrainement
        dans X, et D représente la dimension des exemples de X.
    - W: Numpy array, shape (D, C)
    - y: Numpy array, shape (N,). y[i] = c veut dire que X[i] appartient à la
        classe c, 0 <= c < C
    - reg: float. Terme de regularisation L2

    Outputs:
    - loss: float. Perte du classifieur linéaire softmax
    - dW: Numpy array, shape (D, C). Gradients des poids W
    """
    loss = 0.0
    dW = np.zeros(W.shape)
    N = X.shape[0]
    C = W.shape[1]
    
    # compute vectorized softmax
    f = X @ W
    f -= np.max(f)
    S = np.exp(f)
    S /= S.sum(axis=1, keepdims=True)

    # compute vectorized loss
    loss = - np.sum(np.log(S[:,y].diagonal())) / N + 0.5*reg*np.linalg.norm(W)**2

    # one hot vector
    t = np.zeros((N,C))
    t[np.arange(N),y] = 1

    # compute vectorized gradient with respect to W
    dW = X.transpose() @ (S - t)
    dW = dW/N + reg*W

    return loss, dW


def hinge_naive_forward_backward(X, W, y, reg):
    """Implémentation naive calculant la propagation avant, puis la
       propagation arrière, pour finalement retourner la perte hinge et le
       gradient des poids.
       
       NOTE : la fonction codée est : Hinge + 0.5*reg*||W||^2
       N'oubliez pas le 0.5!

    Inputs:
    - X: Numpy array, shape (N, D)
    - W: Numpy array, shape (D, C)
    - y: Numpy array, shape (N,). y[i] = c veut dire que X[i] appartient à la
         classe c, 0 <= c < C
    - reg: float. Terme de regularisation L2

    Outputs:
    - loss: float. Perte du classifieur linéaire hinge
    - dW: Numpy array, shape (D, C). Gradients des poids W
    """
    loss = 0.0
    dW = np.zeros(W.shape)
    N = X.shape[0]
    D = X.shape[1]
    C = W.shape[1]
    
    # iterate over all data
    for n in range(N):
        
        # compute prediction
        pred = np.zeros(C)
        for c in range(C):
            sum = 0.0
            for d in range(D):
                sum += X[n,d] * W[d,c]
            pred[c] = sum

        # get idx for each best prediction
        idxPred = np.argmax(pred)

        # compute loss
        loss += max(0, 1 + pred[idxPred] - pred[y[n]])

        # compute dW derivative
        if pred[idxPred] != pred[y[n]]:
            dW[:,idxPred] += X[n]
            dW[:,y[n]] -= X[n]

    # add regularization
    loss /= N
    loss += 0.5*reg*np.linalg.norm(W)**2
    dW = dW/N + reg*W

    return loss, dW


def hinge_forward_backward(X, W, y, reg):
    """Implémentation vectorisée calculant la propagation avant, puis la
       propagation arrière, pour finalement retourner la perte hinge et le
       gradient des poids.

       NOTE : la fonction codée est : Hinge + 0.5*reg*||W||^2
       N'oubliez pas le 0.5!

       RAPPEL hinge loss one-vs-one :
       loss = max(0, 1 + score_classe_predite - score_classe_vérité_terrain)
       
    Inputs:
    - X: Numpy array, shape (N, D)
    - W: Numpy array, shape (D, C)
    - y: Numpy array, shape (N,). y[i] = c veut dire que X[i] appartient à la
         classe c, 0 <= c < C
    - reg: float. Terme de regularisation L2

    Outputs:
    - loss: float. Perte du classifieur linéaire hinge
    - dW: Numpy array, shape (D, C). Gradients des poids W
    """
    loss = 0.0
    dW = np.zeros(W.shape)
    W_t = np.transpose(W)

    # loss vectorise
    XW = np.dot(X,W)
    classes_predites = np.argmax(XW, axis = 1)
    score_predicted_class = np.max(XW, axis = 1)
    mat = np.matrix(XW)
    score_true_class = mat[np.arange(len(y)),y]

    # Compute loss
    hinge = np.sum(np.maximum(np.zeros(len(X)), 1+ score_predicted_class - score_true_class))/ len(X) 
    loss = hinge +  0.5*reg*np.linalg.norm(W)**2

    grad = np.zeros((len(W_t),len(X)))
    is_right_class = classes_predites == y

    grad[y , np.arange(len(is_right_class))] = -1
    grad[classes_predites , np.arange(len(is_right_class))] = 1

    grad[:, np.where(is_right_class)] = 0

    # Compute gradient with respect to W
    dW = np.transpose(np.dot(grad,X))/len(X) + reg*W

    return loss, dW
