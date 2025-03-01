import sys
sys.path.append("src/function_help")
from import_library_and_function import *

### SNMF

def encode_X_with_ones(K):
    K_encoded = np.c_[np.ones((K.shape[0], 1)), K]
    return K_encoded


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def update_K(Y, K, X):
    numerator = Y.dot(X.T)
    denominator = K.dot(X).dot(X.T) + 1e-10
    K = K * (numerator / denominator)
    return K


def update_weight(u, K, weight, E_g2, E_delta_2, alpha, epsilon):
    # Get number of samples
    n = u.shape[0]
    
    # Encode K
    W = encode_X_with_ones(K)
    
    j_list = list(range(n))
    random.shuffle(j_list)
    for j in j_list:
        w_i = W[j, :].reshape(1, -1)
        u_i = u[j].reshape(1, -1)

        # Compute gradient
        h = sigmoid(w_i.dot(weight))
        gradient = w_i.T.dot(h - u_i).flatten()

        # Accumulate gradient
        E_g2 = alpha * E_g2 + (1 - alpha) * gradient**2

        # Compute update
        delta = - (np.sqrt(E_delta_2 + epsilon) / np.sqrt(E_g2 + epsilon)) * gradient

        # Accumulate updates
        E_delta_2 = alpha * E_delta_2 + (1 - alpha) * delta**2

        # Apply update
        weight += delta

    return weight, E_g2, E_delta_2


def update_X(Y, K, X, E_g2, E_delta_2, alpha, epsilon, u, weight, epsStab):
    # Calculate Gradient
    gradient = K.T.dot(K.dot(X) - Y)

    # Accumulate gradient
    E_g2 = alpha * E_g2 + (1 - alpha) * gradient**2

    # Compute update
    delta = - (np.sqrt(E_delta_2 + epsilon) / np.sqrt(E_g2 + epsilon)) * gradient

    # Accumulate updates
    E_delta_2 = alpha * E_delta_2 + (1 - alpha) * delta**2
    
    # Check loss if update
    old_loss = cost_function(Y, K, X, u, weight)
    check = True
    count_check = 0
    while(check):
        new_loss = cost_function(Y, K, X + delta, u, weight)
        if new_loss > old_loss:
            delta /= 2
            count_check += 1
            if count_check > 100:
                check = False
        else:
            check = False

    # Apply update
    X += delta
    
    # For improved stability
    X[X <= 0] = epsStab

    return X, E_g2, E_delta_2

def cost_function(Y, K, X, u = np.array([]), weight = np.array([])):
    # Numfer of samples
    n = Y.shape[0]
    
    # Encode K
    W = encode_X_with_ones(K)

    # NMF Loss
    loss1 = (1 / 2) * sum((Y - K.dot(X)).flatten()**2)
    
    # Classification model loss
    if len(u) == 0 or len(weight) == 0:
        loss2 = 0
    else:
        loss2 = (1 / n) * ( sum(np.log(1 + np.exp(W.dot(weight)))) - sum(W.dot(weight) * u) )
    
    # SNMF loss
    loss = loss1 + loss2
    
    return loss, loss1, loss2

def SNMF(Y, u, rank, iter, tolerance, patience, epsStab, alpha, epsilon, init_mode):
    ######################## Initialize #######################
    if init_mode in ['nndsvd', 'nndsvda', 'nndsvdar']:
        K, X = init_NMF(Y, rank, init_mode)
    else:
        K, X = init_NMF(Y, rank, 'random')

    
    # Weight
    W = encode_X_with_ones(K)
    weight = np.zeros(W.shape[1])
    
    # Accumulators for X
    E_g2_X = np.zeros(X.shape)
    E_delta_X2 = np.zeros(X.shape)
    
    # Accumulators for weight
    E_g2_weight = np.zeros(weight.shape)
    E_delta_weight2 = np.zeros(weight.shape)
    
    # Lost history
    loss_list = []
    loss_list1 = []
    loss_list2 = []

    # X history
    X_history = []

    # weight history
    weight_history = []

    # Early stopping
    best_loss = np.inf
    no_improvement_count = 0
    


    ######################## Loop iteractions #######################
    for i in range(iter):
        
        # Update X
        X, E_g2_X, E_delta_X2 = update_X(Y, K, X, E_g2_X, E_delta_X2, 
                                         alpha, epsilon, 
                                         u, weight, epsStab)

        # Save X to cache
        X_history.append(X)
        
        # Update K
        K = update_K(Y, K, X)
        
        # Update weight
        weight, E_g2_weight, E_delta_weight2 = update_weight(u, K, weight, E_g2_weight, E_delta_weight2, 
                                                                 alpha, epsilon)

        weight_history.append(weight)
        
        # Loss function
        loss, loss1, loss2 = cost_function(Y, K, X, u, weight)
        loss_list.append(loss)
        loss_list1.append(loss1)
        loss_list2.append(loss2)
        print("Iteration {},  loss: {:.6f} [{:.6f}, {:.6f}]".format(i, loss, loss1, loss2))

        # Early stopping
        if best_loss - loss > tolerance:
            best_loss = loss
            no_improvement_count = 0
            X_history = X_history[-1:]
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print("Early stopping at iteration {}".format(i))
            break
    best_case = - no_improvement_count - 1
    return X_history[best_case], loss_list, loss_list1, loss_list2, weight_history[best_case]


def SNMF_transform(Y, X, iter, tolerance, patience, epsStab, alpha, epsilon, u = np.array([]), weight = np.array([])):
    tolerance /= 10
    
    # Number of rank
    rank = X.shape[0]
    
    ######################## Initialize #######################
    K = init_W(Y, rank)
    
    # Lost history
    loss_list = []
    loss_list1 = []
    loss_list2 = []

    # K history
    K_history = []
    
    # Early stopping
    best_loss = np.inf
    no_improvement_count = 0
    

    
    ######################## Loop iteractions #######################
    for i in range(iter):
        
        # Update K
        K = update_K(Y, K, X)

        # Save K to cache
        K_history.append(K)

        # Loss function
        if len(u) == 0 or len(weight) == 0:
            loss, loss1, loss2 = cost_function(Y, K, X)
        else:
            loss, loss1, loss2 = cost_function(Y, K, X, u, weight)

        loss_list.append(loss)
        loss_list1.append(loss1)
        loss_list2.append(loss2)

        # Early stopping
        if best_loss - loss > tolerance:
            best_loss = loss
            no_improvement_count = 0
            K_history = K_history[-1:]
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print("Early stopping at iteration {}".format(i))
            print("Iteration {},  loss: {:.6f} [{:.6f}, {:.6f}]".format(i, loss, loss1, loss2))
            break
    best_case = - no_improvement_count - 1
    return K_history[best_case], loss_list, loss_list1, loss_list2