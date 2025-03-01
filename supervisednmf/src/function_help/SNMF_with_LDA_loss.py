import sys
sys.path.append("src/function_help")
from import_library_and_function import *
### SNMF

def encode_X_with_ones(K):
    K_encoded = np.ones((K.shape[0], K.shape[1]))
    return K_encoded

def update_K(Y, K, X):
    mu = 1e-4
    
    numerator = Y.dot(X.T) # (281, 301) @ (301, 50) -> # (281, 50)
    
    denominator = K.dot(X).dot(X.T)  # (281, 50) @ (50, 301) @ (301, 50) -> # (281, 50)
    denominator += mu * K # (281, 50)

#     print("K's numerator:", numerator)
#     print("K's denominator:", denominator)
    denominator += 1e-10
    
    K *= (numerator / denominator) # (281, 50) * [(281, 50) / (281, 50)]
    K = np.abs(K)
    
    return K


def update_weight(u, K, beta, X, Y):
    numerator = (X.dot(Y.T)).dot(u) # (50, 301) @ (301, 281) @ (281, 1) -> # (50, 1)

    denominator = (((X.dot(Y.T)).dot(Y)).dot(X.T)).dot(beta)  # (50, 301) @ (301, 281) @ (281, 301) @ (301, 50) @ (51, 1) -> # (50, 1)
    denominator += 1e-10

#     print("beta's numerator:", numerator)
#     print("beta's denominator:", denominator)
    
    beta *= (numerator/denominator) # (50, 1) * [(50, 1)/(50, 1)]
    beta = np.abs(beta)
    return beta


def update_X(Y, K, X, u, beta):
    gamma = 1
    nu = 1e-7
    lambda_var = 1e-5
    sigma_1, sigma_2 = 0, 0
    
    # Encode K
    # W = encode_X_with_ones(K)

    u = np.expand_dims(u, axis=1)
    beta = np.expand_dims(beta, axis=1)

#     print("Y:", Y.shape)
#     print("K:", K.shape)
#     print("X:", X.shape)
#     print("beta:", beta.shape)
#     print("u:", u.shape)
    
    numerator = (K.T).dot(Y) # (50, 281) @ (281, 301) -> # (50, 301)
    numerator += ((gamma * beta).dot(u.T)).dot(Y) # (50, 1) @ (1, 281) @ (281, 301) -> # (50, 301)

    denominator = ((K.T).dot(K)).dot(X) # (50, 281) @ (281, 50) @ (50, 301) -> # (50, 301)
    denominator += ((((gamma * beta).dot(beta.T)).dot(X)).dot(Y.T)).dot(Y)  # (50, 1) @ (1, 50) @ (50, 301) @ (301, 281) @ (281, 301) -> # (50, 301)
    denominator += nu * X # (50, 301)
    denominator += lambda_var
    denominator += 1e-10

#     print("X's numerator:", numerator)
#     print("X's denominator:", denominator)
    
    X *= (numerator/denominator)
    X = np.abs(X)
    
    return X

def cost_function(Y, K, X, u = np.array([]), beta = np.array([])):
    # SNMF Loss
    gamma = 1

    beta = np.abs(beta)
    
    reconstruction_loss_weight = 1 
    classification_loss_weight = 4
    
    loss1 = (1 / 2) * sum((Y - K.dot(X)).flatten()**2)
    
    # Classification model loss
    if len(u) == 0:
        loss2 = 0
    else:
        loss2 = (gamma/2) * sum((u - (Y.dot(X.T)).dot(beta)).flatten()**2) 
    
    loss = reconstruction_loss_weight * loss1 + classification_loss_weight * loss2

    return loss, loss1, loss2

def SNMF(Y, u, rank, iter, tolerance, patience, epsStab, alpha, epsilon, init_mode):
    ######################## Initialize #######################
    if init_mode in ['nndsvd', 'nndsvda', 'nndsvdar']:
        K, X = init_NMF(Y, rank, init_mode)
    else:
        K, X = init_NMF(Y, rank, 'random')
    
    # beta
    # W = encode_X_with_ones(K)
    W = 0.5 + 0.5 * np.random.standard_normal(size=(K.shape[0], K.shape[1])).astype(K.dtype, copy = False)
    W = np.abs(W)

    # beta = np.zeros(W.shape[1])
    beta = 0.5 + 0.5 * np.random.standard_normal(size=W.shape[1]).astype(Y.dtype, copy = False)
    beta = np.abs(beta)

    # Lost history
    loss_list = []
    loss_list1 = []
    loss_list2 = []

    # X history
    X_history = []

    # beta history
    beta_history = []
    
    # Early stopping
    best_loss = np.inf
    no_improvement_count = 0

    ######################## Loop iteractions #######################
    for i in range(iter):
        
        # Update X
        X = update_X(Y, K, X, u, beta)
        
        # Save X to cache
        X_history.append(X)

        # Update K
        K = update_K(Y, K, X)
        
        # Update beta
        beta = update_weight(u, K, beta, X, Y)

        beta_history.append(beta)
        
        # Loss function
        loss, loss1, loss2 = cost_function(Y, K, X, u, beta)
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
    return X_history[best_case], loss_list, loss_list1, loss_list2, beta_history[best_case]


def SNMF_transform(Y, X, iter, tolerance, patience, epsStab, alpha, epsilon, u = np.array([]), beta = np.array([])):
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
        if len(u) == 0 or len(beta) == 0:
            loss, loss1, loss2 = cost_function(Y, K, X)
        else:
            loss, loss1, loss2 = cost_function(Y, K, X, u, beta)

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