import sys
sys.path.append("src/function_help")
from import_library_and_function import *

def compute_specificity(y_test, y_test_pred):
    y_test = np.array(y_test)
    y_test_pred = np.array(y_test_pred)
    
    if 0 not in y_test:
        return '-'
    else:
        condition = (y_test == 0)

        y_test = y_test[condition]
        y_test_pred = y_test_pred[condition]
        
        result = round(100*accuracy_score(y_test, y_test_pred), 1)
        return result

    
def compute_sensitivity(y_test, y_test_pred):
    y_test = np.array(y_test)
    y_test_pred = np.array(y_test_pred)
    
    if 1 not in y_test:
        return '-'
    else:
        condition = (y_test == 1)

        y_test = y_test[condition]
        y_test_pred = y_test_pred[condition]
        
        result = round(100*accuracy_score(y_test, y_test_pred), 1)
        return result


def compute_f1(y_test, y_test_pred):
    
    y_test = np.array(y_test)
    y_test_pred = np.array(y_test_pred)
    
    # sen
    if 1 not in y_test:
        return '-'
    else:
        condition = (y_test == 1)

        y_test_sen = y_test[condition]
        y_test_pred_sen = y_test_pred[condition]
        
        sen = accuracy_score(y_test_sen, y_test_pred_sen)
    
    # spec
    if 0 not in y_test:
        return '-'
    else:
        condition = (y_test == 0)

        y_test_spec= y_test[condition]
        y_test_pred_spec = y_test_pred[condition]
        
        spec = accuracy_score(y_test_spec, y_test_pred_spec)
        
    try:
        result = round(100 * (2 * spec * sen) / (spec + sen), 1)
        if str(result) == 'nan':
            return '-'
        return result
    except:
        return '-'
    
def compute_auc(y_test, y_test_pred):
    try:
        result = round(100*roc_auc_score(y_test, y_test_pred), 1)
        if str(result) == 'nan':
            return '-'
        return result
    except:
        return '-'
    
def compute_acc(y_test, y_test_pred):
    try:
        result = round(100*accuracy_score(y_test, y_test_pred), 1)
        if str(result) == 'nan':
            return '-'
        return result
    except:
        return '-'


def ranking_metric(y_true, y_pred):
    
    spec = compute_specificity(y_true, y_pred)
    sen = compute_sensitivity(y_true, y_pred)
        
    value = round(((spec + sen) / 2 + spec ) / 2, 2)
    return value

