from sklearn.metrics import roc_auc_score

def calculate_gini_score(a,b):
    """Function that received two parameters; first: a binary variable representing 0=good and 1=bad, and then a second variable with the prediction of the first variable, the second variable can be continuous, integer or binary - continuous is better. Finally, the function returns the GINI Coefficient of the two lists."""    
    gini = 2*roc_auc_score(a,b)-1
    return gini