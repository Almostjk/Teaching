
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_logistic_regressions(X, y):
    crayon = ['#4E79A7','#F28E2C','#E15759','#76B7B2','#59A14F', 
          '#EDC949','#AF7AA1','#FF9DA7','#9C755F','#BAB0AB']
    sns.set_palette(crayon) # set custom color scheme

    labels = list(X.columns)
    
    N, p = X.shape

    rows = int(np.ceil(p/3)) 

    fig, axes = plt.subplots(rows, 3, figsize=(12, rows*(11/4)))

    for i, ax in enumerate(fig.axes):
        if i < p:          
            sns.regplot(X.iloc[:,i], y,  ci=None, logistic=True, y_jitter=0.05, 
                        scatter_kws={'s': 25, 'alpha':.5},  color=crayon[i % 10], ax=ax)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title(labels[i])
            ax.set_xlim(X.iloc[:,i].min(),X.iloc[:,i].max())
        else:
            fig.delaxes(ax)

    sns.despine()
    plt.tight_layout()

    return fig, axes


def plot_feature_importance(model, labels, max_features = 20):
    feature_importance = model.feature_importances_*100
    feature_importance = 100*(feature_importance/np.max(feature_importance))
    table = pd.Series(feature_importance, index = labels).sort_values(ascending=True, inplace=False)
    fig, ax = fig, ax = plt.subplots(figsize=(9,6))
    if len(table) > max_features:
        table.iloc[-max_features:].T.plot(kind='barh', edgecolor='black', width=0.7, linewidth=.8, alpha=0.9, ax=ax)
    else:
        table.T.plot(kind='barh', edgecolor='black', width=0.7, linewidth=.8, alpha=0.9, ax=ax)
    ax.tick_params(axis=u'y', length=0) 
    ax.set_title('Variable importance', fontsize=13)
    sns.despine()
    return fig, ax


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def plot_roc_curves(y_test, y_probs, labels, sample_weight=None):
    
    fig, ax= plt.subplots(figsize=(9,6))

    N, M=  y_probs.shape

    for i in range(M):
        fpr, tpr, _ = roc_curve(y_test, y_probs[:,i], sample_weight=sample_weight)
        auc = roc_auc_score(y_test, y_probs[:,i], sample_weight=sample_weight)
        ax.plot(1-fpr, tpr, label=labels.iloc[i] + ' (AUC = {:.3f})'.format(auc))
    
    ax.plot([0,1],[1,0], linestyle='--', color='black', alpha=0.6)

    ax.set_xlabel('Specificity')
    ax.set_ylabel('Sensitivity')
    ax.set_title('ROC curves', fontsize=14)
    sns.despine()

    plt.legend(fontsize=13, loc ='lower left' )
    
    return fig, ax


from sklearn.calibration import calibration_curve


def plot_calibration_curves(y_true, y_prob, labels=None):
    
    fig, ax = plt.subplots(figsize=(9,6))
    
    if y_prob.ndim==1:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        if labels:
            ax.plot(prob_pred, prob_true, label=labels)
    else: 
        m = y_prob.shape[1]
        for i in range(m):
            prob_true, prob_pred = calibration_curve(y_true, y_prob[:,i], n_bins=10)
            if labels:
                ax.plot(prob_pred, prob_true, label=labels[i])
            else:
                ax.plot(prob_pred, prob_true)
    
    ax.plot([0,1],[0,1], linestyle='--', color='black', alpha=0.5)

    ax.set_xlabel('Estimated probability')
    ax.set_ylabel('Empirical probability')
    if y_prob.ndim==1:
        ax.set_title('Reliability curve', fontsize=14)
    else:
        ax.set_title('Reliability curves', fontsize=14)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    plt.legend(fontsize=13, frameon=False)
    sns.despine()
  
    return fig, ax


