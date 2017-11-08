from datetime import date
from itertools import izip

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor,\
                         ExtraTreeClassifier, ExtraTreeRegressor
from statsmodels.stats.proportion import proportion_confint


def plot_compare_feat_population(data_df, x_col, y_col, normalize=False,
                                 **kwargs):
    """Plots overlaid histograms of a given feature for different
    populations. Typically used to compare two populations based on the
    dependent variable.
    
    Parameters
    ----------
    data_df : DataFrame
        DataFrame that has the x and y variables
    x_col : str
        The feature we wish to compare
    y_col : str
        The column which determines the populations
    normalize : bool, default False
        Whether to normalize the populations
    kwargs : Matplotlib kwargs
    """
    
    def _pivot_column(df, x_col, y_col, normalize):
        group_by_srs = df[[y_col, x_col]]\
            .fillna('none')\
            .groupby([y_col, x_col])\
            .size()

        group_by_df = pd.DataFrame(group_by_srs, columns=['counts'])
        pivot_df = group_by_df\
            .reset_index()\
            .pivot(index=x_col,
                   columns=y_col,
                   values='counts'
                  )
            
        # Normalize counts
        if normalize:
            for col_name in pivot_df:
                col_sum = pivot_df[col_name].sum()
                pivot_df[col_name] = pivot_df[col_name]/col_sum
            
        pivot_df.fillna(0, inplace=True)
        
        return pivot_df

    pivot_df = _pivot_column(data_df, x_col, y_col, normalize)
    pivot_df.plot(kind='bar', **kwargs)

    if normalize:
        y_label = 'Relative Frequency'
    else:
        y_label = 'Frequency'

    if 'ax' in kwargs:
        ax = kwargs['ax']
        ax.set_ylabel(y_label)
    else:
        plt.ylabel(y_label)
    plt.tight_layout()
    
    return pivot_df


def plot_feature_importances(clf, feat_names, top_n=None, **kwargs):
    """Plots the top feature importances.
    
    Parameters
    ----------
    clf : A DecisionTreeClassifier, DecisionTreeRegressor,
          RandomForestClassifier, or RandomForestRegressor object
    feat_names : list
        A list of the feature names
    top_n : int, default None
        The number of top features to plot. If None, plot all features
    kwargs : Matplotlib keyword arguments
    
    Returns a DataFrame of the feature importances.
    """

    clf_tuple = (DecisionTreeClassifier, DecisionTreeRegressor,
                 RandomForestClassifier, RandomForestRegressor)
    if not isinstance(clf, clf_tuple):
        raise TypeError('clf should be one of (RandomForestClassifier, RandomForestRegressor, RandomForestClassifier, RandomForestRegressor)')
        
    feat_imp_df = pd.DataFrame()
    feat_imp_df['feat_name'] = feat_names
    feat_imp_df['feat_importance'] = clf.feature_importances_
    
    feat_imp_df = feat_imp_df\
        .set_index('feat_name')\
        .sort_values('feat_importance')
        
    if top_n is not None:
        plot_df = feat_imp_df.tail(top_n)
    else:
        plot_df = feat_imp_df
        
    plot_df.plot(kind='barh', legend=None)
    
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Name')
    
    return feat_imp_df.iloc[::-1]
    

def plot_proportion_w_confint(data_df, x_col, y_col,
                              top_n=10, max_ci_len=1.0, show_n_obs=None,
                              **kwargs):
    """Plots the proportion of a binary variable grouped by a given
    feature.
    
    Parameters
    ----------
    data_df : DataFrame
        DataFrame that has the x and y variables
    x_col - The name of the x variable
    y_col - The name of the y variable
    top_n - The number of top features by proportion to plot
                 (Default: 10)
    max_ci_len - The maximum ci length (Default: 1.0)
    show_n_obs - Whether to show the number of observations in the plot.
                 If show_n_obs equals 'in_plot', then show on the graph.
                 If show_n_obs equals 'in_axis', then append it to the
                 index.
                 (Default: None)
    kwargs - Matplotlib kwargs
    """
    
    def _add_confint_columns(df):
        """Adds the confidence interval columns to a DataFrame."""
        from statsmodels.stats.proportion import proportion_confint
        
        # Get upper and lower bounds for confidence intervals
        confint_list =\
            [proportion_confint(cnt, n_obs, method='wilson')
                 for cnt, n_obs in izip(df.cnt, df.n_obs)]
    
        # Transpose lists so we can insert them into the DataFrame
        confint_list = zip(*confint_list)
        # Lower bound of confidence interval
        df['ci_lower'] = confint_list[0]
        # Upper bound of confidence interval
        df['ci_upper'] = confint_list[1]
        # Width of confidence interval
        df['ci_length'] = df.ci_upper - df.ci_lower
        # Amount of error to the left of the mean
        df['error_left'] = df.prop - df.ci_lower
        # Amount of error to the right of the mean
        df['error_right'] = df.ci_upper - df.prop
        
        return df
    
    def _create_plot(df, **kwargs):
        # Plot bars
        grouped_df\
            .prop\
            .plot(kind='barh', label='Proportion', **kwargs)

        if 'ax' in kwargs:
            ax = kwargs['ax']
            # Plot error bars
            ax.errorbar(grouped_df.prop, np.arange(len(grouped_df)),
                        xerr=[grouped_df.error_left, grouped_df.error_right],
                        fmt='o', c='black', label='Confidence Interval')
            
            ax.set_xlabel('Proportion')
            ax.set_xlim(0, 1)

            ax.legend(loc=0)
        else:
            # Plot error bars
            plt.errorbar(grouped_df.prop, np.arange(len(grouped_df)),
                         xerr=[grouped_df.error_left, grouped_df.error_right],
                         fmt='o', c='black', label='Confidence Interval')
            
            plt.xlabel('Proportion')
            plt.xlim(0, 1)

            plt.legend(loc=0)

        plt.tight_layout()

    def _plot_n_obs(grouped_df, show_n_obs):
        if show_n_obs == 'in_plot':
            for index, n_obs in enumerate(grouped_df.n_obs):
                n_obs_txt = 'n_obs = {:,}'.format(n_obs)
                plt.text(0.01, index, n_obs_txt,
                         color='white', size=12, verticalalignment='center')
        elif show_n_obs == 'in_axis':
            # Include number of observations in index
            grouped_df.index = ['{} (n_obs = {:,})'.format(sr, n_obs)
                                    for sr, n_obs in zip(grouped_df.index,
                                                         grouped_df.n_obs)]
        elif show_n_obs is not None:
            raise ValueError("show_n_obs should be either 'in_plot' or 'in_axis'")


    
    grouped_df = data_df[[y_col, x_col]]\
        .fillna('none')\
        .groupby(x_col)\
        .agg([np.sum, np.size, np.mean])
        
    grouped_df.columns = ['cnt', 'n_obs', 'prop']
    
    grouped_df = _add_confint_columns(grouped_df)
    
    # Sort values, filter by interval length, and take the top features
    grouped_df = grouped_df\
        .sort_values('prop')\
        .query('ci_length < @max_ci_len')\
        .tail(top_n)
        
    _plot_n_obs(grouped_df, show_n_obs)

    _create_plot(grouped_df, **kwargs)

    
    return grouped_df


def plot_regression_coefficients(clf, feat_names, top_n=None, **kwargs):
    """Plots the most extreme regression coefficients.
    
    feat_names - A list of the feature names
    top_n - The number of most extreme features to plot. If None, plot 
            all features (Default: None)
    kwargs - Matplotlib keyword arguments
    
    Returns a DataFrame of the regression coefficients.
    """
    
    clf_tuple = (ElasticNet, LinearRegression, LogisticRegression)
    if not isinstance(clf, clf_tuple):
        raise TypeError('clf should be one of (ElasticNet, LinearRegression, LogisticRegression)')
        
    def _create_coef_df(clf, feat_names):
        reg_coef_df = pd.DataFrame()
        reg_coef_df['feat_name'] = feat_names
        if isinstance(clf, LogisticRegression):
            reg_coef_df['coef'] = clf.coef_[0]
        else:
            reg_coef_df['coef'] = clf.coef_

        return reg_coef_df\
            .set_index('feat_name')\
            .sort_values('coef')
            
    reg_coef_df = _create_coef_df(clf, feat_names)
        
    if top_n is not None:
        # Most negative coefficients
        reg_coef_df_head = reg_coef_df.head(top_n)
        # Most positive coefficients
        reg_coef_df_tail = reg_coef_df.tail(top_n)
    else:
        # Most negative coefficients
        reg_coef_df_head = reg_coef_df
        # Most positive coefficients
        reg_coef_df_tail = reg_coef_df
    
    # Get yticks
    plot_feat_names = reg_coef_df_head.index.tolist()\
                      + [' ']\
                      + reg_coef_df_tail.index.tolist()
    
    # Plot bar chart
    plt.barh(np.arange(1, top_n+1), reg_coef_df_tail.coef, color=green)
    plt.plot([0, 0], [-top_n - 0.5, top_n + 0.5], '--', color='black')
    plt.barh(np.arange(-top_n, 0), reg_coef_df_head.coef, color=red)
    
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature Name')
    plt.yticks(np.arange(-top_n, top_n + 1), plot_feat_names)
    
    return reg_coef_df.iloc[::-1]


def plot_roc(y_test, y_score, ax=None, title=None, **kwargs):
    """Plots the ROC curve for a binary classifier.
    
    Parameters
    ----------
    y_test : list, Series
        The true values of the observations.
    y_score : list, Series
        The corresponding scores.
    ax : Matplotlib axes object, default None
    title : str, default None
        Plotting title. It will add AUC after this.
    kwargs : Matplotlib keyword arguments
    """
    auc_val = roc_auc_score(y_test, y_score)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    
    def _get_plot_title(title):
        plot_title = 'AUC: {:.3f}'.format(auc_val)
        if title is not None:
            plot_title = '{}\n{}'.format(title, plot_title)
        return plot_title
    
    plot_title = _get_plot_title(title)
        
    if ax is None:
        plt.plot(fpr, tpr, **kwargs)
        plt.title(plot_title)
        plt.xlabel('FPR')
        plt.ylabel('TPR')

        plt.xlim(0, 1)
        plt.ylim(0, 1)
    else:
        ax.plot(fpr, tpr, **kwargs)
        ax.set_title(plot_title)
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    
def save_fig(filename, directory='plots'):
    """Saves a Matplotlib figure with the date prepended to the
    filename.

    Parameters
    ----------
    filename : str
        The name of the file
    directory : str, default 'plots'
        Where the image will be saved
    """

    month = date.today().month
    day = date.today().day

    if filename.endswith('.png'):
        save_name = '{}/{:02d}{:02d}_{}'\
            .format(directory, month, day, filename)
    elif '.' in filename:
        save_name = '{}/{:02d}{:02d}_{}.png'\
            .format(directory, month, day, filename.split('.')[0])
    else:
        save_name = '{}/{:02d}{:02d}_{}.png'\
            .format(directory, month, day, filename)
    plt.savefig(save_name)
