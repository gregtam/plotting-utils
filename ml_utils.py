# TODO: Write function to clean df column names, i.e., remove spaces
# and convert to lower case.
from datetime import date, datetime
from itertools import chain
from math import log
import os
import re

import numpy as np
import pandas as pd



def clean_col_name(col_name):
    """Replaces specific characters in a string with an underscore. This
    may be necessary when creating tables in-database as these
    characters might not be allowed in column names.
    """

    replace_chars = list(' .-():/')

    for char in replace_chars:
        col_name = col_name.replace(char, '_')

    return col_name\
        .lower()\
        .rstrip('_')



def create_balanced_train_test_splits(df, class_col, train_class_size,
                                      n_iter=5):
    """Creates multiple iterations of train test splits of a DataFrame,
    where the training sets are balanced.

    Parameters
    ----------
    df : DataFrame
        The data that we want to use for cross validation
    class_col : str
        The column which we want to have even distribution of
    train_class_size : int
        The desired size of each class for training
    n_iter : int, default 5
        The number of times to iterate through different training and
        test splits

    Returns
    -------
    train_test_set_list : list
        A list of tuples of training and test sets
    """

    def _create_balanced_train_df():
        """Creates a balanced training set."""
        class_df_list =\
            [_subset_single_class(value)
                 for value in class_values]

        train_df = pd.concat(class_df_list)
        return train_df

    def _subset_single_class(value):
        """Subsets a single class for training."""
        return df[df[class_col] == value]\
            .sample(train_class_size)

    def _create_complementary_test_df(train_df):
        """Creates a test set that contains the remaining data in df
        that is not in the training set.
        """

        # Indices from the training set
        train_indices = train_df.index
        # Remaining indices
        test_indices = np.setdiff1d(df.index, train_indices)
        # Test set from indices
        test_df = df.loc[test_indices]

        return test_df


    # Get the distinct class values
    class_values = sorted(df[class_col].unique())

    train_test_set_list = []
    for i in range(n_iter):
        train_df = _create_balanced_train_df()
        test_df = _create_complementary_test_df(train_df)

        train_test_set_list.append((train_df, test_df))

    return train_test_set_list


def extract_dt_rule_string(obs, tree, feature_names):
    """This function gets, for a given observation, the set of rules the
    observation follows in a Decision Tree.

    Parameters
    ----------
    obs : list
        A list of the observation's feature values
    tree: An sklearn Tree object
    feature_names: list
        A list of the feature nameis

    Returns
    -------
    dt_rule_str : str
        A string representing the Decision Tree rules.
    """

    def _extract_split_rule(node, direction):
        """Gets the splitting rule for a decision tree at a given node."""
        feat_num = tree.feature[node]
        feat_name = feature_names[feat_num]
        threshold = tree.threshold[node]

        if feat_num < 0:
            return ''

        if direction == 'left':
            return '{} <= {}'.format(feat_name, threshold)
        elif direction == 'right':
            return '{} > {}'.format(feat_name, threshold)

    def _recurse_tree(node, left_rules, right_rules, rule_list=[]):
        """Recurses down the tree and extracts the rules."""
        if tree.children_left[node] < 0 and tree.children_right[node] < 0:
            return ' AND '.join(rule_list)

        feat_num = tree.feature[node]
        if obs[feat_num] <= tree.threshold[node]:
            rule_list.append(left_rules[node])
            return _recurse_tree(tree.children_left[node],
                                 left_rules, right_rules, rule_list)
        else:
            rule_list.append(right_rules[node])
            return _recurse_tree(tree.children_right[node],
                                 left_rules, right_rules, rule_list)


    left_rules = [_extract_split_rule(i, 'left')
                      for i in range(len(tree.feature))]
    right_rules = [_extract_split_rule(i, 'right')
                       for i in range(len(tree.feature))]

    dt_rule_str = _recurse_tree(0, left_rules, right_rules)

    return dt_rule_str


def get_common_dummies(data, top_n=10, prefix_sep='_', clean_col=True):
    """Returns dummy variables, but only for the most common values.

    Parameters
    ----------
    data : Series or DataFrame
    top_n : int, list, or dict, default 10
        A int, list, or dict representing the number of most common
        features to create dummy variables. If it is a int, then it
        will apply that to each feature. If it is a list, then it will
        select different amounts for each feature. It applies it
        element-wise. Alternatively, a dict can be used instead to
        specify the amounts by column.
    prefix_sep : string, default '_'
        Delimiter to use to separate prefix from column value
    clean_col : boolean, default True
        Whether to clean up the final DataFrame column names

    Returns
    -------
    dummy_df : DataFrame
        A DataFrame with the new dummy columns
    """

    if not isinstance(top_n, (int, list, dict)):
        raise ValueError('top_n must be an int, list, or a dict.')
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise ValueError('data must be a Pandas Series or DataFrame.')

    if isinstance(data, pd.Series):
        # Gather top_n most common values
        most_common_values = data.value_counts()[:top_n].index
        # Create dummy variables for all values
        all_dummy_df = pd.get_dummies(data, prefix_sep=prefix_sep)
        # Filter by most common values
        dummy_df = all_dummy_df[most_common_values]

    elif isinstance(data, pd.DataFrame):
        distinct_col_vals = {}

        # Get most common values
        for i, col in enumerate(data):
            val_list = data[col].value_counts().index.tolist()

            if isinstance(top_n, int):
                most_common_values = val_list[:top_n]
            elif isinstance(top_n, list):
                most_common_values = val_list[:top_n[i]]
            elif isinstance(top_n, dict):
                most_common_values = val_list[:top_n[col]]

            most_common_values = [col + prefix_sep + s
                                      for s in most_common_values]
            distinct_col_vals[col] = most_common_values

        # Unnest all columns into a single list
        all_columns = list(chain(*list(distinct_col_vals.values())))
        # Filter selected columns
        dummy_df = pd.get_dummies(data, prefix_sep=prefix_sep)[all_columns]

    if clean_col:
        dummy_df.columns = dummy_df.columns.map(clean_col_name)

    return dummy_df


def get_df_column_type(df, col_name):
    """Returns the type of a DataFrame's columns."""
    srs = df[col_name].dropna()
    single_val = srs.iloc[0]

    return type(single_val).__name__


def get_list_type_dummies(data, prefix_sep='_', clean_col=True,
                          include_prefix=True):
    """Creates dummy variables from a Pandas DataFrame column whose
    types are lists. It will create a dummy variable for each possible
    value in all of the lists and whether the observation has that
    variable.

    Parameters
    ----------
    data : Series
    prefix_sep : str, default '_'
        The string that will separate the column name its value
    clean_col : bool, default True
        Whether or not to clean up the column names
    include_prefix : bool, default True
        Whether or not to include a prefix for the table name

    Returns
    -------
    dummy_df : DataFrame
        A DataFrame with the new dummy columns
    """

    def _get_distinct_values():
        """Gets a list of all distinct values."""
        # Get reason code column (dropping nulls)
        srs = data.dropna()
        # Make a list from the list of lists
        all_vals = list(chain.from_iterable(srs))
        # Get distinct values
        distinct_vals = set(all_vals)
        # Sort in alphabetical order
        return sorted(distinct_vals)

    def _check_in_array(col_array):
        """Check if the value is in array."""
        if col_array is None:
            return 0
        return int(val in col_array)


    if not isinstance(data, pd.Series):
        raise ValueError('data must be a Pandas Series.')

    # Get the series' distinct values
    distinct_vals = _get_distinct_values()

    dummy_df = pd.DataFrame()

    # Create dummy variables for reason codes
    for val in distinct_vals:
        dummy_df[val] = data.map(_check_in_array)

    if include_prefix:
        dummy_df.columns = dummy_df.columns\
            .map(lambda s: data.name + prefix_sep + s)
    if clean_col:
        dummy_df.columns = dummy_df.columns\
            .map(lambda s: clean_col_name(s))

    return dummy_df
