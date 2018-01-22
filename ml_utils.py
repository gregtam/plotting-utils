import os
from itertools import chain

import numpy as np
import pandas as pd



def _clean_col_name(col_name):
    replace_chars = list(' .-():/')
    
    for char in replace_chars:
        col_name = col_name.replace(char, '_')
    
    return col_name\
        .lower()\
        .rstrip('_')


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
    
    def _extract_split_rule(tree, node, dir, feature_names):
        """Gets the splitting rule for a decision tree at a given node."""
        feat_num = tree.feature[node]
        feat_name = feature_names[feat_num]
        threshold = tree.threshold[node]

        if feat_num < 0:
            return ''

        if dir == 'left':
            return '{} <= {}'.format(feat_name, threshold)
        elif dir == 'right':
            return '{} > {}'.format(feat_name, threshold)
    
    def _recurse_tree(obs, tree, node, left_rules, right_rules, rule_list=[]):
        """Recurses down the tree and extracts the rules."""
        if tree.children_left[node] < 0 and tree.children_right[node] < 0:
            return ' AND '.join(rule_list)

        feat_num = tree.feature[node]
        if obs[feat_num] <= tree.threshold[node]:
            rule_list.append(left_rules[node])
            return _recurse_tree(obs, tree, tree.children_left[node], 
                                 left_rules, right_rules, rule_list)
        else:
            rule_list.append(right_rules[node])
            return _recurse_tree(obs, tree, tree.children_right[node],
                                 left_rules, right_rules, rule_list)
        
    left_rules = [_extract_split_rule(tree, i, 'left', feature_names)
                      for i in xrange(len(tree.feature))]
    right_rules = [_extract_split_rule(tree, i, 'right', feature_names)
                       for i in xrange(len(tree.feature))]
    
    dt_rule_str = _recurse_tree(obs, tree, 0, left_rules, right_rules)
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
        most_common_values = data.value_counts()[:top_n].index
        dummy_df = pd.get_dummies(data, prefix_sep=prefix_sep)[most_common_values]
    
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
        all_columns = list(chain(*distinct_col_vals.values()))
        # Filter selected columns
        dummy_df = pd.get_dummies(data, prefix_sep=prefix_sep)[all_columns]
        
    if clean_col:
        dummy_df.columns = dummy_df.columns.map(_clean_col_name)
    
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
    
    def check_in_array(val, col_array):
        """Check if the value is in array."""
        if col_array is None:
            return 0
        return int(val in col_array)
    
    def get_distinct_values(data):
        """Gets a list of all distinct values."""
        # Get reason code column (dropping nulls)
        srs = data.dropna()
        # Make a list from the list of lists
        all_vals = list(chain.from_iterable(srs))
        # Get distinct values
        distinct_vals = set(all_vals)
        # Sort in alphabetical order
        return sorted(distinct_vals)

    if not isinstance(data, pd.Series):
        raise ValueError('data must be a Pandas Series.')
    
    distinct_vals = get_distinct_values(data)
    
    dummy_df = pd.DataFrame()

    # Create dummy variables for reason codes
    for val in distinct_vals:
        dummy_df[val] = data.map(lambda arr: check_in_array(val, arr))
 
    if include_prefix:
        dummy_df.columns = dummy_df.columns\
            .map(lambda s: data.name + prefix_sep + s)
    if clean_col:
        dummy_df.columns = dummy_df.columns.map(lambda s: _clean_col_name(s))
 
    return dummy_df


def save_large_df_to_excel(df, file_path, sheet_prefix='page'):
    """Saves a large DataFrame to an excel file. A large DataFrame is
    one that has more than 2**20 rows (the maximum that Microsoft Excel
    can display). This function will split the rows across multiple
    sheets.

    Parameters
    ----------
    df : DataFrame
        The DataFrame we wish to save
    file_path : str
        The name of the output file or the file path
    sheet_prefix : str, default 'page'
        The prefix of the sheet names
    """

    def _split_file_path(file_path):
        """Returns a list representing the file path containing the file."""
        if '/' in file_path:
            # Split file path into a list
            split_path = file_path.split('/')
            # Take all except last entry and append together
            directories = split_path
            file_dir = '/'.join(directories[:-1])
            file_name = directories[-1]

            return file_dir, file_name
        else:
            # Otherwise, return current directory as '.'
            return '.', file_name

    file_dir, file_name = _split_file_path(file_path)
    # Prevents overwriting if file already exists
    if file_name in os.listdir(file_dir):
        raise ValueError('Filename {} already exists'.format(file_name))
    writer = pd.ExcelWriter(file_path, engine='xlsxwriter')

    i = 0
    while True:
        # Remove one row to leave space for header
        max_num_rows = 2**20 - 1
        # Subselected DataFrame to write as a single sheet
        temp_df = df.iloc[max_num_rows*i: max_num_rows*(i+1)]

        # Terminate if we have reached the end of the DataFrame
        if temp_df.shape[0] == 0:
            break

        # Save sheet to excel file
        temp_df.to_excel(writer, sheet_name='{}_{}'.format(sheet_prefix, i))
        i += 1

    writer.save()
