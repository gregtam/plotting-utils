from datetime import date, datetime
from math import log
import os
import re

import numpy as np
import pandas as pd



def colour_df_by_group(df, group_col, colours=None, file_path=None):
    """Colours the rows of a DataFrame, with differing colours for
    alternating groups. This makes it easier to visually separate
    different groups when viewing a DataFrame. The groups are defined by
    the same values of a given column.

    Parameters
    ----------
    df : DataFrame
        The DataFrame we wish to colour
    group_col : str or list
        The name of the column that defines the groups
    colours: list or list of lists, default None
        A list of the colours of the groups. Should have CSS colour
        formatting. If None, then defaults to the #FFFFFF and #AAAAAA
        CSS colours.

        E.g., 'background-color: #FF0000' or 'background-color: red'
    file_path : str, default None
        The name of the file path to save the coloured DataFrame. If it
        is None, then do not save.
    """
    # TODO: Fix default colouring

    def _check_for_input_errors(group_col, colours, file_path):
        """Check parameters for errors."""
        array_like_types = (list, tuple, np.ndarray)
        if colours is not None:
            if not isinstance(colours, array_like_types):
                raise ValueError('colours should be None or array-like.')

            if colours is not None and log(len(colours), 2) != len(group_col):
                raise ValueError('colours should have length of 2 to the '
                                 'power of however many values of group_col '
                                 'there are.')

        if file_path and not _is_valid_colour_format(colours):
            raise ValueError('If saving to a file, then colour names must be '
                             'in the format #rgb or #rrggbb.')

    def _is_valid_colour_format(colour):
        """Checks whether the colour is the proper format. It should
        be in the form 'background-color: #rrggbb', '#rrggbb', or a
        3-tuple with red, green, and blue values in [0, 1].
        """

        pattern = '^(background-color: )?#[0-9a-fA-F]{6}'
        if isinstance(colour, str) and not bool(re.match(pattern, colour)):
            raise ValueError("colour should be of the form '#FFFFFF' or "
                             "'background-color: #FFFFFF.'")

        if isinstance(colour, tuple):
            for val in tuple:
                if val < 0 or val > 1:
                    raise ValueError('colour tuple values should be between '
                                     '0 and 1.')

    def _listify(vals):
        """Converts to a list if not already a list."""
        if not isinstance(vals, list):
            return [vals]
        return vals

    def _convert_colour_to_hex_css(colour):
        """Convert the colour to hexadecimal (for CSS) if not already."""
        if isinstance(colour, tuple):
            # RGB channels in hexadecimal
            hex_colour_list = [_convert_decimal_to_hex(dec) for dec in colour]
            hex_value = ''.join(hex_colour_list)

            return 'background-color: #{}'.format(hex_value)

        elif re.match('#[0-9a-fA-F]{6}', colour):
            return 'background-color: {}'.format(colour)

        else:
            return colour

    def _convert_decimal_to_hex(rgb_float):
        """Converts a single decimal number to hexadecimal."""
        # Converts from [0, 1] to [0, 255]
        rgb_num = int(255 * rgb_float)

        # Converts to hex and trims off the '0x' from the beginning
        hex_string = hex(rgb_num)[2:]

        if len(hex_string) == 1:
            # Pad with a leading 0
            return '0{}'.format(hex_string)
        else:
            return hex_string

    def _reshape_colours(colours):
        """Reshapes the colours list into a numpy array so it can be
        indexed by tuples.
        """

        # Number of dimensions
        num_dim = log(len(colours), 2)
        # New shape should be of the form (2, 2, ...)
        new_shape = (2,) * int(num_dim)

        return np.array(colours).reshape(*new_shape)

    def _get_group_colours(df, group_col, colours):
        """Returns a DataFrame detailing the colours of each row."""

        def _retrieve_colour(indices):
            """Retrieves the colour based off the indices."""
            if len(indices) == 1:
                return colours[int(indices)]
            else:
                return colours[tuple(indices)]

        # Maps the unique values of group_col to colour groups
        index_df = _create_index_df(df, group_col)

        # Get the cell colour from the index
        colour_group_srs = index_df.apply(_retrieve_colour, axis=1)

        # Create an empty DataFrame with same indices and columns as df
        style_df = pd.DataFrame(columns=df.columns, index=df.index)

        # Fill in the rows of each column with the colour group values
        for col in style_df:
            style_df[col] = colour_group_srs
        return style_df

    def _create_index_df(df, group_col):
        """Create a DataFrame containing the indexes for colouring."""
        # Create empty DataFrame
        index_df = pd.DataFrame()

        # Create alternating indices for each column
        for col in group_col:
            new_col_name = '{}_index'.format(col)
            index_df[new_col_name] = _create_alternating_index_cols(df[col])

        return index_df

    def _create_alternating_index_cols(srs):
        """Creates a Series of 0s and 1s that indicate groups of rows
        with the same value.
        """

        index_dict = dict((v, i % 2) for i, v in enumerate(srs.unique()))
        return srs.map(index_dict)

    def _value_to_colour_group_map(df, colours):
        """Maps the value of a column to its corresponding colour group."""
        # Unique values of the column we wish to group sorted in order
        # of appearance
        unique_vals = df.drop_duplicates()

        # A mapping between the value and its colour group. Groups will
        # alternate between 0 and 1 for various values.
        return dict([(val, colours[i%2]) for i, val in enumerate(unique_vals)])


    # Convert to numpy array if not already
    group_col = _listify(group_col)
    if colours is not None:
        colours = _listify(colours)

    _check_for_input_errors(group_col, colours, file_path)

    # Set default colours
    if colours is None:
        colours = ['background-color: #FFFFFF', 'background-color: #AAAAAA']
    else:
        # Converts to a single list of CSS hexadecimal colour strings
        colours = [_convert_colour_to_hex_css(colour) for colour in colours]
        # Reshape colours so it can be indexed by tuples
        colours = _reshape_colours(colours)

    # Returns the styled DataFrame
    styled_df = df.style\
        .apply(lambda df: _get_group_colours(df, group_col, colours),
               axis=None)

    if file_path is not None:
        styled_df.to_excel(file_path, index=False, engine='openpyxl')

    return styled_df


def date_file_path(file_path, directory='', date_format='%m%d'):
    """Prepend the date to a file path/file name.

    Parameters
    ----------
    file_path : str
        The file path or file name. If it is a file path, then directory
        must be empty.
    directory : str, default ''
        The directory of the file. If not empty, then file_path must be
        just a file name.
    date_format : str, default '%m%d'
        The desired format of the date

    Returns
    -------
    dated_file_path : str
    """

    def _check_for_input_errors(file_path, directory):
        """Check parameters for errors."""
        if not isinstance(file_path, str):
            raise ValueError('file_path must be a string.')
        if not isinstance(directory, str):
            raise ValueError('directory must be a string.')
        if '/' in file_path and len(directory) > 0:
            raise ValueError('Cannot specify a folder in file_path and also '
                             'specify the directory parameter.')

    def _prepend_date(file_name):
        """Prepends the date to the file name."""
        date_prefix = _get_date_prefix(date_format)
        return '{}_{}'.format(date_prefix, file_name)

    def _get_date_prefix(date_format):
        """Gets the date prefix string from today's date."""
        today_date_str = datetime.strftime(date.today(), date_format)
        return today_date_str


    _check_for_input_errors(file_path, directory)

    if '/' not in file_path and directory == '':
        dated_file_path = _prepend_date(file_path)

    elif '/' in file_path and directory == '':
        # Split folders and file name into a list
        folders_and_file_name = file_path.split('/')
        # Prepend the date just to the file name
        folders_and_file_name[-1] = _prepend_date(folders_and_file_name[-1])
        # Join together again to create the dated file path
        dated_file_path = '/'.join(folders_and_file_name)

    else:
        dated_file_name = _prepend_date(file_path)
        # Prepend directory to dated_file_name
        dated_file_path = '{}/{}'.format(directory, dated_file_name)

    return dated_file_path


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
