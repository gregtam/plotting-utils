from functools import reduce
from textwrap import dedent
from warnings import warn

from impala.sqlalchemy import BIGINT, BOOLEAN, DECIMAL, DOUBLE, FLOAT, INT,\
                              SMALLINT, STRING, TIMESTAMP, TINYINT
import numpy as np
import pandas as pd
import pandas.io.sql as psql
import psycopg2
from sqlalchemy import Column, Table, MetaData
from sqlalchemy import all_, and_, any_, not_, or_
from sqlalchemy import alias, between, case, cast, column, distinct, extract,\
                       false, func, intersect, literal, literal_column,\
                       select, text, true, union, union_all
from sqlalchemy import CHAR, REAL, VARCHAR
from sqlalchemy.sql.selectable import Alias, Select



def _drop_table(table_name, schema, engine, print_query=False):
    """Drops a SQL table."""
    if schema is None:
        drop_str = 'DROP TABLE IF EXISTS {};'.format(table_name)
    else:
        drop_str = 'DROP TABLE IF EXISTS {}.{};'.format(schema, table_name)

    if print_query:
        print(drop_str)

    psql.execute(drop_str, engine)


def _from_df_type_to_sql_type(type_val):
    """Converts a DataFrame data type to a SQL type."""
    type_str = type_val.name

    if type_str == 'object':
        return 'STRING'
    elif 'int' in type_str:
        return 'INTEGER'
    elif 'float' in type_str:
        return 'FLOAT'
    elif 'datetime' in type_str:
        return 'TIMESTAMP'


def _get_create_col_list(data, partitioned_by):
    """Returns a list of the column names and their data types to
    include in a CREATE TABLE query excluding ones for partitioning.
    """

    if isinstance(partitioned_by, str):
        partitioned_by = [partitioned_by]

    if isinstance(data, (Alias, Table)):
        create_col_list = [_get_single_partitioned_str(data, col.name)
                               for col in data.c
                                   if col.name not in partitioned_by]
    elif isinstance(data, pd.DataFrame):
        create_col_list = ['{} {}'.format(k, _from_df_type_to_sql_type(v))
                               for k, v in data.dtypes.items()
                                   if k not in partitioned_by]
    return create_col_list


def _get_partition_col_list(data, partitioned_by):
    """Returns a list of the column names and their data types to
    include in the PARTITIONED BY clause of a CREATE TABLE query.
    """

    if isinstance(partitioned_by, str):
        partitioned_by = [partitioned_by]

    if isinstance(data, (Alias, Table)):
        partition_col_list = [_get_single_partitioned_str(data, col_str)
                                  for col_str in partitioned_by]
    elif isinstance(data, pd.DataFrame):
        partition_col_list = ['{} {}'.format(k, _from_df_type_to_sql_type(v))
                                  for k, v in data.dtypes.items()
                                      if k in partitioned_by]
    return partition_col_list


def _get_single_partitioned_str(data, col_str):
    """Returns a string of the column name and type for partitioning."""
    col = data.c[col_str]
    return '{} {}'.format(col.name, col.type.__visit_name__)


def _separate_schema_table(full_table_name, con):
    """Separates schema name and table name"""
    if '.' in full_table_name:
        return full_table_name.split('.')
    else:
        schema_name = psql.read_sql('SELECT current_schema();', con).iloc[0, 0]
        table_name = full_table_name
        return schema_name, full_table_name



def assign_column_types(data, col_type_dict):
    """Assign column types to a SQLAlchemy Alias. This is necessary if
    we want to run save_tables(), since creating a blank table requires
    data types for each column.

    Parameters
    ----------
    data : SQLAlchemy selectable
        The selectable object that we wish to assign column types
    col_type_dict : dict
        Maps the column name to its desired data type
    """

    for col_name, data_type in col_type_dict.items():
        data.c[col_name].type = data_type


def balance_classes(data, class_col, class_sizes=1000, class_values=None):
    """Balance the classes of a data set so they are evenly distributed.

    Parameters
    ----------
    data : SQLAlchemy Alias/Table
    class_col : str
        The column name that we want to have even distribution of
    class_sizes : int or dict, default 1000
        The desired number of rows of each class in our final data set.
            - If int, then all classes will be the same size
            - If dict, then it should map the class value to its size.
              Therefore, class_values should not be specified.
    class_values : list, default None
        The values that the class_col can take. If None, then it will
        find it automatically. If not None, then class_sizes should not
        be a dict, then that will already determine the classes.
    """

    def _get_class_values():
        """Gets the distinct class values from the database."""
        class_values_tpl_list =\
            select([distinct(column(class_col))],
                   from_obj=data
                  )\
            .execute()\
            .fetchall()

        # Converts from a list of tuples as a list of values
        class_values = [tpl[0] for tpl in class_values_tpl_list]

        return class_values

    def _subset_all_classes():
        """Returns a list of subsetted Aliases for each class."""
        if isinstance(class_sizes, int):
            single_class_subset_aliases =\
                [_subset_single_class(class_val, class_sizes)
                     for class_val in class_values]
        elif isinstance(class_sizes, dict):
            single_class_subset_aliases =\
                [_subset_single_class(class_val, class_size)
                     for class_val, class_size in class_sizes.items()]

        return single_class_subset_aliases

    def _subset_single_class(class_val, class_size):
        """Subsets the data by a single class."""
        class_subset_alias =\
            select(data.c)\
            .where(column(class_col) == class_val)\
            .limit(class_size)\
            .alias('class_subset')

        # Nests it in another select, since there is a glitch which
        # prevents us from selecting from a union if a limit and/or
        # order by is specified. (However, we can write it to a
        # DataFrame).
        return select(class_subset_alias.c)


    if isinstance(class_sizes, dict) and class_values is not None:
        raise ValueError('If class_sizes is a dict, then class_values must be '
                         'None.')

    # If class_values is a dict, then we can infer classes. Only
    # retrieve the class values from the database if we can't infer.
    if class_values is None and not isinstance(class_sizes, dict):
        class_values = _get_class_values()

    single_class_subset_aliases = _subset_all_classes()

    balanced_class_union_alias =\
        union_all(*single_class_subset_aliases)\
        .alias('balanced_class_union')

    return balanced_class_union_alias


def clear_schema(schema_name, con, print_query=False):
    """Remove all tables in a given schema.

    Parameters
    ----------
    schema_name : str
        Name of the schema in SQL
    con : SQLAlchemy engine object or psycopg2 connection object
    print_query : bool, default False
        If True, print the resulting query
    """

    sql = f'SHOW TABLES IN {schema_name};'

    if print_query:
        print(dedent(sql))

    table_names = psql.read_sql(sql, con).table_name

    for table_name in table_names:
        del_sql = 'DROP TABLE IF EXISTS {schema_name}.{table_name};'\
            .format(**locals())
        psql.execute(del_sql, con)


def compute_percent_missing(full_table_name, con, print_query=False):
    """This function takes a schema name and table name as an input and
    creates a SQL query to compute the number of missing entries for
    each column. It will also determine the total number of rows in the
    table.

    Parameters
    ----------
    full_table_name : str
        Name of the table in SQL. Input can also include have the schema
        name prepended, with a '.', e.g., 'schema_name.table_name'.
    con : SQLAlchemy engine object or psycopg2 connection object
    print_query : bool, default False
        If True, print the resulting query

    Returns
    -------
    pct_df : DataFrame
    """

    column_names = get_column_names(full_table_name, con).column_name
    schema_name, table_name = _separate_schema_table(full_table_name, con)

    num_missing_sql_list = ['SUM(({name} IS NULL)::INTEGER) AS {name}'\
                                .format(name=name) for name in column_names]

    num_missing_list_str = ',\n           '.join(num_missing_sql_list)

    sql = '''
    SELECT {num_missing_list_str},
           COUNT(*) AS total_count
      FROM {schema_name}.{table_name};
    '''.format(**locals())

    # Read in the data from the query and transpose it
    pct_df = psql.read_sql(sql, con).T

    # Rename the column to 'pct_null'
    pct_df.columns = ['pct_null']

    # Get the number of rows of table_name
    total_count = pct_df.ix['total_count', 'pct_null']

    # Remove the total_count from the DataFrame
    pct_df = pct_df[:-1]/total_count
    pct_df.reset_index(inplace=True)
    pct_df.columns = ['column_name', 'pct_null']
    pct_df['table_name'] = table_name

    if print_query:
        print(dedent(sql))

    return pct_df


def convert_table_to_df(data):
    """Converts a SQLAlchemy Alias, Select, or Table to a pandas
    DataFrame. This function will use fetchall(), then convert that
    result to a DataFrame. That way, we do not have to use a psql and
    an extra, unneeded engine object.

    Note that because Alias and Table objects cannot be executed, they
    will not retain the same ordering as Select objects.

    Parameters
    ----------
    data : SQLAlchemy Alias/Table
        The object we will convert to a DataFrame.

    Returns
    -------
    df : DataFrame
         A DataFrame representation of the data.
    """

    def _get_slct_object():
        """Returns a Selectable object."""
        if isinstance(data, (Alias, Table)):
            # Alias and Table cannot be executed, so we must select it
            return select(data.c)
        elif isinstance(data, Select):
            # An Alias can be executed, so just return itself back. Note
            # that if we selected the Alias, it would lose any of the
            # ordering that mmight be specified
            return data

    def _get_column_names():
        """Returns a list of the table's column names."""
        return [s.name for s in data.c]


    # Set the select object
    slct = _get_slct_object()

    # Fetch all rows (as a list of tuples, where each tuple value
    # represents the columns)
    tpl_list = slct.execute().fetchall()
    col_names = _get_column_names()
    df = pd.DataFrame(tpl_list, columns=col_names)

    return df


def count_distinct_values(data, approx=False):
    """Counts the number of distinct values for each column of a table.

    Parameters
    ----------
    data : SQLAlchemy Table/Alias
        The object representing the table or the table name
    approx : bool, default False
        Whether to approximate (uses ndv() function)

    Returns
    -------
    count_distinct_df : DataFrame
    """

    def _check_for_input_errors():
        """Check parameters for errors."""
        if not isinstance(data, (Alias, Table)):
            raise TypeError('data must be of Alias or Table type.')

    def _fetch_num_distinct_values(col):
        """Fetches the number of distinct values for a given column."""
        if approx:
            distinct_count_col = func.ndv(col)
        else:
            distinct_count_col = func.count(col.distinct())

        distinct_count =\
            select([distinct_count_col],
                   from_obj=data
                  )\
            .execute()\
            .scalar()

        return distinct_count


    _check_for_input_errors()

    count_distinct_df = pd.DataFrame(columns=['column_name', 'n_distinct'])
    for tbl_col in data.c:
        # Fetch the number of distinct values for the column
        distinct_count = _fetch_num_distinct_values(tbl_col)

        # Create new row to add to DataFrame
        new_row = (tbl_col.name, distinct_count)

        # Add new row to DataFrame
        count_distinct_df.loc[count_distinct_df.shape[0]] = new_row

    return count_distinct_df


def count_distincts_by_group(data, group_by_cols, count_distinct_cols):
    """Groups by column(s) and counts the distincts of multiple columns.
    Since Impala cannot handle computing multiple count distincts, we
    must do them separately, then join the results together.

    Parameters
    ----------
    data : SQLAlchemy Alias/Table
        The data which we want to compute our group by on
    group_by_cols : str or list of str
        The column(s) which we want to group by
    count_distinct_cols : str or list of str
        The column(s) which we want to do count the distincts on

    Returns
    -------
    grouped_count_distinct_alias : SQLAlchemy Alias
    """

    def _convert_to_col_list(cols):
        """Converts to a list of columns."""
        if isinstance(cols, str):
            return [column(cols)]
        elif isinstance(cols, list):
            return [column(s) for s in cols]

    def _compute_single_distinct(count_col):
        """Counts distincts for a single columns."""
        group_by_slct =\
            select(group_by_col_list
                   + [func.count(distinct(count_col))
                          .label('n_distinct_{}'.format(count_col.name))
                     ],
                   from_obj=data
                  )\
            .group_by(*group_by_col_list)

        return group_by_slct

    def _outer_join(data_1, data_2):
        """Outer joins to grouped by objects. To be used by reduce()."""

        # Specifies the join clause
        join_cond_list = [data_1.c[col.name] == data_2.c[col.name]
                              for col in group_by_col_list]
        join_clause = and_(*join_cond_list)

        # Remaining count columns
        data_1_col_list =\
            [col for col in data_1.c if col.name not in group_by_cols]
        data_2_col_list =\
            [col for col in data_2.c if col.name not in group_by_cols]

        # Set the join
        data_full_join = data_1\
            .join(data_2,
                  full=True,
                  onclause=join_clause
                 )

        # Forms the coalesced key, so that all entries of the key are filled
        coalesce_keys =\
            [func.coalesce(data_1.c[col.name], data_2.c[col.name])
                     .label(col.name)
                 for col in group_by_col_list]

        # Final Alias to return, so that it can be used in the next iteration
        outer_join_alias =\
            select(coalesce_keys
                   + data_1_col_list
                   + data_2_col_list,
                   from_obj=data_full_join
                  )\
            .alias('outer_join')

        return outer_join_alias


    # Convert str list to column list
    group_by_col_list = _convert_to_col_list(group_by_cols)
    count_distinct_col_list = _convert_to_col_list(count_distinct_cols)

    # Grouped by Aliases
    grouped_slct_list =\
        [_compute_single_distinct(count_col)
             for count_col in count_distinct_col_list]
    # Assign different Alias names to each
    grouped_alias_list = [data.alias('foo_{}'.format(i))
                              for i, data in enumerate(grouped_slct_list)]

    # Join together all distinct Alias results
    grouped_count_distinct_alias = reduce(_outer_join, grouped_alias_list)

    return grouped_count_distinct_alias


def count_rows(data, print_commas=False):
    """Counts the number of rows from a SQLAlchemy Alias or Table.

    Parameters
    ----------
    data : SQLAlchemy Alias/Table
    print_commas : bool, default False
        Whether or not to print commas for the thousands separator

    Returns
    -------
    row_count : int
    """

    row_count =\
        select([func.count('*')],
               from_obj=data
              )\
        .execute()\
        .scalar()

    if print_commas:
        print('{:,}'.format(row_count))

    return row_count


def fetch_column_names(full_table_name, con, order_by='ordinal_position',
                     reverse=False, print_query=False):
    """Fetches all of the column names of a specific table.

    Parameters
    ----------
    full_table_name : str
        Name of the table in SQL. Input can also include have the schema
        name prepended, with a '.', e.g., 'schema_name.table_name'.
    con : SQLAlchemy engine object or psycopg2 connection object
    order_by : str, default 'ordinal_position'
        Specified way to order columns. Can be either 'ordinal_position'
        or 'alphabetically'.
    reverse : bool, default False
        If True, then reverse the ordering
    print_query : bool, default False
        If True, print the resulting query

    Returns
    -------
    column_names_df : DataFrame
    """

    def _reorder_df(df, order_by):
        """Reorders the DataFrame."""
        if order_by not in ('ordinal_position', 'alphabetically'):
            raise ValueError("order_by must be either 'ordinal_position' or"
                             "'alphabetically'.")
        if order_by == 'alphabetically':
            df = df\
                .sort_values('name')\
                .reset_index(drop=True)
        return df

    def _reverse_df(df, reverse):
        """Reverses the DataFrame."""
        if reverse:
            df = df.iloc[::-1].reset_index(drop=True)
        return df

    sql = 'DESCRIBE {};'.format(full_table_name)
    if print_query:
        print(sql)

    column_names_df = psql.read_sql(sql, con)
    column_names_df = _reorder_df(column_names_df, order_by)
    column_names_df = _reverse_df(column_names_df, reverse)

    return column_names_df


def fetch_table_names(con, schema_name=None, print_query=False):
    """Fetches all the table names in the specified database.

    Parameters
    ----------
    con : SQLAlchemy engine object or psycopg2 connection object
    schema_name : str
        Specify the schema of interest. If left blank, then it will
        return all tables in the database.
    print_query : bool, default False
        If True, print the resulting query

    Returns
    -------
    table_names_df : DataFrame
    """

    if schema_name is None:
        sql = 'SHOW TABLES;'
    else:
        sql = 'SHOW TABLES IN {};'.format(schema_name)

    if print_query:
        print(sql)

    table_names_df = psql.read_sql(sql, con)
    return table_names_df


def print_actual_query(slct):
    """Prints the actual SQL query from a SQLAlchemy selectable object
    instead of with the parameters anonymized.

    Parameters
    ----------
    slct : SQLAlchemy Select object
        The SQLAlchemy object that we wish to print the query of
    """

    # Prints the query with proper formatting
    print(slct.compile(compile_kwargs={'literal_binds': True}))


def save_df_to_db(df, table_name, engine, schema=None, batch_size=0,
                  partitioned_by=[], drop_table=False, print_query=False):
    """Saves a Pandas DataFrame to a database as a table. This function
    is useful if the user does not have access to SSH into the database
    and create tables from flat CSV files.

    Parameters
    ----------
    df : DataFrame
        The DataFrame we wish to save
    table_name : str
        A string indicating what we want to name the table
    engine : SQLAlchemy engine object
    batch_size : int, default 0
        The number indicates how many rows of the DataFrame  we would
        like to add at a time to the table. If 0, then add all.
    schema : str, default None
        The name of the schema
    partitioned_by : str or list, default None
        The specified partition key(s), if applicable
    drop_table : bool, default False
        If True, drop the table if it exists before creating new table
    print_query : bool, default False
        If True, print the resulting query
    """

    def _create_empty_table(full_table_name):
        """Creates an empty table based on a DataFrame."""
        # Set create table string
        create_str = 'CREATE TABLE {}'.format(full_table_name)

        # Specify column names and data types
        create_col_list = _get_create_col_list(df, partitioned_by)
        partition_col_list = _get_partition_col_list(df, partitioned_by)

        sep_str = ',\n    '
        create_col_str = sep_str.join(create_col_list)
        partition_col_str = sep_str.join(partition_col_list)

        if len(partition_col_list) > 0:
            create_table_str = ('{create_str} (\n'
                                '    {create_col_str}\n'
                                ')\n'
                                ' PARTITIONED BY (\n'
                                '    {partition_col_str}\n'
                                ');'
                               ).format(**locals())
        else:
            create_table_str = ('{create_str} (\n'
                                '    {create_col_str}\n'
                                ');'
                               ).format(**locals())

        if print_query:
            print(create_table_str)

        # Create the table with no rows
        psql.execute(create_table_str, engine)

        return create_col_list, partition_col_list

    def _add_quotes_to_data(df):
        """Adds quotes to string data types and converts missing values
        to NULL.
        """

        df = df.copy()
        for col_name in df:
            data_type = _from_df_type_to_sql_type(df[col_name].dtypes)
            if data_type in ['STRING', 'TIMESTAMP']:
                df[col_name] = df[col_name].map(_add_quotes)

        return df

    def _add_quotes(x):
        """Adds quotation marks to a string."""
        # Assigns null values 'NULL'. Otherwise, they we will be set as
        # 'None'
        if pd.isnull(x):
            return 'NULL'
        else:
            return "'{}'".format(x)

    def _get_partition_vals():
        """Gets the values used for partitioning."""
        # Filter so only distinct values of partition column(s) remain
        distinct_df = df[partitioned_by].drop_duplicates()

        # List of dictionaries that map the column names to their values
        partition_vals = [row.fillna('NULL').to_dict()
                              for _, row in distinct_df.iterrows()]

        return partition_vals

    def _filter_df_on_partition():
        """Filters a DataFrame on a partition dictionary."""
        sub_df = df.copy()
        for k, v in partition_dict.items():
            if v == 'NULL':
                sub_df = sub_df[pd.isnull(sub_df[k])]
            else:
                sub_df = sub_df[sub_df[k] == v]
        return sub_df

    def _add_rows_to_table(sub_df, partition_dict=None):
        """Adds a subset of rows to a SQL table from a DataFrame. The
        purpose of this is to do it in batches for quicker insert time.
        """

        # FIXME: Incorporate inserting rows with partitions in batches
        insert_str = f'INSERT INTO {full_table_name}\n'

        # VALUES Clause
        # Each entry represents a row of the DataFrame that is being
        # inserted into the table
        values_list = [_create_row_insert_sql(sub_df.iloc[i], partition_dict)
                           for i in range(len(sub_df))]
        values_str = 'VALUES\n{}'.format(',\n'.join(values_list))

        # Add PARTITION Clause if specified
        if partition_dict is not None:
            partition_vals_str = _get_partition_vals_str(partition_dict)
            partition_str = f'PARTITION {partition_vals_str}\n'
            insert_values_str = f'{insert_str}{partition_str}{values_str};'
        else:
            insert_values_str = f'{insert_str}{values_str};'

        if print_query:
            print(insert_values_str)

        psql.execute(insert_values_str, engine)

    def _create_row_insert_sql(row_srs, partition_dict=None):
        """Converts a DataFrame row to a string to be used in an INSERT
        SQL query.
        """

        # Fills missing numeric columns with NULL, then casts the
        # remaining columns to string
        str_row_srs = row_srs.fillna('NULL').astype(str)

        # Remove partition columns since they should not be in the
        # VALUES part of the query
        if partition_dict is not None:
            str_row_srs = str_row_srs.drop(list(partition_dict.keys()))

        insert_sql = ', '.join(str_row_srs)
        insert_sql = '({})'.format(insert_sql)

        return insert_sql

    def _get_partition_vals_str(partition_dict):
        """Returns the partition string from the partition dict."""
        partition_vals_str_list =\
            [f'{k}={v}' for k, v in partition_dict.items()]

        partition_vals_str =\
            '({})'.format(', '.join(partition_vals_str_list))

        return partition_vals_str


    if batch_size < 0 or not isinstance(batch_size, int):
        raise ValueError('batch_size should be a non-negative integer.')

    if drop_table:
        _drop_table(table_name, schema, engine)

    # Set full table name
    if schema is None:
        full_table_name = table_name
    else:
        full_table_name = '{}.{}'.format(schema, table_name)

    create_col_list, partition_col_list = _create_empty_table(full_table_name)
    df = _add_quotes_to_data(df)

    if isinstance(partitioned_by, str):
        partitioned_by = [partitioned_by]

    if len(partitioned_by) > 0:
        # List of dicts representing the partitions
        partition_vals = _get_partition_vals()
        for partition_dict in partition_vals:
            sub_df = _filter_df_on_partition()
            _add_rows_to_table(sub_df, partition_dict)

    else:
        if batch_size == 0:
            # Add all rows at once
            _add_rows_to_table(df)
        else:
            nrows = df.shape[0]
            # Gets indices that define the start points of each batch
            batch_indices = list(range(0, nrows, batch_size)) + [nrows]

            # Add rows in batches
            for i in np.arange(len(batch_indices) - 1):
                start_index = batch_indices[i]
                stop_index = batch_indices[i+1]
                sub_df = df.iloc[start_index:stop_index]
                _add_rows_to_table(sub_df)


def save_table(data, table_name, engine, schema=None,
               partitioned_by=[], drop_table=False, print_query=False,
               stage=None):
    """Saves a SQLAlchemy selectable object to database.

    Parameters
    ----------
    data : SQLAlchemy Alias
        A table we wish to save
    table_name : str
        What we want to name the table
    engine : SQLAlchemy engine object
    schema : str, default None
        The name of the schema
    partitioned_by : str or list, default None
        The specified partition key(s), if applicable
    drop_table : bool, default False
        If True, drop the table if it exists before creating new table
    print_query : str, default False
        If True, print the resulting query
    stage : str, default None
        Determines which stage of the creation process should be done.
        It should be one of 'drop', 'create', or 'insert'. This is
        implemented since sometimes there can be a glitch in Impala
        where there is a delay between dropping and creating a table.
        So, running them all consecutively in a function may not work.
        This parameter allows to specify which stage is being done.
    """

    def _create_empty_table():
        """Creates an empty table based on a SQLAlchemy selected table."""
        # Set full table name
        if schema is None:
            full_table_name = table_name
        else:
            full_table_name = '{}.{}'.format(schema, table_name)

        # Set create table string
        create_str = 'CREATE TABLE {}'.format(full_table_name)

        # Specify column names and data types. Double quotes allow for
        # column names with different punctuation (e.g., spaces).
        create_col_list = _get_create_col_list(data, partitioned_by)
        partition_col_list = _get_partition_col_list(data, partitioned_by)

        sep_str = ',\n    '
        create_col_str = sep_str.join(create_col_list)
        partition_col_str = sep_str.join(partition_col_list)

        if len(partition_col_list) > 0:
            create_table_str = ('{create_str} (\n'
                                '    {create_col_str}\n'
                                ')\n'
                                ' PARTITIONED BY (\n'
                                '    {partition_col_str}\n'
                                ');'
                               ).format(**locals())
        else:
            create_table_str = ('{create_str} (\n'
                                '    {create_col_str}\n'
                                ');'
                               ).format(**locals())

        if print_query:
            print(create_table_str)

        # Create the table with no rows
        psql.execute(create_table_str, engine)

    if drop_table and (stage == 'drop' or stage is None):
        # TODO: There is a delay when dropping and creating a table
        # immediately after. Look into using cur instead of engine to
        # potentially solve it.
        _drop_table(table_name, schema, engine, print_query)

    # Create an empty table with the desired columns
    if stage == 'create' or stage is None:
        _create_empty_table()

    if stage == 'insert' or stage is None:
        metadata = MetaData(engine)
        created_table = Table(table_name, metadata,
                              autoload=True, schema=schema)

        # Insert rows from selected table into the new table
        insert_sql = created_table\
            .insert()\
            .from_select(data.c,
                         select=data
                        )

        psql.execute(insert_sql, engine)


def subset_join(main_data, subset_data, join_key):
    """Subsets the rows of main_data according to subset_data. This is
    achieved by joining the main_data to the subset_data on the join_key
    and selecting the columns of main_data.

    Parameters
    ----------
    main_data : SQLAlchemy Alias/Table
        The main table which will be subsetted
    subset_data : SQLAlchemy Alias/Table
        The table which defines what values of main_data to subset
    join_key : str or list of str
        The join key(s) that define how to join main_data to subset_data

    Returns
    -------
    main_subsetted_alias : SQLAlchemy Alias
    """

    def _create_on_clause():
        """Creates the join on clause."""
        if isinstance(join_key, str):
            join_clause = (main_data.c[join_key] == subset_data.c[join_key])
        elif isinstance(join_key, list):
            join_clause_list = [main_data.c[k] == subset_data.c[k]
                                    for k in join_key]
            join_clause = and_(*join_clause_list)
        else:
            raise ValueError('join_key should be a string or list of strings.')

        return join_clause


    # Defines what the tables should be joined on
    join_clause = _create_on_clause()

    # Join main data to subset data
    data_join = main_data\
        .join(subset_data,
              onclause=join_clause
             )

    main_subsetted_alias =\
        select(main_data.c,
               from_obj=data_join
              )\
        .alias('main_subsetted')

    return main_subsetted_alias
