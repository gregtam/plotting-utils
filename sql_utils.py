from __future__ import division
from textwrap import dedent

import numpy as np
import pandas as pd
import pandas.io.sql as psql
import psycopg2
from sqlalchemy import Column, Table, MetaData
from sqlalchemy import all_, and_, any_, not_, or_
from sqlalchemy import alias, between, case, cast, column, distinct, extract,\
                       false, func, intersect, literal, literal_column,\
                       select, text, true, union, union_all
from sqlalchemy import BigInteger, Boolean, Date, DateTime, Integer, Float,\
                       Numeric, String
from sqlalchemy.dialects.postgresql import aggregate_order_by
from sqlalchemy.sql.selectable import Alias



def _drop_table(table_name, schema, engine):
    """Drops a SQL table."""
    if schema is None:
        drop_str = 'DROP TABLE IF EXISTS {};'.format(table_name)
    else:
        drop_str = 'DROP TABLE IF EXISTS {}.{};'.format(schema, table_name)
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
        partition_by = [partitioned_by]

    if isinstance(data, (Alias, Table)):
        create_col_list = [_get_single_partitioned_str(data, col.name)
                               for col in data.c
                                   if col.name not in partitioned_by]
    elif isinstance(data, pd.DataFrame):
        create_col_list = ['{} {}'.format(k, _from_df_type_to_sql_type(v))
                               for k, v in data.dtypes.iteritems()
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
                                  for k, v in data.dtypes.iteritems()
                                      if k in partitioned_by]
    return partition_col_list


def _get_single_partitioned_str(selected_table, col_str):
    """Returns a string of the column name and type for partitioning."""
    col = selected_table.c[col_str]
    return '{} {}'.format(col.name, col.type.__visit_name__)
    

def _separate_schema_table(full_table_name, con):
    """Separates schema name and table name"""
    if '.' in full_table_name:
        return full_table_name.split('.')
    else:
        schema_name = psql.read_sql('SELECT current_schema();', con).iloc[0, 0]
        table_name = full_table_name
        return schema_name, full_table_name



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

    sql = '''
    SHOW TABLES IN {schema_name};
    '''.format(**locals())

    if print_query:
        print dedent(sql)

    table_names = psql.read_sql(sql, con).table_name

    for table_name in table_names:
        del_sql = 'DROP TABLE IF EXISTS {schema_name}.{table_name};'\
            .format(**locals())
        psql.execute(del_sql, con)


def count_distinct_values(tbl, engine, approx=False):
    """Counts the number of distinct values for each column of a table.
    
    Parameters
    ----------
    tbl : str or SQLAlchemy Table
    engine : SQLAlchemy engine object
    approx : bool, default False
        Whether to approximate (uses ndv() function)

    Returns
    -------
    count_distinct_df : DataFrame
    """

    if not isinstance(tbl, (str, Table, Alias)):
        raise TypeError('tbl must be of str or Table type.')
    if isinstance(tbl, str):
        metadata = MetaData(engine)
        tbl = Table(tbl, metadata, autoload=True)

    count_distinct_df = pd.DataFrame(columns=['column_name', 'n_distinct'])
    for tbl_col in tbl.c:
        if ndv:
            count =\
                select([func.ndv(column(tbl_col))],
                       from_obj=tbl
                      )\
                .execute()\
                .scalar()
        else:
            count =\
                select([func.count(distinct(column(tbl_col)))],
                       from_obj=tbl
                      )\
                .execute()\
                .scalar()

        new_row = (tbl_col.name, count)
        count_distinct_df.loc[count_distinct_df.shape[0]] = new_row

    return count_distinct_df


def count_rows(from_obj, print_commas=False):
    """Counts the number of rows from a table or alias.

    Parameters
    ----------
    from_obj : A SQLAlchemy Table or Alias object
    print_commas : bool, default False
        Whether or not to print commas for the thousands separator

    Returns
    -------
    row_count : int
    """

    row_count =\
        select([func.count('*')],
               from_obj=from_obj
              )\
        .execute()\
        .scalar()
    if print_commas:
        print '{:,}'.format(row_count)
    return row_count


def get_column_names(full_table_name, con, order_by='ordinal_position',
                     reverse=False, print_query=False):
    """Gets all of the column names of a specific table.

    Parameters
    ----------
    con : SQLAlchemy engine object or psycopg2 connection object
    full_table_name : str
        Name of the table in SQL. Input can also include have the schema
        name prepended, with a '.', e.g., 'schema_name.table_name'.
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

    def _reorder(df, order_by):
        """Reorders the DataFrame."""
        if order_by not in ('ordinal_position', 'alphabetically'):
            raise ValueError("order_by must be either 'ordinal_position' or"
                             "'alphabetically'.")
        if order_by == 'alphabetically':
            df = df\
                .sort_values('name')\
                .reset_index(drop=True)
        return df

    def _reverse(df, reverse):
        """Reverses the DataFrame."""
        if reverse:
            df = df.iloc[::-1].reset_index(drop=True)
        return df

    sql = 'DESCRIBE {};'.format(full_table_name)
    if print_query:
        print sql

    column_names_df = psql.read_sql(sql, con)
    column_names_df = _reorder(column_names_df, order_by)
    column_names_df = _reverse(column_names_df, reverse)

    return column_names_df


def get_function_code(function_name, con, print_query=False):
    """Returns a SQL function's source code.
    
    Parameters
    ----------
    function_name : str
        The name of the function
    con : SQLAlchemy engine object or psycopg2 connection object
    print_query : bool, default False
        If True, print the resulting query

    Returns
    -------
    func_code : str
    """

    sql = '''
    SELECT pg_get_functiondef(oid)
      FROM pg_proc
     WHERE proname = '{function_name}'
    '''.format(**locals())

    if print_query:
        print dedent(sql)

    func_code = psql.read_sql(sql, con).iloc[0, 0]
    return func_code


def get_table_names(con, schema_name=None, print_query=False):
    """Gets all the table names in the specified database.

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
        print sql

    table_names_df = psql.read_sql(sql, con)
    return table_names_df


def get_percent_missing(full_table_name, con, print_query=False):
    """This function takes a schema name and table name as an input and
    creates a SQL query to determine the number of missing entries for
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
        print dedent(sql)

    return pct_df


def get_process_ids(con, usename=None, print_query=False):
    """Gets the process IDs of current running activity.

    Parameters
    ----------
    con : SQLAlchemy engine object or psycopg2 connection object
    usename : str, default None
        Username to filter by. If None, then do not filter.
    print_query : bool, default False
        If True, print the resulting query

    Returns
    -------
    pid_df : DataFrame
    """

    if usename is None:
        where_clause = ''
    else:
        where_clause = "WHERE usename = '{}'".format(usename)

    sql = '''
    SELECT datname, procpid, usename, current_query, query_start
      FROM pg_stat_activity
     {}
    '''.format(where_clause)

    if print_query:
        print dedent(sql)

    pid_df = psql.read_sql(sql, con)
    return pid_df


def kill_process(con, pid, print_query=False):
    """Kills a specified process.

    Parameters
    ----------
    con : SQLAlchemy engine object or psycopg2 connection object
    pid : int
        The process ID that we want to kill
    print_query : bool, default False
        If True, print the resulting query
    """

    sql = '''
    SELECT pg_cancel_backend({});
    '''.format(pid)

    psql.execute(sql, con)


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

    def _add_quotes(x):
        """Adds quotation marks to a string."""
        # Assigns null values 'NULL'. Otherwise, they we will be set as
        # 'None'
        if pd.isnull(x):
            return 'NULL'
        else:
            return "'{}'".format(x)

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

    def _add_rows_to_table(sub_df, full_table_name, partition_col_list,
                           create_col_list, print_query,
                           partition_dict=None):
        """Adds a subset of rows to a SQL table from a DataFrame. The
        purpose of this is to do it in batches for quicker insert time.
        """

        # TODO: Incorporate inserting rows with partitions in batches
        insert_str = 'INSERT INTO {}\n'.format(full_table_name)

        # VALUES Clause
        values_list = [_row_to_insert(sub_df.iloc[i], partition_dict)
                           for i in xrange(len(sub_df))]
        values_str = 'VALUES\n{}'.format(',\n'.join(values_list))

        # PARTITION Clause
        if partition_dict is not None:
            partition_vals_str = _get_partition_vals_str(partition_dict)
            partition_str = 'PARTITION {}\n'.format(partition_vals_str)
            insert_values_str = '{insert_str}{partition_str}{values_str};'\
                .format(**locals())
        else:
            insert_values_str = '{insert_str}{values_str};'.format(**locals())

        if print_query:
            print insert_values_str
        psql.execute(insert_values_str, engine)

    def _create_empty_table(df, full_table_name, engine, partitioned_by,
                            print_query):
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
            print create_table_str

        # Create the table with no rows
        psql.execute(create_table_str, engine)

        return create_col_list, partition_col_list

    def _convert_nan_to_none(vec):
        """Converts NaN values to None in lists."""
        return [val if not pd.isnull(val) else None for val in vec]

    def _filter_on_partition(df, partition_dict):
        """Filters a DataFrame on a partition dictionary."""
        sub_df = df.copy()
        for k, v in partition_dict.iteritems():
            if v == 'NULL':
                sub_df = sub_df[pd.isnull(sub_df[k])]
            else:
                sub_df = sub_df[sub_df[k] == v]
        return sub_df

    def _get_partition_vals_str(partition_dict):
        """Returns the partition string from the partition dict."""
        partition_vals_str_list = ['{}={}'.format(k, v)
                                       for k, v in partition_dict.iteritems()]
        partition_vals_str = '({})'.format(', '.join(partition_vals_str_list))
        return partition_vals_str

    def _get_partition_vals(df, parititoned_by):
        """Gets the values used for partitioning."""
        distinct_df = df[partitioned_by].drop_duplicates()

        partition_vals = [row.fillna('NULL').to_dict()
                              for i, row in distinct_df.iterrows()]
        return partition_vals

    def _row_to_insert(row_srs, partition_val=None):
        """Converts a DataFrame row to a string to be used in an INSERT
        SQL query.
        """

        # Fills missing numeric columns with NULL, then casts the
        # remaining columns to string
        str_row_srs = row_srs.fillna('NULL').astype(str)

        # Don't put partition columns into VALUES portion of query.
        if partition_val is not None:
            str_row_srs = str_row_srs.drop(partition_val.keys())

        insert_sql = ', '.join(str_row_srs)
        insert_sql = '({})'.format(insert_sql)
        return insert_sql

    if batch_size < 0 or not isinstance(batch_size, int):
        raise ValueError('batch_size should be a non-negative integer.')

    if drop_table:
        _drop_table(table_name, schema, engine)

    # Set full table name
    if schema is None:
        full_table_name = table_name
    else:
        full_table_name = '{}.{}'.format(schema, table_name)

    create_col_list, partition_col_list = _create_empty_table(df,
                                                              full_table_name,
                                                              engine,
                                                              partitioned_by,
                                                              print_query
                                                             )
    df = _add_quotes_to_data(df)

    if isinstance(partitioned_by, str):
        partitioned_by = [partitioned_by]

    if len(partitioned_by) > 0:
        # List of dicts representing the partitions
        partition_vals = _get_partition_vals(df, partitioned_by)
        for partition_dict in partition_vals:
            sub_df = _filter_on_partition(df, partition_dict)
            _add_rows_to_table(sub_df, full_table_name, partition_col_list,
                               create_col_list, print_query, partition_dict)

    else:
        if batch_size == 0:
            _add_rows_to_table(df, full_table_name, partition_col_list,
                               create_col_list, print_query)
        else:
            nrows = df.shape[0]
            batch_indices = range(0, nrows, batch_size) + [nrows]

            # Add rows in batches
            for i in np.arange(len(batch_indices) - 1):
                start_index = batch_indices[i]
                stop_index = batch_indices[i+1]
                sub_df = df.iloc[start_index:stop_index]
                _add_rows_to_table(sub_df, full_table_name, partition_col_list,
                                   create_col_list, print_query)


def save_table(selected_table, table_name, engine, schema=None,
               partitioned_by=[], drop_table=False, print_query=False):
    """Saves a SQLAlchemy selectable object to database.
    
    Parameters
    ----------
    selected_table : SQLAlchemy selectable object
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
    """

    def _create_empty_table(selected_table, table_name, engine, schema,
                            partitioned_by, print_query):
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
        create_col_list =_get_create_col_list(selected_table, partitioned_by)
        partition_col_list = _get_partition_col_list(selected_table, partitioned_by)

        sep_str = ',\n    '
        create_col_str = sep_str.join(create_col_list)
        partition_col_str = sep_str.join(partition_col_list)

        if len(partition_col_list) > 0:
            create_table_str = ('{create_str} ('
                                '\n    {create_col_str}'
                                '\n)'
                                '\n PARTITIONED BY ('
                                '\n    {partition_col_str}'
                                '\n);'
                               ).format(**locals())
        else:
            create_table_str = ('{create_str} ('
                                '\n    {create_col_str}'
                                '\n);'
                               ).format(**locals())

        if print_query:
            print create_table_str

        # Create the table with no rows
        psql.execute(create_table_str, engine)

    if drop_table:
        _drop_table(table_name, schema, engine)

    # Create an empty table with the desired columns
    _create_empty_table(selected_table, table_name, engine, schema,
                        partitioned_by, print_query)
 
    metadata = MetaData(engine)
    created_table = Table(table_name, metadata, autoload=True, schema=schema)

    # Insert rows from selected table into the new table
    insert_sql = created_table\
        .insert()\
        .from_select(selected_table.c,
                     select=selected_table
                    )
    psql.execute(insert_sql, engine)
