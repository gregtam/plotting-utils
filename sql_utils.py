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
                                              select, text, true
from sqlalchemy import BigInteger, Boolean, Date, DateTime, Integer, Float,\
                       Numeric, String
from sqlalchemy.dialects.postgresql import aggregate_order_by
from sqlalchemy.sql.selectable import Alias


def _get_distribution_str(distribution_key, randomly):
    """Set distribution key string"""
    if distribution_key is None and not randomly:
        return ''
    elif distribution_key is None and randomly:
        return 'DISTRIBUTED RANDOMLY'
    elif distribution_key is not None and not randomly:
        if isinstance(distribution_key, Column):
            return 'DISTRIBUTED BY ({})'.format(distribution_key.name)
        elif isinstance(distribution_key, str):
            return 'DISTRIBUTED BY ({})'.format(distribution_key)
        else:
            raise ValueError('distribution_key must be a string or a Column.')
    else:
        raise ValueError('distribution_key and randomly cannot both be specified.')
    

def _separate_schema_table(full_table_name, conn):
    """Separates schema name and table name"""
    if '.' in full_table_name:
        return full_table_name.split('.')
    else:
        schema_name = psql.read_sql('SELECT current_schema();', conn).iloc[0, 0]
        table_name = full_table_name
        return schema_name, full_table_name



def clear_schema(schema_name, conn, print_query=False):
    """Remove all tables in a given schema.

    Parameters
    ----------
    schema_name : str
        Name of the schema in SQL
    conn : A psycopg2 connection object
    print_query : bool, default False
        If True, print the resulting query
    """

    sql = '''
    SELECT table_name
      FROM information_schema.tables
     WHERE table_schema = '{schema_name}'
    '''.format(**locals())

    if print_query:
        print dedent(sql)

    table_names = psql.read_sql(sql, conn).table_name

    for table_name in table_names:
        del_sql = 'DROP TABLE IF EXISTS {schema_name}.{table_name};'\
            .format(**locals())
        psql.execute(del_sql, conn)


def get_column_names(full_table_name, conn, order_by='ordinal_position',
                     reverse=False, print_query=False):
    """Gets all of the column names of a specific table.

    Parameters
    ----------
    conn : A psycopg2 connection object
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
    """

    schema_name, table_name = _separate_schema_table(full_table_name, conn)

    if reverse:
        reverse_key = ' DESC'
    else:
        reverse_key = ''

    sql = '''
    SELECT table_name, column_name, data_type
      FROM information_schema.columns
     WHERE table_schema = '{schema_name}'
       AND table_name = '{table_name}'
     ORDER BY {order_by}{reverse_key};
    '''.format(**locals())

    if print_query:
        print dedent(sql)

    return psql.read_sql(sql, conn)


def get_function_code(function_name, conn, print_query=False):
    """Returns a SQL function's source code."""
    sql = '''
    SELECT pg_get_functiondef(oid)
      FROM pg_proc
     WHERE proname = '{function_name}'
    '''.format(**locals())

    if print_query:
        print dedent(sql)

    return psql.read_sql(sql, conn).iloc[0, 0]


def get_table_names(conn, schema_name=None, print_query=False):
    """Gets all the table names in the specified database.

    Parameters
    ----------
    conn : A psycopg2 connection object
    schema_name : str
        Specify the schema of interest. If left blank, then it will 
        return all tables in the database.
    print_query : bool, default False
        If True, print the resulting query
    """

    if schema_name is None:
        where_clause = ''
    else:
        where_clause = "WHERE table_schema = '{}'".format(schema_name)

    sql = '''
    SELECT table_name
      FROM information_schema.tables
     {}
    '''.format(where_clause)

    if print_query:
        print dedent(sql)

    return psql.read_sql(sql, conn)


def get_percent_missing(full_table_name, conn, print_query=False):
    """This function takes a schema name and table name as an input and
    creates a SQL query to determine the number of missing entries for
    each column. It will also determine the total number of rows in the
    table.

    Parameters
    ----------
    full_table_name : str
        Name of the table in SQL. Input can also include have the schema
        name prepended, with a '.', e.g., 'schema_name.table_name'.
    conn : A psycopg2 connection object
    print_query : bool, default False
        If True, print the resulting query
    """

    column_names = get_column_names(full_table_name, conn).column_name
    schema_name, table_name = _separate_schema_table(full_table_name, conn)

    num_missing_sql_list = ['SUM(({name} IS NULL)::INTEGER) AS {name}'\
                                .format(name=name) for name in column_names]

    num_missing_list_str = ',\n           '.join(num_missing_sql_list)

    sql = '''
    SELECT {num_missing_list_str},
           COUNT(*) AS total_count
      FROM {schema_name}.{table_name};
    '''.format(**locals())

    # Read in the data from the query and transpose it
    pct_df = psql.read_sql(sql, conn).T

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


def get_process_ids(conn, usename=None, print_query=False):
    """Gets the process IDs of current running activity.

    Parameters
    ----------
    conn : A psycopg2 connection object
    usename : str, default None
        Username to filter by. If None, then do not filter.
    print_query : bool, default False
        If True, print the resulting query

    Returns a Pandas DataFrame
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

    return psql.read_sql(sql, conn)


def count_distinct_values(tbl, engine):
    """Counts the number of distinct values for each column of a table.
    
    Parameters
    ----------
    tbl : str or SQLAlchemy Table
    engine : SQLAlchemy engine object
    """

    if not isinstance(tbl, (str, Table, Alias)):
        raise TypeError('tbl must be of str or Table type.')
    if isinstance(tbl, str):
        metadata = MetaData(engine)
        tbl = Table(tbl, metadata, autoload=True)

    count_distinct_df = pd.DataFrame(columns=['column_name', 'n_distinct'])
    for tbl_col in tbl.c:
        group_by_alias =\
            select([tbl_col],
                   from_obj=tbl
                  )\
            .group_by(tbl_col)\
            .alias('group_by')

        count =\
            select([func.count('*')],
                   from_obj=group_by_alias
                  )\
            .execute()\
            .scalar()

        new_row = (tbl_col.name, count)
        count_distinct_df.loc[count_distinct_df.shape[0]] = new_row

    return count_distinct_df


def kill_process(conn, pid, print_query=False):
    """Kills a specified process.

    Parameters
    ----------
    conn : A psycopg2 connection object
    pid : int
        The process ID that we want to kill
    print_query : bool, default False
        If True, print the resulting query
    """

    sql = '''
    SELECT pg_cancel_backend({});
    '''.format(pid)

    psql.execute(sql, conn)


def save_df_to_db(df, table_name, engine, batch_size=0,
                  distribution_key=None, randomly=False, drop_table=False,
                  print_query=False):
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
    distribution_key : str, default ''
        The specified distribution key, if applicable
    randomly : bool, default False
        If True, distribute the table randomly
    drop_table : bool, default False
        If True, drop the table if before creating the new table
    print_query : bool, default False
        If True, print the resulting query
    """

    def _add_quotes(x):
        """Adds quotation marks to a string."""
        if x is None:
            return 'NULL'
        else:
            return "'{}'".format(x)

    def _get_data_type(col_name):
        """Gets the data type of a Pandas DataFrame column."""
        test_entry = df[col_name].dropna().iloc[0]
        if isinstance(test_entry, (int, float)):
            return 'numeric'
        elif isinstance(test_entry, (str)):
            return 'text'

    def _get_array_str(col_name):
        """Takes a Pandas DataFrame column, creates a SQL string to
        create an ARRAY out of it, then UNNESTs it.
        """

        def _get_unnest_array_str(array_str):
            """Adds the final UNNEST and ARRAY operators."""
            return 'UNNEST(ARRAY[{}])'.format(array_str)

        data_type = _get_data_type(col_name)

        if data_type == 'numeric':
            # Creates a list from a numeric column
            vals_list = df[col_name]\
                .fillna('NULL')\
                .astype(str)\
                .tolist()
        elif data_type == 'text':
            # Creates a list from a text column by adding single quotes
            vals_list = df[col_name].map(_add_quotes).tolist()
        else:
            pass

        array_str = ', '.join(vals_list)
        return _get_unnest_array_str(array_str)

    def _from_df_type_to_sql_type(type_val):
        """Converts a DataFrame data type to a SQL type."""
        type_str = type_val.name

        if type_str == 'object':
            return 'TEXT'
        elif 'int' in type_str:
            return 'INTEGER'
        elif 'float' in type_str:
            return 'NUMERIC'

    def _create_empty_table(df, table_name, distribution_key, randomly):
        """Creates an empty table based on a DataFrame."""
        # Set create table string
        create_str = 'CREATE TABLE {} ('.format(table_name)

        # Specify column names and data types
        name_type_list = ['{} {}'.format(k, _from_df_type_to_sql_type(v))
                              for k, v in df.dtypes.iteritems()]
        columns_str = ',\n'.join(name_type_list)

        # Set distribution key
        distribution_str = _get_distribution_str(distribution_key, randomly)

        create_table_str = '{create_str}{columns_str}) {distribution_str};'\
            .format(**locals())

        # Create the table with no rows
        psql.execute(create_table_str, engine)

    def _convert_nan_to_none(vec):
        """Converts NaN values to None in lists."""
        return [val if not np.isnan(val) else None for val in vec]

    def _add_rows_to_table(sub_df, tbl):
        """Adds a subset of rows to a SQL table from a DataFrame. The
        purpose of this is to do it in batches for quicker insert time.
        """

        # Convert all DataFrame columns to lists, and create a
        # dictionary of these lists. For numeric lists, NULL values
        # will appear as NaN types. We will need to convert these NaN
        # values to None so they are interpreted properly in SQL.
        col_dict = {}
        for col, type_val in sub_df.dtypes.iteritems():
            # If all entries in the column for this batch are NULL, then
            # do not include in the insert, or an error will occur if it
            # tries to add a column of only NULL values.
            all_null = np.all(pd.isnull(sub_df[col].tolist()))
            if all_null:
                break

            if _from_df_type_to_sql_type(type_val) == 'TEXT':
                col_dict[col] = sub_df[col].tolist()
            else:
                col_dict[col] = _convert_nan_to_none(sub_df[col].tolist())

        # Apply an unnest on each
        col_unnest_list = [(column(col), func.unnest(col_list))
                               for col, col_list in col_dict.iteritems()]
        col_names =  zip(*col_unnest_list)[0]
        col_lists =  zip(*col_unnest_list)[1]

        # Form a select statement
        select_statement = select(col_lists)

        # Add the rows using an insert
        tbl.insert()\
            .from_select(col_names,
                         select_statement
                        )\
            .execute()

    if drop_table:
        drop_sql = 'DROP TABLE IF EXISTS {}'.format(table_name)
        psql.execute(drop_sql, engine)

    _create_empty_table(df, table_name, distribution_key, randomly)

    metadata = MetaData(engine)
    new_tbl = Table(table_name, metadata, autoload=True)

    if batch_size < 0 or not isinstance(batch_size, int):
        raise ValueError('batch_size should be a non-negative integer')
    elif batch_size == 0:
        _add_rows_to_table(df, new_tbl)
    else:
        nrows = df.shape[0]
        batch_indices = range(0, nrows, batch_size) + [nrows]
        
        # Add rows in batches
        for i in np.arange(len(batch_indices) - 1):
            start_index = batch_indices[i]
            stop_index = batch_indices[i+1]
            _add_rows_to_table(df.iloc[start_index:stop_index], new_tbl)


def save_table(selected_table, table_name, engine, distribution_key=None,
               randomly=False, drop_table=False, temp=False,
               print_query=False):
    """Saves a SQLAlchemy selectable object to database.
    
    Parameters
    ----------
    selected_table : SQLAlchemy selectable object
        A table we wish to save
    table_name : str
        What we want to name the table
    engine : SQLAlchemy engine object
    distribution_key : str, default None
        The specified distribution key, if applicable
    randomly : bool, default False
        If True, distribute table randomly
    drop_table : bool, default False
        If True, drop the table if it exists before creating new table
    temp : bool, default False
        If True, then create a temporary table instead
    print_query : str, default False
        If True, print the resulting query
    """

    def _create_empty_table(selected_table, table_name, engine,
                            distribution_key, randomly, temp, print_query):
        """Creates an empty table based on a SQLAlchemy selected table."""
        # Set create table string
        if temp:
            create_str = 'CREATE TEMP TABLE {} ('.format(table_name)
        else:
            create_str = 'CREATE TABLE {} ('.format(table_name)

        # Specify column names and data types
        columns_str = ',\n'.join(['{} {}'.format(s.name, s.type)
                                      for s in selected_table.c])
        # Set distribution key
        distribution_str = _get_distribution_str(distribution_key, randomly)

        create_table_str = '{create_str}{columns_str}) {distribution_str};'\
            .format(**locals())

        if print_query:
            print create_table_str

        # Create the table with no rows
        psql.execute(create_table_str, engine)

    if drop_table:
        psql.execute('DROP TABLE IF EXISTS {};'.format(table_name), engine)

    # Create an empty table with the desired columns
    _create_empty_table(selected_table, table_name, engine, distribution_key,
                        randomly, temp, print_query)
 
    metadata = MetaData(engine)
    created_table = Table(table_name, metadata, autoload=True)
    # Insert rows from selected table into the new table
    insert_sql = created_table\
        .insert()\
        .from_select(selected_table.c,
                     select=selected_table
                    )
    psql.execute(insert_sql, engine)
