# python-utils
This repo includes files with various reusable Python functions for data science tasks.

*The `python3.5` branch uses `str.format()` functions throughout, since the master branch uses f strings, which are only available in Python 3.6+.*

### Python Files
- `ml_utils.py`: Various useful functions to manipulate pandas DataFrames for use in machine learning, such as balancing classes or creating a limited number of dummy variables.

- `pandas_excel_utils.py`: Functions for saving pandas DataFrames as Excel sheets in better formats (e.g., colouring to visually separate groups or saving very large DataFrames).

- `plotting_utils.py`: Plotting utilities, mostly centring around machine learning model results.

- `sql_utils.py`: Reusable utility functions for interacting with an Impala cluster (e.g., saving a sub-select as a table or counting the number of non-null entries of all columns).
