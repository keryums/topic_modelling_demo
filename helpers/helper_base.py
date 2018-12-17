### K Ong | Fall 2018
### Helper functions for basic reading and rendering

### The following function is taken from Jeremy Howard: https://www.linkedin.com/in/howardjeremy/
### It allows jupyter to render wide tables showing all columns and a scroll bar
### replacing the default format of ellipsis indicating more columns 

from IPython.core.display import HTML, display
display(HTML("<style>.container {width:100% !important;} </style>"))

def display_all(df):
    with pd.option_context("display.max_rows", 1000): 
        with pd.option_context("display.max_columns", 1000): 
            display(df)


### The following functions read the zipped json data into a pandas dataframe
### Provided on J McCauley's website: http://jmcauley.ucsd.edu/data/amazon/
 
import pandas as pd
import gzip

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')
