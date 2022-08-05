# time
import datetime as dt
from time import time

# data calculation
import numbers
import math
from math import ceil, floor
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from collections import defaultdict
from itertools import chain, combinations

# data processing
import pandas as pd
import numpy as np
from numpy.lib.index_tricks import IndexExpression

# utils
import os
from inspect import signature
from scipy.special import comb
from sklearn.utils import indexable, check_random_state, _safe_indexing
from sklearn.utils import _approximate_mode
from sklearn.utils.validation import _num_samples
from sklearn.model_selection._split import _BaseKFold
from sklearn.base import _pprint

import warnings
warnings.filterwarnings('ignore')

# vis
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
pd.set_option('display.float_format', lambda x: '%.5f' % x)
mpl.rcParams['figure.figsize'] = (20, 3)
mpl.rcParams['axes.grid'] = False

# modeling
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import *
from tensorflow.keras import layers
from tensorflow.keras.callbacks import History, EarlyStopping
from tensorflow import keras


from Blog.BlogHelper.Modeling_Helpers import *
from Blog.BlogHelper.Plot_Helpers import *
from Blog.BlogHelper.Experiment_Helpers import *
from Blog.BlogHelper.Windows_Helpers import *
