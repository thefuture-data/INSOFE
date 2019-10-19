
import pandas as pd
import numpy as np
import sys
import random
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from impyute.imputation.cs import fast_knn
from math import sqrt
