#!/usr/bin/env python3

import os, sys
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
from matplotlib import cm
from matplotlib.font_manager import FontProperties
import numpy as np 
import matplotlib
rc('text', usetex=True)
import sympy
from scipy.interpolate import interp1d
import re 
from matplotlib.patches import Rectangle
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as ticker
import pandas as pd 
import copy
from scipy.interpolate import Rbf
import math
import configparser

class Figure():
    def __init__(self):
        print("Hello")

    def get_inf(self, inf):
        self.cf = configparser.ConfigParser()
        self.cf.read(inf)
    
    def get_data(self):
        # print("Hello")
        for opt in self.cf.get("PLOT_CONFI", 'result_file').split('\n')

