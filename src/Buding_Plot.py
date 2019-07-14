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
        pass 

    def get_inf(self, inf):
        self.cf = configparser.ConfigParser()
        self.cf.read(inf)
    
    def load_data(self):
        self.data = []
        for opt in self.cf.get("PLOT_CONFI", 'result_file').split('\n'):
            dat = pd.read_csv("{}{}".format(self.cf.get('PLOT_CONFI', 'path'), opt.split()[1]), delimiter='\t')
            dat = dat.drop(["Index"], axis=1)
            self.data.append(dat)
        self.data = pd.concat(self.data, axis=0, join='outer', ignore_index=True)

    def plot(self):
        self.load_data()
        self.figures_inf()
        for fig in self.figs:
            self.drawpicture(fig)
            # print(fig)
        # print(self.figs)

    def drawpicture(self, fig):
        fig['fig'] = plt.figure(figsize=(10, 8))
        if fig['type'] == "2D_Stat_Profile":
            ax = fig['fig'].add_axes([0.16, 0.16, 0.68, 0.81])
            axc = fig['fig'].add_axes([0.845, 0.16, 0.02, 0.81])
            self.basic_selection(fig)

    def figures_inf(self):
        if self.cf.has_option('PLOT_CONFI', 'plot'):
            ppt = self.cf.get('PLOT_CONFI', 'plot').split('\n')
            self.figs = []
            for line in ppt:
                pic = {}
                pic['type'] = line.split(',')[0]
                if "ALL" in ','.join(line.split(',')[1:]):
                    for item in self.cf.sections():
                        if pic['type'] in item:
                            self.figs.append({
                                'name':     self.cf.get(item, 'plot_name'),
                                'section':  item,
                                'type':     pic['type']
                                })
                else:
                    for item in line.split(',')[1:]:
                        if self.cf.has_section('PLOT_{}_{}'.format(pic['type'], item.strip())):
                            sec = 'PLOT_{}_{}'.format(pic['type'], item.strip())
                            self.figs.append({
                                'section':  sec,
                                'name':     self.cf.get(sec, 'plot_name'),
                                'type':     pic['type']
                                })

    def bool_symbol(self, bo):
        x = []
        bo = bo[4:].split()
        for it in bo:
            if it[0] == '_' and it not in x:
                x.append(it[1:])
        return x

    def basic_selection(self, fig):
        if self.cf.has_option(fig['section'], 'selection'):
            bo = self.cf.get(fig['section'], 'selection')
            if bo[0:3] == '&Bo':
                x_sel = self.bool_symbol(bo)
                bo = bo[4:]
                for x in x_sel:
                    bo = bo.replace("_{}".format(x), "self.data['{}']".format(x))
                fig['data'] = self.data[eval(bo)]
                print(fig['data'])




