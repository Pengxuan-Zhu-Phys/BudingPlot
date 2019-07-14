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
        self.Analytic_funcs()
        for fig in self.figs:
            self.drawpicture(fig)


    def drawpicture(self, fig):
        fig['fig'] = plt.figure(figsize=(10, 8))
        if fig['type'] == "2D_Stat_Profile":
            ax = fig['fig'].add_axes([0.16, 0.16, 0.68, 0.81])
            axc = fig['fig'].add_axes([0.845, 0.16, 0.02, 0.81])
            self.basic_selection(fig)
            self.GetStatData(fig)
            fig['var']['BestPoint'] = {
                'x': fig['var']['data'].loc[fig['var']['data'].Stat.idxmin()].x,
                'y': fig['var']['data'].loc[fig['var']['data'].Stat.idxmin()].y,
                'Stat': fig['var']['data'].loc[fig['var']['data'].Stat.idxmin()].Stat
            }
            print(fig['var']['data'])
            print(fig['var']['BestPoint'])
            # fig['var']['data'].assign({'PL': fig['var']['data'].'Stat' })


    def GetStatData(self, fig):
        if self.cf.has_option(fig['section'], 'stat_variable'):
            fig['var'] = {}
            if self.cf.get(fig['section'], 'stat_variable').split(',')[0] == 'CHI2':
                self.get_variable_data(fig, 'x', self.cf.get(fig['section'], 'x_variable'))
                self.get_variable_data(fig, 'y', self.cf.get(fig['section'], 'y_variable'))
                self.get_variable_data(fig, 'Stat', ",".join(self.cf.get(fig['section'], 'stat_variable').split(',')[1:]).strip())
                fig['var']['data'] = pd.DataFrame({
                    'x': fig['var']['x'], 
                    'y': fig['var']['y'], 
                    'Stat': fig['var']['Stat']})
                fig['var'].pop('x')
                fig['var'].pop('y')
                fig['var'].pop('Stat')


    def get_variable_data(self, fig, name, varinf):
        if varinf[0:3] == '&Eq':
            varinf = varinf[4:]
            x_sel = self.var_symbol(varinf)
            for x in x_sel:
                varinf = varinf.replace("_{}".format(x), "fig['data']['{}']".format(x))
            if "&FC_" in varinf:
                for ii in range(len(self.funcs)):
                    varinf = varinf.replace(self.funcs[ii]['name'], "self.funcs[{}]['expr']".format(ii))
            fig['var'][name] = eval(varinf)
        elif varinf in fig['data'].columns.values:
            fig['var'][name] = fig['data'][name]
        else:
            print("No Variable {} found in Data!".format(varinf))
            sys.exit(0)


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

    def Analytic_funcs(self):
        self.funcs = []
        for item in self.cf.sections():
            if "FUNCTION1D" in item:
                fun = {}
                fun['name'] = "&FC_{}".format(self.cf.get(item, 'name'))
                fun['data'] = np.loadtxt(self.cf.get(item, 'file'))
                fun['expr'] = interp1d(fun['data'][:, 0], fun['data'][:, 1])
                self.funcs.append(fun)
        self.funcs = tuple(self.funcs)

    def var_symbol(self, bo):
        x = []
        bo = bo.split()
        for it in bo:
            if it[0] == '_' and it not in x:
                x.append(it[1:])
        return x

    def basic_selection(self, fig):
        if self.cf.has_option(fig['section'], 'selection'):
            bo = self.cf.get(fig['section'], 'selection')
            if bo[0:3] == '&Bo':
                bo = bo[4:]
                x_sel = self.var_symbol(bo)
                for x in x_sel:
                    bo = bo.replace("_{}".format(x), "self.data['{}']".format(x))
            if "&FC_" in bo:
                for ii in range(len(self.funcs)):
                    bo = bo.replace(self.funcs[ii]['name'], "self.funcs[{}]['expr']".format(ii))
            fig['data'] = self.data[eval(bo)].reset_index()




