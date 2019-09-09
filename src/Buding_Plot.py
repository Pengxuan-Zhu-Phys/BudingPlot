#!/usr/bin/env python3

import os, sys
import json
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
import time
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

pwd = os.path.abspath(os.path.dirname(__file__))

class Figure():
    def __init__(self):
        pass 

    def get_inf(self, inf):
        self.cf = configparser.ConfigParser()
        self.cf.read(inf)
    
    def load_data(self):
        self.data = []
        for opt in self.cf.get("PLOT_CONFI", 'result_file').split('\n'):
            dat = pd.read_csv("{}{}".format(self.cf.get('PLOT_CONFI', 'path'), opt.split()[1]))
            # dat = dat.drop(["Index"], axis=1)
            self.data.append(dat)
        self.data = pd.concat(self.data, axis=0, join='outer', ignore_index=True)

    def load_colorset(self):
        if self.cf.has_option('COLORMAP', 'colorset'):
            self.colors = self.cf.get('COLORMAP', 'colorset').split('\n')
            for cs in self.colors:
                cset = {}
                cset['name'] = cs.split('|')[0].strip()
                if cs.split('|')[1].strip()[0:4] == 'SYS_':
                    cset['cmap'] = plt.get_cmap(cs.split('|')[1].strip()[4:])
                elif cs.split('|')[1].strip()[0:4] == 'Manu':
                    from matplotlib.colors import LinearSegmentedColormap
                    cset['cmap'] = LinearSegmentedColormap.from_list(cset['name'] ,eval(cs.split('|')[1].strip()[4:].strip()), N=256)
                else:
                    cset['cmap'] = cs.split('|')[1].strip()
                self.colors[self.colors.index(cs)] = cset
        elif self.cf.has_option("COLORMAP", 'colorsetting'):
            with open("{}/{}".format(pwd, self.cf.get('COLORMAP', 'colorsetting')), 'r', encoding='utf-8') as f1:
                # self.colors = json.load(f1)
                self.colors = eval(f1.read())
            for ii in range(len(self.colors)):
                if isinstance(self.colors[ii]['colormap'], str):
                    self.colors[ii]['colormap'] = plt.get_cmap(self.colors[ii]['colormap'])
                else:
                    from matplotlib.colors import LinearSegmentedColormap
                    self.colors[ii]['colormap'] = LinearSegmentedColormap.from_list(self.colors[ii]['name'], tuple(self.colors[ii]['colormap']), N=255)

    def plot(self):
        self.load_data()
        self.figures_inf()
        self.Analytic_funcs()
        self.load_colorset()
        for fig in self.figs:
            self.drawpicture(fig)

    # Plot Method is write in the drawpicture :
    # Add " elif fig['type'] == $FIGURE_TYPE$: "
    def drawpicture(self, fig):
        fig['fig'] = plt.figure(figsize=(10, 8))
        fig['fig'].text(0., 0., "Test", color="None")
        fig['fig'].text(1., 1., "Test", color="None")
        if fig['type'] == "2D_Stat_Profile":
            print("\n=== Ploting Figure : {} ===".format(fig['name']))
            fig['start'] = time.time()
            ax = fig['fig'].add_axes([0.13, 0.13, 0.74, 0.83])
            axc = fig['fig'].add_axes([0.875, 0.13, 0.015, 0.83])
            self.basic_selection(fig)
            print("\tTimer: {:.2f} Second;  Message from '{}' -> Data loading completed".format(time.time()-fig['start'], fig['section']))
            self.GetStatData(fig)
            fig['var']['BestPoint'] = {
                'x':    fig['var']['data'].loc[fig['var']['data'].Stat.idxmin()].x,
                'y':    fig['var']['data'].loc[fig['var']['data'].Stat.idxmin()].y,
                'Stat': fig['var']['data'].loc[fig['var']['data'].Stat.idxmin()].Stat
            }
            # Two Method to achieve Profile Likelihood Data in Pandas! Lambda expr is more compact in code
            # fig['var']['data'] = fig['var']['data'].assign( PL=np.exp( -0.5 * fig['var']['data'].Stat )/np.exp(-0.5 * fig['var']['BestPoint']['Stat'])  )
            fig['var']['data'] = fig['var']['data'].assign( PL=lambda x: np.exp( -0.5 * x.Stat )/np.exp(-0.5 * fig['var']['BestPoint']['Stat'])  )
            fig['var']['lim'] = {
                'x': [fig['var']['data'].x.min(), fig['var']['data'].x.max()],
                'y': [fig['var']['data'].y.min(), fig['var']['data'].y.max()],
            }
            fig['ax'] = {}
            self.ax_setlim(fig, 'xyc')
            XI, YI = np.meshgrid(
                np.linspace(fig['ax']['lim']['x'][0], fig['ax']['lim']['x'][1], int(self.cf.get(fig['section'], 'x_nbin'))+1),
                np.linspace(fig['ax']['lim']['y'][0], fig['ax']['lim']['y'][1], int(self.cf.get(fig['section'], 'y_nbin'))+1)                
            )
            fig['ax']['var'] = {
                'x':    XI,
                'y':    YI,
                'dx':   (fig['ax']['lim']['x'][1] - fig['ax']['lim']['x'][0])/int(self.cf.get(fig['section'], 'x_nbin')),
                'dy':   (fig['ax']['lim']['y'][1] - fig['ax']['lim']['y'][0])/int(self.cf.get(fig['section'], 'y_nbin'))
            }
            fig['ax']['grid'] = pd.DataFrame(
                index=np.linspace(fig['ax']['lim']['x'][0], fig['ax']['lim']['x'][1], int(self.cf.get(fig['section'], 'x_nbin'))+1),
                columns=np.linspace(fig['ax']['lim']['y'][0], fig['ax']['lim']['y'][1], int(self.cf.get(fig['section'], 'y_nbin'))+1)
            ).unstack().reset_index().rename(columns={'level_0':'yy','level_1':'xx',0:'z'})
            fig['ax']['grid'] = pd.DataFrame({
                "xi":   fig['ax']['grid']['xx'],
                'yi':   fig['ax']['grid']['yy'],
                'PL':   fig['ax']['grid'].apply(lambda tt: fig['var']['data'][ (fig['var']['data'].x > tt['xx'] - 0.5*fig['ax']['var']['dx']) & (fig['var']['data'].x < tt['xx'] + 0.5*fig['ax']['var']['dx']) & (fig['var']['data'].y > tt['yy'] - 0.5*fig['ax']['var']['dy']) & (fig['var']['data'].y < tt['yy'] + 0.5*fig['ax']['var']['dy']) ].PL.max(axis=0, skipna=True), axis=1)
            }).fillna({'PL': 0.})
            XI, YI = np.meshgrid(
                np.linspace(0, 1, int(self.cf.get(fig['section'], 'x_nbin'))+1),
                np.linspace(0, 1, int(self.cf.get(fig['section'], 'y_nbin'))+1)                
            )
            fig['ax']['mashgrid'] = {
                'x': XI,
                'y': YI
            }
            self.ax_setticks(fig, 'xyc')
            from matplotlib.ticker import MaxNLocator
            from matplotlib.tri import Triangulation, TriAnalyzer, UniformTriRefiner
            fig['ax']['tri'] = Triangulation(fig['ax']['grid']['xi'], fig['ax']['grid']['yi'])
            fig['ax']['refiner'] = UniformTriRefiner(fig['ax']['tri'])
            fig['ax']['tri_refine_PL'], fig['ax']['PL_refine']          = fig['ax']['refiner'].refine_field(fig['ax']['grid']['PL'], subdiv=3)
            fig['ax']['PL_refine'] = ( fig['ax']['PL_refine'] > 0.) * fig['ax']['PL_refine'] / (np.max(fig['ax']['PL_refine'])) + 0. * ( fig['ax']['PL_refine'] < 0.)
            print("\tTimer: {:.2f} Second;  Message from '{}' -> Data analysis completed".format(time.time()-fig['start'], fig['section']))

            levels = MaxNLocator(nbins=50).tick_values(fig['ax']['lim']['c'][0], fig['ax']['lim']['c'][1])
            self.ax_setcmap(fig)
            a1 = ax.tricontourf(fig['ax']['tri_refine_PL'], fig['ax']['PL_refine'], cmap=fig['colorset']['colormap'], levels=levels, zorder=1)
            plt.colorbar(a1, axc, ticks=fig['ax']['ticks']['c'], orientation='vertical', extend='neither')

            if 'curve' in fig['colorset'].keys():
                for curve in fig['colorset']['curve']:
                    ct = ax.tricontour(fig['ax']['tri_refine_PL'], fig['ax']['PL_refine'], [math.exp(-0.5 * curve['value'])], colors=curve['linecolor'], linewidths=curve['linewidth'], zorder=(10-curve['tag'])*4)

            if self.cf.has_option(fig['section'], 'BestPoint'):
                if eval(self.cf.get(fig['section'], 'BestPoint')) and 'bestpoint' in fig['colorset'].keys():
                    ax.scatter(fig['var']['BestPoint']['x'], fig['var']['BestPoint']['y'], 300, marker='*', color=fig['colorset']['bestpoint'][0], zorder=2000)
                    ax.scatter(fig['var']['BestPoint']['x'], fig['var']['BestPoint']['y'], 50, marker='*', color=fig['colorset']['bestpoint'][1], zorder=2100)


            ax.set_xticks(fig['ax']['ticks']['x'])
            ax.set_yticks(fig['ax']['ticks']['y'])
            ax.set_xlim(fig['ax']['lim']['x'][0], fig['ax']['lim']['x'][1])
            ax.set_ylim(fig['ax']['lim']['y'][0], fig['ax']['lim']['y'][1])

            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.tick_params(
                labelsize=fig['colorset']['ticks']['labelsize'], 
                direction=fig['colorset']['ticks']['direction'], 
                bottom=fig['colorset']['ticks']['bottom'], 
                left=fig['colorset']['ticks']['left'], 
                top=fig['colorset']['ticks']['top'], 
                right=fig['colorset']['ticks']['right'],
                which='both'
            )
            ax.tick_params(which='major', length=fig['colorset']['ticks']['majorlength'], color=fig['colorset']['ticks']['majorcolor'])
            ax.tick_params(which='minor', length=fig['colorset']['ticks']['minorlength'], color=fig['colorset']['ticks']['minorcolor'])
            axc.tick_params(
                labelsize=fig['colorset']['colorticks']['labelsize'], 
                direction=fig['colorset']['colorticks']['direction'], 
                bottom=fig['colorset']['colorticks']['bottom'], 
                left=fig['colorset']['colorticks']['left'], 
                top=fig['colorset']['colorticks']['top'], 
                right=fig['colorset']['colorticks']['right'], 
                color=fig['colorset']['colorticks']['color']
            )
            ax.set_xlabel(r"{}".format(self.cf.get(fig['section'], 'x_label')), fontsize=30)
            ax.set_ylabel(r"{}".format(self.cf.get(fig['section'], 'y_label')), fontsize=30)
            axc.set_ylabel(r"{}".format(self.cf.get(fig['section'], 'c_label')), fontsize=30)
            ax.xaxis.set_label_coords(0.5, -0.068)

            if 'save' in self.cf.get(fig['section'], 'print_mode'):
                from matplotlib.backends.backend_pdf import PdfPages
                fig['fig'] = plt
                fig['file'] = "{}/{}".format(self.figpath, fig['name'])
                fig['fig'].savefig("{}.pdf".format(fig['file']), format='pdf')
                self.compress_figure_to_PS(fig['file'])
                print("\tTimer: {:.2f} Second;  Figure {} saved as {}".format(time.time()-fig['start'], fig['name'], "{}/{}.pdf".format(self.figpath, fig['name'])))
            if 'show' in self.cf.get(fig['section'], 'print_mode'):
                plt.show()
        elif fig['type'] == "2D_Scatter":
            print("\n=== Ploting Figure : {} ===".format(fig['name']))
            fig['start'] = time.time()
            ax = fig['fig'].add_axes([0.13, 0.13, 0.83, 0.83])
            self.basic_selection(fig)
            print("\tTimer: {:.2f} Second;  Message from '{}' -> Data loading completed".format(time.time()-fig['start'], fig['section']))
            self.Get2DData(fig)
            # print(fig['var']['data'])
            fig['var']['lim'] = {
                'x': [fig['var']['data'].x.min(), fig['var']['data'].x.max()],
                'y': [fig['var']['data'].y.min(), fig['var']['data'].y.max()],
            }
            fig['ax'] = {}
            self.ax_setlim(fig, 'xy')
            self.ax_setcmap(fig)
            # print(fig['colorset'])
            if self.cf.has_option(fig['section'], 'marker'):
                marker = self.cf.get(fig['section'], 'marker').split(',')
                fig['ax']['markercolor'] = marker[0].strip()
                fig['ax']['markertype'] = marker[1].strip()
            else:
                fig['ax']['markercolor'] = 'Blue'
                fig['ax']['markertype'] = 'round'

            ax.scatter(
                    fig['var']['data'].x, 
                    fig['var']['data'].y, 
                    marker= fig['colorset']['scattermarker'][fig['ax']['markertype']],
                    c=      fig['colorset']['scattercolor'][fig['ax']['markercolor']],
                    s=      fig['colorset']['marker']['size'],
                    alpha=  fig['colorset']['marker']['alpha']
                )
            self.ax_setticks(fig, 'xy')

            ax.set_xticks(fig['ax']['ticks']['x'])
            ax.set_yticks(fig['ax']['ticks']['y'])
            ax.set_xlim(fig['ax']['lim']['x'][0], fig['ax']['lim']['x'][1])
            ax.set_ylim(fig['ax']['lim']['y'][0], fig['ax']['lim']['y'][1])
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.tick_params(
                labelsize=fig['colorset']['ticks']['labelsize'], 
                direction=fig['colorset']['ticks']['direction'], 
                bottom=fig['colorset']['ticks']['bottom'], 
                left=fig['colorset']['ticks']['left'], 
                top=fig['colorset']['ticks']['top'], 
                right=fig['colorset']['ticks']['right'],
                which='both'
            )
            ax.tick_params(which='major', length=fig['colorset']['ticks']['majorlength'], color=fig['colorset']['ticks']['majorcolor'])
            ax.tick_params(which='minor', length=fig['colorset']['ticks']['minorlength'], color=fig['colorset']['ticks']['minorcolor'])
            ax.set_xlabel(r"{}".format(self.cf.get(fig['section'], 'x_label')), fontsize=30)
            ax.set_ylabel(r"{}".format(self.cf.get(fig['section'], 'y_label')), fontsize=30)
            ax.xaxis.set_label_coords(0.5, -0.068)
            
            if 'save' in self.cf.get(fig['section'], 'print_mode'):
                from matplotlib.backends.backend_pdf import PdfPages
                fig['fig'] = plt
                fig['file'] = "{}/{}".format(self.figpath, fig['name'])
                fig['fig'].savefig("{}.pdf".format(fig['file']), format='pdf')
                # self.compress_figure_to_PS(fig['file'])
                print("\tTimer: {:.2f} Second;  Figure {} saved as {}".format(time.time()-fig['start'], fig['name'], "{}/{}.pdf".format(self.figpath, fig['name'])))
            if 'show' in self.cf.get(fig['section'], 'print_mode'):
                plt.show()

        

    def compress_figure_to_PS(self, figpath):
        os.system('pdf2ps {}.pdf {}.ps'.format(figpath, figpath))


    def ax_setcmap(self, fig):
        if self.cf.has_option(fig['section'], 'colorset'):
            cname = self.cf.get(fig['section'], 'colorset')
            if cname[0] == '&':
                for ii in self.colors:
                    if cname[1:] == ii['name']:
                        fig['colorset'] = ii 
                        break

    def ax_setlim(self, fig, label):
        fig['ax']['lim'] = {}
        for aa in label:
            tem = self.cf.get(fig['section'], '{}_lim'.format(aa)).split(',')
            fig['ax']['lim'][aa] = tem
            for it in tem:
                if ('AUTO' in it) and (aa in fig['var']['lim'].keys()):
                    fig['ax']['lim'][aa][tem.index(it)] = float(it.split('_')[1].strip()) * (fig['var']['lim'][aa][1] // float(it.split('_')[1].strip()) + 1)
                else:
                    fig['ax']['lim'][aa][tem.index(it)] = float(it.strip())
            
    def ax_setticks(self, fig, axislabel):
        fig['ax']['ticks'] = {}
        for aa in axislabel:
            tick = self.cf.get(fig['section'], '{}_ticks'.format(aa))
            if 'AUTO' in tick:
                a = float(tick.split('_')[-1])
                low = fig['ax']['lim'][aa][0] //a
                upp = fig['ax']['lim'][aa][1] //a + 1
                fig['ax']['ticks'][aa] = np.linspace(low*a, upp*a, upp-low+1)
            elif tick[0:4]=='Manu':
                p_rec = re.compile(r'[[].*?[]]', re.S)
                a = re.findall(p_rec, tick[5:].strip())
                tk = []
                label = []
                for it in a:
                    it = it.strip().strip('[').strip(']')
                    tk.append(float(it.split(',')[0]))
                    label.append(it.split(',')[1].strip())
                fig['ax']['ticks'][aa] = tuple([tk, label])
            else:
                tick = tick.split(',')
                for ii in range(len(tick)):
                    tick[ii] = float(tick[ii].strip())
                fig['ax']['ticks'][aa] = np.linspace(tick[0], tick[1], tick[2])
            if aa == 'y':
                if fig['ax']['lim']['y'][0] in fig['ax']['ticks']['y']:
                    fig['ax']['ticks']['y'] = fig['ax']['ticks']['y'][np.where(fig['ax']['ticks']['y'] != fig['ax']['lim']['y'][0])]


    def GetStatData(self, fig):
        if self.cf.has_option(fig['section'], 'stat_variable'):
            fig['var'] = {}
            if self.cf.get(fig['section'], 'stat_variable').split(',')[0] == 'CHI2':
                self.get_variable_data(fig, 'x', self.cf.get(fig['section'], 'x_variable'))
                self.get_variable_data(fig, 'y', self.cf.get(fig['section'], 'y_variable'))
                self.get_variable_data(fig, 'Stat', ",".join(self.cf.get(fig['section'], 'stat_variable').split(',')[1:]).strip())
                fig['var']['data'] = pd.DataFrame({
                    'x':    fig['var']['x'], 
                    'y':    fig['var']['y'], 
                    'Stat': fig['var']['Stat']})
                fig['var'].pop('x')
                fig['var'].pop('y')
                fig['var'].pop('Stat')

    def Get2DData(self, fig):
        fig['var'] = {}
        self.get_variable_data(fig, 'x', self.cf.get(fig['section'], 'x_variable'))
        self.get_variable_data(fig, 'y', self.cf.get(fig['section'], 'y_variable'))
        fig['var']['data'] = pd.DataFrame({
            'x':    fig['var']['x'],
            'y':    fig['var']['y']
        })


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
        self.figpath = "{}{}".format(self.cf.get('PLOT_CONFI', 'path'), self.cf.get('PLOT_CONFI', 'save_dir'))
        if not os.path.exists(self.figpath):
            os.makedirs(self.figpath)
        if self.cf.has_option('PLOT_CONFI', 'plot'):
            ppt = self.cf.get('PLOT_CONFI', 'plot').split('\n')
            self.figs = []
            for line in ppt:
                pic = {}
                pic['type'] = line.split(',')[0]
                if "ALL" in ','.join(line.split(',')[1:]).upper():
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




