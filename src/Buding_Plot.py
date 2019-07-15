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
import time

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
        if fig['type'] == "2D_Stat_Profile":
            print("\n=== Ploting Figure : {} ===".format(fig['name']))
            fig['start'] = time.time()
            ax = fig['fig'].add_axes([0.16, 0.16, 0.68, 0.81])
            axc = fig['fig'].add_axes([0.845, 0.16, 0.015, 0.81])
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
            fig['var']['data'] = fig['var']['data'].assign( Delta_Chi2 = fig['var']['data'].Stat - fig['var']['BestPoint']['Stat'] )
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
                'PL':   fig['ax']['grid'].apply(lambda tt: fig['var']['data'][ (fig['var']['data'].x > tt['xx'] - 0.5*fig['ax']['var']['dx']) & (fig['var']['data'].x < tt['xx'] + 0.5*fig['ax']['var']['dx']) & (fig['var']['data'].y > tt['yy'] - 0.5*fig['ax']['var']['dy']) & (fig['var']['data'].y < tt['yy'] + 0.5*fig['ax']['var']['dy']) ].PL.max(axis=0, skipna=True), axis=1),
                'DeltaChi2': fig['ax']['grid'].apply(lambda tt: fig['var']['data'][ (fig['var']['data'].x > tt['xx'] - 0.5*fig['ax']['var']['dx']) & (fig['var']['data'].x < tt['xx'] + 0.5*fig['ax']['var']['dx']) & (fig['var']['data'].y > tt['yy'] - 0.5*fig['ax']['var']['dy']) & (fig['var']['data'].y < tt['yy'] + 0.5*fig['ax']['var']['dy']) ].Delta_Chi2.min(axis=0, skipna=True), axis=1)
            }).fillna({'PL': 0., "DeltaChi2": 18.6})
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
            fig['ax']['tri_refine_PL'], fig['ax']['PL_refine']          = fig['ax']['refiner'].refine_field(fig['ax']['grid']['PL'])
            fig['ax']['tri_refine_DC'], fig['ax']['DeltaChi2_refine']   = fig['ax']['refiner'].refine_field(fig['ax']['grid']['DeltaChi2'])
            fig['ax']['PL_refine'] = ( fig['ax']['PL_refine'] > 0.) * fig['ax']['PL_refine'] / (np.max(fig['ax']['PL_refine'])) + 0. * ( fig['ax']['PL_refine'] > 0.)
            print("\tTimer: {:.2f} Second;  Message from '{}' -> Data analysis completed".format(time.time()-fig['start'], fig['section']))

            levels = MaxNLocator(nbins=50).tick_values(fig['ax']['lim']['c'][0], fig['ax']['lim']['c'][1])
            self.ax_setcmap(fig)

            a1 = ax.tricontourf(fig['ax']['tri_refine_PL'], fig['ax']['PL_refine'], cmap=fig['cmap'], levels=levels, zorder=1)
            plt.colorbar(a1, axc, ticks=fig['ax']['ticks']['c'], orientation='vertical', extend='neither')
            if self.cf.has_option(fig['section'], 'sign_curve'):
                fig['ax']['curve'] = eval(self.cf.get(fig['section'], 'sign_curve'))
                for curve in fig['ax']['curve']:
                    ct = ax.tricontour(fig['ax']['tri_refine_DC'], fig['ax']['DeltaChi2_refine'], [curve[0]], colors=curve[3], linewidths=curve[4])
                    # plt.clabel(ct, inline=True, fmt=r"{}".format(curve[2].replace('\\sigma', '\sigma') ), fontsize=8*curve[4], inline_spacing=2, use_clabeltext=True)

            ax.set_xlim(fig['ax']['lim']['x'][0], fig['ax']['lim']['x'][1])
            ax.set_ylim(fig['ax']['lim']['y'][0], fig['ax']['lim']['y'][1])
            ax.tick_params(labelsize=20, direction='in', bottom=True, left=True, top=True, right=True, which='both', color='w')
            axc.tick_params(labelsize=20, direction='in', bottom=True, left=False, top=True, right=False, which='both', color='w')
            ax.set_xlabel(r"{}".format(self.cf.get(fig['section'], 'x_label')), fontsize=30)
            ax.set_ylabel(r"{}".format(self.cf.get(fig['section'], 'y_label')), fontsize=30)
            axc.set_ylabel(r"{}".format(self.cf.get(fig['section'], 'c_label')), fontsize=30)
            if self.cf.has_option(fig['section'], 'BestPoint'):
                if eval(self.cf.get(fig['section'], 'BestPoint')):
                    ax.scatter(fig['var']['BestPoint']['x'], fig['var']['BestPoint']['y'], 300, marker='*', color='Tomato', zorder=20)
                    ax.scatter(fig['var']['BestPoint']['x'], fig['var']['BestPoint']['y'], 50, marker='*', color='w', zorder=21)
            if 'save' in self.cf.get(fig['section'], 'print_mode'):
                fig['fig'] = plt
                fig['fig'].savefig("{}/{}.pdf".format(self.figpath, fig['name']))
                print("\tTimer: {:.2f} Second;  Figure {} saved as {}".format(time.time()-fig['start'], fig['name'], "{}/{}.pdf".format(self.figpath, fig['name'])))
            if 'show' in self.cf.get(fig['section'], 'print_mode'):
                plt.show()


    def ax_setcmap(self, fig):
        if self.cf.has_option(fig['section'], 'c_cmap'):
            fig['cmap'] = self.cf.get(fig['section'], 'c_cmap')
            if '&' == fig['cmap'][0]:
                for cs in self.colors:
                    if cs['name'] == fig['cmap'][1:]:
                        fig['cmap'] = cs['cmap']
                        break
            else:
                fig['cmap'] = plt.get_cmap(fig['cmap'])

    def ax_setlim(self, fig, label):
        fig['ax']['lim'] = {}
        for aa in label:
            tem = self.cf.get(fig['section'], f'{aa}_lim').split(',')
            fig['ax']['lim'][aa] = tem
            for it in tem:
                if ('AUTO' in it) and (aa in fig['var']['lim'].keys()):
                    fig['ax']['lim'][aa][tem.index(it)] = float(it.split('_')[1].strip()) * (fig['var']['lim'][aa][1] // float(it.split('_')[1].strip()) + 1)
                else:
                    fig['ax']['lim'][aa][tem.index(it)] = float(it.strip())
            
    def ax_setticks(self, fig, axislabel):
        fig['ax']['ticks'] = {}
        for aa in axislabel:
            tick = self.cf.get(fig['section'], f'{aa}_ticks')
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
        self.figpath = "{}{}".format(self.cf.get('PLOT_CONFI', 'path'), self.cf.get('PLOT_INFO', 'save_dir'))
        if not os.path.exists(self.figpath):
            os.makedirs(self.figpath)
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




