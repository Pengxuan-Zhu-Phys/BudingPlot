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
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, FixedLocator, NullLocator, AutoMinorLocator)
import re 
import subprocess


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
            with open("{}{}".format(self.cf.get('PLOT_CONFI', 'path'), self.cf.get('COLORMAP', 'colorsetting')), 'r', encoding='utf-8') as f1:
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
            if self.cf.get(fig['section'], 'x_scale').strip().lower() == "flat" and self.cf.get(fig['section'], 'y_scale').strip().lower() == 'flat':
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
                }).fillna({'PL': -0.1})
            elif self.cf.get(fig['section'], 'x_scale').strip().lower() == "log" and self.cf.get(fig['section'], 'y_scale').strip().lower() == 'flat':
                XI, YI = np.meshgrid(
                    np.logspace(math.log10(fig['ax']['lim']['x'][0]), math.log10(fig['ax']['lim']['x'][1]), int(self.cf.get(fig['section'], 'x_nbin'))+1),
                    np.linspace(fig['ax']['lim']['y'][0], fig['ax']['lim']['y'][1], int(self.cf.get(fig['section'], 'y_nbin'))+1)                     
                )
                fig['ax']['var'] = {
                    'x':    XI,
                    'y':    YI,
                    'dx':   (math.log10(fig['ax']['lim']['x'][1]) - math.log10(fig['ax']['lim']['x'][0]))/int(self.cf.get(fig['section'], 'x_nbin')),
                    'dy':   (fig['ax']['lim']['y'][1] - fig['ax']['lim']['y'][0])/int(self.cf.get(fig['section'], 'y_nbin'))
                }
                fig['ax']['grid'] = pd.DataFrame(
                    index=np.logspace(math.log10(fig['ax']['lim']['x'][0]), math.log10(fig['ax']['lim']['x'][1]), int(self.cf.get(fig['section'], 'x_nbin'))+1),
                    columns=np.linspace(fig['ax']['lim']['y'][0], fig['ax']['lim']['y'][1], int(self.cf.get(fig['section'], 'y_nbin'))+1)
                ).unstack().reset_index().rename(columns={'level_0':'yy','level_1':'xx',0:'z'})
                fig['ax']['grid'] = pd.DataFrame({
                    "xi":   fig['ax']['grid']['xx'],
                    'yi':   fig['ax']['grid']['yy'],
                    'PL':   fig['ax']['grid'].apply(lambda tt: fig['var']['data'][ (fig['var']['data'].x > 10**(math.log10(tt['xx']) - 0.5*fig['ax']['var']['dx'])) & (fig['var']['data'].x < 10**(math.log10(tt['xx']) + 0.5*fig['ax']['var']['dx'])) & (fig['var']['data'].y > tt['yy'] - 0.5*fig['ax']['var']['dy']) & (fig['var']['data'].y < tt['yy'] + 0.5*fig['ax']['var']['dy']) ].PL.max(axis=0, skipna=True), axis=1)
                }).fillna({'PL': -0.1})
            elif self.cf.get(fig['section'], 'x_scale').strip().lower() == "flat" and self.cf.get(fig['section'], 'y_scale').strip().lower() == 'log':
                XI, YI = np.meshgrid(
                    np.linspace(fig['ax']['lim']['x'][0], fig['ax']['lim']['x'][1], int(self.cf.get(fig['section'], 'x_nbin'))+1),
                    np.logspace(math.log10(fig['ax']['lim']['y'][0]), math.log(fig['ax']['lim']['y'][1]), int(self.cf.get(fig['section'], 'y_nbin'))+1)                                       
                )    
                fig['ax']['var'] = {
                    'x':    XI,
                    'y':    YI,
                    'dx':   (fig['ax']['lim']['x'][1] - fig['ax']['lim']['x'][0])/int(self.cf.get(fig['section'], 'x_nbin')),
                    'dy':   (math.log10(fig['ax']['lim']['y'][1]) - math.log10(fig['ax']['lim']['y'][0]))/int(self.cf.get(fig['section'], 'y_nbin'))
                }
                fig['ax']['grid'] = pd.DataFrame(
                    index=np.linspace(fig['ax']['lim']['x'][0], fig['ax']['lim']['x'][1], int(self.cf.get(fig['section'], 'x_nbin'))+1),
                    columns=np.logspace(math.log10(fig['ax']['lim']['y'][0]), math.log10(fig['ax']['lim']['y'][1]), int(self.cf.get(fig['section'], 'y_nbin'))+1)
                ).unstack().reset_index().rename(columns={'level_0':'yy','level_1':'xx',0:'z'})
                fig['ax']['grid'] = pd.DataFrame({
                    "xi":   fig['ax']['grid']['xx'],
                    'yi':   fig['ax']['grid']['yy'],
                    'PL':   fig['ax']['grid'].apply(lambda tt: fig['var']['data'][ (fig['var']['data'].x > tt['xx'] - 0.5*fig['ax']['var']['dx']) & (fig['var']['data'].x < tt['xx'] + 0.5*fig['ax']['var']['dx']) & (fig['var']['data'].y > 10**(math.log10(tt['yy']) - 0.5*fig['ax']['var']['dy'])) & (fig['var']['data'].y <10**(math.log10(tt['yy']) + 0.5*fig['ax']['var']['dy'])) ].PL.max(axis=0, skipna=True), axis=1)
                }).fillna({'PL': -0.1})
            elif self.cf.get(fig['section'], 'x_scale').strip().lower() == "log" and self.cf.get(fig['section'], 'y_scale').strip().lower() == 'log':
                XI, YI = np.meshgrid(
                    np.logspace(math.log10(fig['ax']['lim']['x'][0]), math.log10(fig['ax']['lim']['x'][1]), int(self.cf.get(fig['section'], 'x_nbin'))+1),
                    np.logspace(math.log10(fig['ax']['lim']['y'][0]), math.log(fig['ax']['lim']['y'][1]), int(self.cf.get(fig['section'], 'y_nbin'))+1)                                     
                )
                fig['ax']['var'] = {
                    'x':    XI,
                    'y':    YI,
                    'dx':   (math.log10(fig['ax']['lim']['x'][1]) - math.log10(fig['ax']['lim']['x'][0]))/int(self.cf.get(fig['section'], 'x_nbin')),
                    'dy':   (math.log10(fig['ax']['lim']['y'][1]) - math.log10(fig['ax']['lim']['y'][0]))/int(self.cf.get(fig['section'], 'y_nbin'))
                }
                fig['ax']['grid'] = pd.DataFrame(
                    index=np.logspace(math.log10(fig['ax']['lim']['x'][0]), math.log10(fig['ax']['lim']['x'][1]), int(self.cf.get(fig['section'], 'x_nbin'))+1),
                    columns=np.logspace(math.log10(fig['ax']['lim']['y'][0]), math.log10(fig['ax']['lim']['y'][1]), int(self.cf.get(fig['section'], 'y_nbin'))+1)
                ).unstack().reset_index().rename(columns={'level_0':'yy','level_1':'xx',0:'z'})
                fig['ax']['grid'] = pd.DataFrame({
                    "xi":   fig['ax']['grid']['xx'],
                    'yi':   fig['ax']['grid']['yy'],
                    'PL':   fig['ax']['grid'].apply(lambda tt: fig['var']['data'][ (fig['var']['data'].x > 10**(math.log10(tt['xx']) - 0.5*fig['ax']['var']['dx'])) & (fig['var']['data'].x < 10**(math.log10(tt['xx']) + 0.5*fig['ax']['var']['dx'])) & (fig['var']['data'].y > 10**(math.log10(tt['yy']) - 0.5*fig['ax']['var']['dy'])) & (fig['var']['data'].y <10**(math.log10(tt['yy']) + 0.5*fig['ax']['var']['dy'])) ].PL.max(axis=0, skipna=True), axis=1)
                }).fillna({'PL': -0.1})
            else:
                print("No such Mode For X,Y scale -> {},{}".format(self.cf.get(fig['section'], 'x_scale'), self.cf.get(fig['section'], 'y_scale')))
                sys.exit(0)
            XI, YI = np.meshgrid(
                np.linspace(0., 1., int(self.cf.get(fig['section'], 'x_nbin'))+1),
                np.linspace(0., 1., int(self.cf.get(fig['section'], 'y_nbin'))+1)                
            )
            fig['ax']['mashgrid'] = pd.DataFrame(
                index = np.linspace(0., 1., int(self.cf.get(fig['section'], 'x_nbin'))+1),
                columns = np.linspace(0., 1., int(self.cf.get(fig['section'], 'y_nbin'))+1) 
            ).unstack().reset_index().rename(columns={'level_0':'yy','level_1':'xx',0:'z'})
            fig['ax']['mashgrid'] = pd.DataFrame({
                'x':  fig['ax']['mashgrid']['xx'],
                'y':  fig['ax']['mashgrid']['yy'],
                'PL':   fig['ax']['grid']['PL']
            })
            self.ax_setticks(fig, 'xyc')

            from matplotlib.ticker import MaxNLocator
            levels = MaxNLocator(nbins=100).tick_values(fig['ax']['lim']['c'][0], fig['ax']['lim']['c'][1])
            self.ax_setcmap(fig)
            from matplotlib.tri import Triangulation, TriAnalyzer, UniformTriRefiner
            fig['ax']['tri'] = Triangulation(fig['ax']['mashgrid']['x'], fig['ax']['mashgrid']['y'])
            fig['ax']['refiner'] = UniformTriRefiner(fig['ax']['tri'])
            fig['ax']['tri_refine_PL'], fig['ax']['PL_refine'] = fig['ax']['refiner'].refine_field(fig['ax']['mashgrid']['PL'], subdiv=3)
            fig['ax']['PL_refine'] = ( fig['ax']['PL_refine'] > 0.) * fig['ax']['PL_refine'] / (np.max(fig['ax']['PL_refine'])) + 0. * ( fig['ax']['PL_refine'] < 0.)

            print("\tTimer: {:.2f} Second;  Message from '{}' -> Data analysis completed".format(time.time()-fig['start'], fig['section']))

            a1 = ax.tricontourf(fig['ax']['tri_refine_PL'], fig['ax']['PL_refine'], cmap=fig['colorset']['colormap'], levels=levels, zorder=1, transform=ax.transAxes)
            plt.colorbar(a1, axc, ticks=fig['ax']['ticks']['c'], orientation='vertical', extend='neither')
            if 'curve' in fig['colorset'].keys():
                for curve in fig['colorset']['curve']:
                    ct = ax.tricontour(fig['ax']['tri_refine_PL'], fig['ax']['PL_refine'], [math.exp(-0.5 * curve['value'])], colors=curve['linecolor'], linewidths=curve['linewidth'], zorder=(10-curve['tag'])*4, transform=ax.transAxes)

                

            if self.cf.has_option(fig['section'], 'BestPoint'):
                if eval(self.cf.get(fig['section'], 'BestPoint')) and 'bestpoint' in fig['colorset'].keys():
                    ax.scatter(fig['var']['BestPoint']['x'], fig['var']['BestPoint']['y'], 300, marker='*', color=fig['colorset']['bestpoint'][0], zorder=2000)
                    ax.scatter(fig['var']['BestPoint']['x'], fig['var']['BestPoint']['y'], 50, marker='*', color=fig['colorset']['bestpoint'][1], zorder=2100)
                    # print(fig['data'][fig['data']["chi2_h1"] == fig['var']['BestPoint']['Stat']])

            if self.cf.get(fig['section'], 'x_scale').strip().lower() == "flat":
                if not self.cf.get(fig['section'], 'x_ticks')[0:4] == 'Manu':
                    ax.set_xticks(fig['ax']['ticks']['x'])
                    ax.xaxis.set_minor_locator(AutoMinorLocator())
            elif self.cf.get(fig['section'], 'x_scale').strip().lower() == "log":
                ax.set_xscale('log')
            if self.cf.get(fig['section'], 'x_ticks')[0:4] == 'Manu':
                ax.xaxis.set_major_locator(ticker.FixedLocator(fig['ax']['ticks']['x'][0]))
                ax.set_xticklabels(fig['ax']['ticks']['x'][1])
                if self.cf.get(fig['section'], 'x_scale').strip().lower() == "flat":
                    ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.set_xlim(fig['ax']['lim']['x'][0], fig['ax']['lim']['x'][1])

            if self.cf.get(fig['section'], 'y_scale').strip().lower() == 'flat':
                if not self.cf.get(fig['section'], 'y_ticks')[0:4] == "Manu":
                    ax.set_yticks(fig['ax']['ticks']['y'])
                    ax.yaxis.set_minor_locator(AutoMinorLocator())
            elif self.cf.get(fig['section'], 'y_scale').strip().lower() == 'log':
                ax.set_yscale('log')
            if self.cf.get(fig['section'], 'y_ticks')[0:4] == "Manu":
                ax.yaxis.set_major_locator(ticker.FixedLocator(fig['ax']['ticks']['y'][0]))
                ax.set_yticklabels(fig['ax']['ticks']['y'][1])
                if self.cf.get(fig['section'], 'y_scale').strip().lower() == 'flat':
                    ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.set_ylim(fig['ax']['lim']['y'][0], fig['ax']['lim']['y'][1])

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

            if self.cf.has_option(fig['section'], 'Line_draw'):
                self.drawline(fig, ax)
            
            if self.cf.has_option(fig['section'], "Text"):
                self.drawtext(fig, ax)

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
                'y': [fig['var']['data'].y.min(), fig['var']['data'].y.max()]
            }
            fig['ax'] = {}
            self.ax_setlim(fig, 'xy')
            self.ax_setcmap(fig)
            # print(fig['colorset'])
            if self.cf.has_option(fig['section'], 'marker'):
                lines = self.cf.get(fig['section'], 'marker').split('\n')
                if len(lines) == 1:
                    marker = lines[0].split(',')
                    if len(marker) == 2:
                        fig['ax']['markercolor'] = marker[0].strip()
                        fig['ax']['markertype'] = marker[1].strip()
                        ax.scatter(
                                fig['var']['data'].x, 
                                fig['var']['data'].y, 
                                marker= fig['colorset']['scattermarker'][fig['ax']['markertype']],
                                c=      fig['colorset']['scattercolor'][fig['ax']['markercolor']],
                                s=      fig['colorset']['marker']['size'],
                                alpha=  fig['colorset']['marker']['alpha']
                            )
                    elif len(marker) > 2:
                        if marker[2].strip()[0:3] == "&Bo":
                            fig['ax']['markercolor'] = marker[0].strip()
                            fig['ax']['markertype'] = marker[1].strip()
                            self.scatter_classify_data(fig, marker[2].strip()[4:])
                            ax.scatter(
                                fig['classify']['x'],
                                fig['classify']['y'],
                                marker= fig['colorset']['scattermarker'][fig['ax']['markertype']],
                                c=      fig['colorset']['scattercolor'][fig['ax']['markercolor']],
                                s=      fig['colorset']['marker']['size'],
                                alpha=  fig['colorset']['marker']['alpha']
                            )
                else:
                    for line in lines:
                        marker = line.split(',')
                        if len(marker) > 2:
                            fig['ax']['markercolor'] = marker[0].strip()
                            fig['ax']['markertype'] = marker[1].strip()
                            self.scatter_classify_data(fig, marker[2].strip()[4:])
                            ax.scatter(
                                fig['classify']['x'],
                                fig['classify']['y'],
                                marker= fig['colorset']['scattermarker'][fig['ax']['markertype']],
                                c=      fig['colorset']['scattercolor'][fig['ax']['markercolor']],
                                s=      fig['colorset']['marker']['size'],
                                alpha=  fig['colorset']['marker']['alpha']
                            )
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

            if self.cf.get(fig['section'], 'x_scale').strip().lower() == "flat":
                if not self.cf.get(fig['section'], 'x_ticks')[0:4] == 'Manu':
                    ax.set_xticks(fig['ax']['ticks']['x'])
                    ax.xaxis.set_minor_locator(AutoMinorLocator())
            elif self.cf.get(fig['section'], 'x_scale').strip().lower() == "log":
                ax.set_xscale('log')
            if self.cf.get(fig['section'], 'x_ticks')[0:4] == 'Manu':
                ax.xaxis.set_major_locator(ticker.FixedLocator(fig['ax']['ticks']['x'][0]))
                ax.set_xticklabels(fig['ax']['ticks']['x'][1])
                if self.cf.get(fig['section'], 'x_scale').strip().lower() == "flat":
                    ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.set_xlim(fig['ax']['lim']['x'][0], fig['ax']['lim']['x'][1])

            if self.cf.get(fig['section'], 'y_scale').strip().lower() == 'flat':
                if not self.cf.get(fig['section'], 'y_ticks')[0:4] == "Manu":
                    ax.set_yticks(fig['ax']['ticks']['y'])
                    ax.yaxis.set_minor_locator(AutoMinorLocator())
            elif self.cf.get(fig['section'], 'y_scale').strip().lower() == 'log':
                ax.set_yscale('log')
            if self.cf.get(fig['section'], 'y_ticks')[0:4] == "Manu":
                ax.yaxis.set_major_locator(ticker.FixedLocator(fig['ax']['ticks']['y'][0]))
                ax.set_yticklabels(fig['ax']['ticks']['y'][1])
                if self.cf.get(fig['section'], 'y_scale').strip().lower() == 'flat':
                    ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.set_ylim(fig['ax']['lim']['y'][0], fig['ax']['lim']['y'][1])

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
            
            if self.cf.has_option(fig['section'], 'Line_draw'):
                self.drawline(fig, ax)
            
            if self.cf.has_option(fig['section'], "Text"):
                self.drawtext(fig, ax)

            if 'save' in self.cf.get(fig['section'], 'print_mode'):
                from matplotlib.backends.backend_pdf import PdfPages
                fig['fig'] = plt
                fig['file'] = "{}/{}".format(self.figpath, fig['name'])
                fig['fig'].savefig("{}.pdf".format(fig['file']), format='pdf')
                # self.compress_figure_to_PS(fig['file'])
                print("\tTimer: {:.2f} Second;  Figure {} saved as {}".format(time.time()-fig['start'], fig['name'], "{}/{}.pdf".format(self.figpath, fig['name'])))
            if 'show' in self.cf.get(fig['section'], 'print_mode'):
                plt.show()
        elif fig['type'] == "2DC_Scatter":
            print("\n=== Ploting Figure : {} ===".format(fig['name']))
            fig['start'] = time.time()
            ax = fig['fig'].add_axes([0.13, 0.13, 0.74, 0.83])
            axc = fig['fig'].add_axes([0.875, 0.13, 0.015, 0.83])
            self.basic_selection(fig)
            print("\tTimer: {:.2f} Second;  Message from '{}' -> Data loading completed".format(time.time()-fig['start'], fig['section']))
            self.Get3DData(fig)
            fig['var']['lim'] = {
                'x': [fig['var']['data'].x.min(), fig['var']['data'].x.max()],
                'y': [fig['var']['data'].y.min(), fig['var']['data'].y.max()],
                'c': [fig['var']['data'].c.min(), fig['var']['data'].c.max()]
            }
            fig['ax'] = {}
            self.ax_setlim(fig, 'xyc')
            self.ax_setcmap(fig)

            fig['ax']['a1'] = []
            if self.cf.has_option(fig['section'], 'marker'):
                lines = self.cf.get(fig['section'], 'marker').split('\n')
                for line in lines:
                    marker = line.split(',')
                    if marker[0].strip() == "&Color":
                        fig['ax']['markertype'] = marker[1].strip()
                        if len(marker) == 2:
                            fig['ax']['a1'].append(ax.scatter(
                                fig['var']['data'].x,
                                fig['var']['data'].y,
                                c       = fig['var']['data'].c,
                                marker  = fig['colorset']['scattermarker'][fig['ax']['markertype']],
                                s       = fig['colorset']['marker']['size'],
                                alpha   = fig['colorset']['marker']['alpha'],
                                cmap    = fig['colorset']['colormap'],
                                vmin    = fig['ax']['lim']['c'][0],
                                vmax    = fig['ax']['lim']['c'][1]
                            ))
                        elif len(marker) > 2:
                            if marker[2].strip()[0:3] == '&Bo':
                                self.scatter_classify_data(fig, marker[2].strip()[4:])
                                fig['ax']['a1'].append(ax.scatter(
                                    fig['classify']['x'],
                                    fig['classify']['y'],
                                    c       = fig['classify']['c'],
                                    marker  = fig['colorset']['scattermarker'][fig['ax']['markertype']],
                                    s       = fig['colorset']['marker']['size'],
                                    alpha   = fig['colorset']['marker']['alpha'],
                                    cmap    = fig['colorset']['colormap'],
                                    vmin    = fig['ax']['lim']['c'][0],
                                    vmax    = fig['ax']['lim']['c'][1]
                                ))
                    else:
                        fig['ax']['markercolor']    = marker[0].strip()
                        fig['ax']['markertype']     = marker[1].strip()
                        if len(marker) == 2:
                            ax.scatter(
                                fig['var']['data'].x,
                                fig['var']['data'].y,
                                marker= fig['colorset']['scattermarker'][fig['ax']['markertype']],
                                c=      fig['colorset']['scattercolor'][fig['ax']['markercolor']],
                                s=      fig['colorset']['marker']['size'],
                                alpha=  fig['colorset']['marker']['alpha']
                            )
                        elif len(marker) > 2:
                            if marker[2].strip()[0:3] == '&Bo':
                                self.scatter_classify_data(fig, marker[2].strip()[4:])
                                ax.scatter(
                                    fig['classify']['x'],
                                    fig['classify']['y'],
                                    marker= fig['colorset']['scattermarker'][fig['ax']['markertype']],
                                    c=      fig['colorset']['scattercolor'][fig['ax']['markercolor']],
                                    s=      fig['colorset']['marker']['size'],
                                    alpha=  fig['colorset']['marker']['alpha']
                                )  

            self.ax_setticks(fig, 'xyc')
            if self.cf.get(fig['section'], 'x_scale').strip().lower() == "flat":
                if not self.cf.get(fig['section'], 'x_ticks')[0:4] == 'Manu':
                    ax.set_xticks(fig['ax']['ticks']['x'])
                    ax.xaxis.set_minor_locator(AutoMinorLocator())
            elif self.cf.get(fig['section'], 'x_scale').strip().lower() == "log":
                ax.set_xscale('log')
            if self.cf.get(fig['section'], 'x_ticks')[0:4] == 'Manu':
                ax.xaxis.set_major_locator(ticker.FixedLocator(fig['ax']['ticks']['x'][0]))
                ax.set_xticklabels(fig['ax']['ticks']['x'][1])
                if self.cf.get(fig['section'], 'x_scale').strip().lower() == "flat":
                    ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.set_xlim(fig['ax']['lim']['x'][0], fig['ax']['lim']['x'][1])

            if self.cf.get(fig['section'], 'y_scale').strip().lower() == 'flat':
                if not self.cf.get(fig['section'], 'y_ticks')[0:4] == "Manu":
                    ax.set_yticks(fig['ax']['ticks']['y'])
                    ax.yaxis.set_minor_locator(AutoMinorLocator())
            elif self.cf.get(fig['section'], 'y_scale').strip().lower() == 'log':
                ax.set_yscale('log')
            if self.cf.get(fig['section'], 'y_ticks')[0:4] == "Manu":
                ax.yaxis.set_major_locator(ticker.FixedLocator(fig['ax']['ticks']['y'][0]))
                ax.set_yticklabels(fig['ax']['ticks']['y'][1])
                if self.cf.get(fig['section'], 'y_scale').strip().lower() == 'flat':
                    ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.set_ylim(fig['ax']['lim']['y'][0], fig['ax']['lim']['y'][1])
                        
            
            if self.cf.has_option(fig['section'], 'c_scale'):
                if self.cf.get(fig['section'], 'c_scale').strip().lower() == 'log':
                    from matplotlib.colors import LogNorm
                    plt.colorbar(fig['ax']['a1'][0], axc, norm=LogNorm(fig['var']['lim']['c'][0], fig['var']['lim']['c'][1]), orientation='vertical', extend='neither')
                elif self.cf.get(fig['section'], 'c_scale').strip().lower() == "flat":
                    plt.colorbar(fig['ax']['a1'][0], axc, ticks=fig['ax']['ticks']['c'], orientation='vertical', extend='neither')
            # axc.set_ylim(fig['ax']['lim']['c'][0], fig['ax']['lim']['c'][1])
            

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
            # axc.tick_params(which='major', length=fig['colorset']['ticks']['majorlength'], color=fig['colorset']['ticks']['majorcolor'])

            ax.set_xlabel(r"{}".format(self.cf.get(fig['section'], 'x_label')), fontsize=30)
            ax.set_ylabel(r"{}".format(self.cf.get(fig['section'], 'y_label')), fontsize=30)
            axc.set_ylabel(r"{}".format(self.cf.get(fig['section'], 'c_label')), fontsize=30)
            ax.xaxis.set_label_coords(0.5, -0.068)

            if self.cf.has_option(fig['section'], 'Line_draw'):
                self.drawline(fig, ax)
            
            if self.cf.has_option(fig['section'], "Text"):
                self.drawtext(fig, ax)

            if 'save' in self.cf.get(fig['section'], 'print_mode'):
                from matplotlib.backends.backend_pdf import PdfPages
                fig['fig'] = plt
                fig['file'] = "{}/{}".format(self.figpath, fig['name'])
                fig['fig'].savefig("{}.pdf".format(fig['file']), format='pdf')
                self.compress_figure_to_PS(fig['file'])
                print("\tTimer: {:.2f} Second;  Figure {} saved as {}".format(time.time()-fig['start'], fig['name'], "{}/{}.pdf".format(self.figpath, fig['name'])))
            if 'show' in self.cf.get(fig['section'], 'print_mode'):
                plt.show()
        elif fig['type'] == "1D_Stat":
            print("\n=== Ploting Figure : {} ===".format(fig['name']))
            fig['start'] = time.time()
            self.basic_selection(fig)
            print("\tTimer: {:.2f} Second;  Message from '{}' -> Data loading completed".format(time.time()-fig['start'], fig['section']))
            self.Get1DStatData(fig)
            if 'x' in fig['var']['type'] and 'CHI2' in fig['var']['type'] and 'PDF' in fig['var']['type']:
                ax = fig['fig'].add_axes([0.13, 0.13, 0.74, 0.83])

                fig['var']['BestPoint'] = {
                    'x':    fig['var']['data'].loc[fig['var']['data'].CHI2.idxmin()].x,
                    'CHI2': fig['var']['data'].loc[fig['var']['data'].CHI2.idxmin()].CHI2, 
                    'PDF':  fig['var']['data'].loc[fig['var']['data'].CHI2.idxmin()].PDF
                }
                fig['var']['data'] = fig['var']['data'].assign( PL=lambda x: np.exp( -0.5 * x.CHI2 )/np.exp( -0.5 * fig['var']['BestPoint']['CHI2']) )
                fig['var']['lim'] = {
                    'x':    [fig['var']['data'].x.min(), fig['var']['data'].x.max()]
                }
                fig['ax'] = {}
                self.ax_setlim(fig, 'x')
                if self.cf.get(fig['section'], 'x_scale').strip().lower() == "flat":
                    XXGrid = np.linspace(0, 1, int(self.cf.get(fig['section'], 'x_nbin'))+1)
                    XI = np.linspace(fig['ax']['lim']['x'][0], fig['ax']['lim']['x'][1], int(self.cf.get(fig['section'], 'x_nbin'))+1)
                    fig['ax']['var'] = {
                        'x':    XI,
                        'dx':   (fig['ax']['lim']['x'][1] - fig['ax']['lim']['x'][0])/int(self.cf.get(fig['section'], 'x_nbin'))
                    }
                    fig['ax']['grid'] = pd.DataFrame({
                        "xx":   XI
                    })
                    fig['ax']['grid'] = pd.DataFrame({
                        "xi":   fig['ax']['grid']['xx'],
                        "xxgrid":   XXGrid,
                        'PL':   fig['ax']['grid'].apply(lambda tt: fig['var']['data'][ (fig['var']['data'].x >= tt['xx'] -0.5*fig['ax']['var']['dx']) & (fig['var']['data'].x < tt['xx'] + 0.5*fig['ax']['var']['dx']) ].PL.max(axis=0, skipna=True), axis=1),
                        'PDF':  fig['ax']['grid'].apply(lambda tt: fig['var']['data'][ (fig['var']['data'].x >= tt['xx'] -0.5*fig['ax']['var']['dx']) & (fig['var']['data'].x < tt['xx'] + 0.5*fig['ax']['var']['dx']) ].PDF.sum(), axis=1)
                    }).fillna({"PL": 0.0, "PDF": 0.0})
                    from scipy.stats import gaussian_kde 
                    # pdfkde = gaussian_kde(fig['ax']['grid'].xi, bw_method=0.03*fig['ax']['var']['dx'], weights=fig['ax']['grid'].PDF)
                    if self.cf.has_option(fig['section'], 'pdf_kde_bw'):
                        fig['ax']['pdf_kde_bw'] = float(self.cf.get(fig['section'], 'pdf_kde_bw'))
                    else:
                        fig['ax']['pdf_kde_bw'] = 'silverman'
                    fig["ax"]["pdfkde"] = gaussian_kde(fig['ax']['grid'].xxgrid, bw_method=fig['ax']['pdf_kde_bw'], weights=fig['ax']['grid'].PDF)
                    xgrid = np.linspace(0, 1, 10000)
                    fig['ax']['pdfkdedata'] = pd.DataFrame({
                        'xx':   np.linspace(fig['ax']['lim']['x'][0], fig['ax']['lim']['x'][1], 10000),
                        'pdf':  fig["ax"]["pdfkde"].evaluate(xgrid),
                    })
                    fig['var']['BestPoint']['PDF'] = fig['ax']['pdfkde'].evaluate(fig['var']['BestPoint']['x'])/fig['ax']['pdfkdedata']['pdf'].max()
                    fig['ax']['pdfkdedata']['pdf'] = fig['ax']['pdfkdedata']['pdf']/fig['ax']['pdfkdedata']['pdf'].max()
                    fig['ax']['pdfpara'] = {
                        'norm':                     sum(fig['ax']['pdfkdedata'].pdf),
                        '1sigma_critical_prob':     self.find_critical_prob(fig['ax']['pdfkdedata'], 0.675),
                        '2sigma_critical_prob':     self.find_critical_prob(fig['ax']['pdfkdedata'], 0.95),
                        'mode':                     fig['ax']['pdfkdedata'].iloc[fig['ax']['pdfkdedata'].pdf.idxmax()].xx
                    }
                    axpl = interp1d(fig['ax']['grid']['xi'], fig['ax']['grid']['PL'], kind='linear')
                    plgrid = axpl(np.linspace(fig['ax']['lim']['x'][0], fig['ax']['lim']['x'][1], 10000))
                elif self.cf.get(fig['section'], 'x_scale').strip().lower() == "log":
                    XXGrid = np.linspace(0, 1, int(self.cf.get(fig['section'], 'x_nbin'))+1)
                    XI = np.logspace(math.log10(fig['ax']['lim']['x'][0]), math.log10(fig['ax']['lim']['x'][1]), int(self.cf.get(fig['section'], 'x_nbin'))+1)
                    fig['ax']['var'] = {
                        'x':    XI,
                        'dx':   (math.log10(fig['ax']['lim']['x'][1]) - math.log10(fig['ax']['lim']['x'][0]))/int(self.cf.get(fig['section'], 'x_nbin'))
                    }
                    fig['ax']['grid'] = pd.DataFrame({
                        "xx":   XI
                    })
                    fig['ax']['grid'] = pd.DataFrame({
                        "xi":   fig['ax']['grid']['xx'],
                        "xxgrid":   XXGrid,
                        'PL':   fig['ax']['grid'].apply(lambda tt: fig['var']['data'][ (fig['var']['data'].x >= 10**(math.log10(tt['xx']) -0.5*fig['ax']['var']['dx'])) & (fig['var']['data'].x < 10**(math.log10(tt['xx']) + 0.5*fig['ax']['var']['dx'])) ].PL.max(axis=0, skipna=True), axis=1),
                        'PDF':  fig['ax']['grid'].apply(lambda tt: fig['var']['data'][ (fig['var']['data'].x >= 10**(math.log10(tt['xx']) -0.5*fig['ax']['var']['dx'])) & (fig['var']['data'].x < 10**(math.log10(tt['xx']) + 0.5*fig['ax']['var']['dx'])) ].PDF.sum(), axis=1)
                    }).fillna({'PL': 0.0, 'PDF': 0.0})
                    # print(fig['ax']['grid'])
                    from scipy.stats import gaussian_kde 
                    # pdfkde = gaussian_kde(fig['ax']['grid'].xi, bw_method=0.03*fig['ax']['var']['dx'], weights=fig['ax']['grid'].PDF)
                    if self.cf.has_option(fig['section'], 'pdf_kde_bw'):
                        fig['ax']['pdf_kde_bw'] = float(self.cf.get(fig['section'], 'pdf_kde_bw'))
                    else:
                        fig['ax']['pdf_kde_bw'] = 'silverman'
                    fig["ax"]["pdfkde"] = gaussian_kde(fig['ax']['grid'].xxgrid, bw_method=fig['ax']['pdf_kde_bw'], weights=fig['ax']['grid'].PDF)
                    xgrid = np.linspace(0, 1, 10000)
                    fig['ax']['pdfkdedata'] = pd.DataFrame({
                        'xx':   np.logspace(math.log10(fig['ax']['lim']['x'][0]), math.log10(fig['ax']['lim']['x'][1]), 10000), 
                        'pdf':  fig["ax"]["pdfkde"].evaluate(xgrid),
                    })

                    fig['var']['BestPoint']['PDF'] = fig['ax']['pdfkde'].evaluate(fig['var']['BestPoint']['x'])/fig['ax']['pdfkdedata']['pdf'].max()
                    fig['ax']['pdfkdedata']['pdf'] = fig['ax']['pdfkdedata']['pdf']/fig['ax']['pdfkdedata']['pdf'].max()
                    fig['ax']['pdfpara'] = {
                        'norm':                     sum(fig['ax']['pdfkdedata'].pdf),
                        '1sigma_critical_prob':     self.find_critical_prob(fig['ax']['pdfkdedata'], 0.675),
                        '2sigma_critical_prob':     self.find_critical_prob(fig['ax']['pdfkdedata'], 0.95),
                        'mode':                     fig['ax']['pdfkdedata'].iloc[fig['ax']['pdfkdedata'].pdf.idxmax()].xx
                    }
                    axpl = interp1d(fig['ax']['grid']['xi'], fig['ax']['grid']['PL'], kind='linear')
                    plgrid = axpl(np.logspace(math.log10(fig['ax']['lim']['x'][0]), math.log10(fig['ax']['lim']['x'][1]), 10000))
                    ax.set_xscale("log")
                self.ax_setcmap(fig)
                ax.fill_between( fig['ax']['pdfkdedata']['xx'], -0.09, -0.12, where=fig['ax']['pdfkdedata']['pdf'] > fig['ax']['pdfpara']['2sigma_critical_prob'], edgecolor=fig['colorset']['1dpdf']['2sigma']['edge'], facecolor=fig['colorset']['1dpdf']['2sigma']['facecolor'],  alpha=fig['colorset']['1dpdf']['2sigma']['alpha'], zorder=5 )
                ax.fill_between( fig['ax']['pdfkdedata']['xx'], -0.09, -0.12, where=fig['ax']['pdfkdedata']['pdf'] > fig['ax']['pdfpara']['1sigma_critical_prob'], edgecolor=fig['colorset']['1dpdf']['1sigma']['edge'],facecolor=fig['colorset']['1dpdf']['1sigma']['facecolor'], alpha=fig['colorset']['1dpdf']['1sigma']['alpha'],zorder=6)                
                ax.fill_between( fig['ax']['pdfkdedata']['xx'], 0, fig['ax']['pdfkdedata']['pdf'],  where=fig['ax']['pdfkdedata']['pdf'] > fig['ax']['pdfpara']['2sigma_critical_prob'],  edgecolor=fig['colorset']['1dpdf']['2sigma']['edge'], facecolor=fig['colorset']['1dpdf']['2sigma']['facecolor'],  alpha=fig['colorset']['1dpdf']['2sigma']['alpha'], zorder=5 )
                ax.fill_between( fig['ax']['pdfkdedata']['xx'], 0, fig['ax']['pdfkdedata']['pdf'],  where=fig['ax']['pdfkdedata']['pdf'] > fig['ax']['pdfpara']['1sigma_critical_prob'],  edgecolor=fig['colorset']['1dpdf']['1sigma']['edge'], facecolor=fig['colorset']['1dpdf']['1sigma']['facecolor'],  alpha=fig['colorset']['1dpdf']['1sigma']['alpha'], zorder=6 )
                ax.set_xlim(fig['ax']['lim']['x'][0], fig['ax']['lim']['x'][1])
                ax.plot([fig['ax']['lim']['x'][0], fig['ax']['lim']['x'][1]], [0, 0], '-', color='grey', linewidth=0.8, zorder=0)
                ax.plot([fig['ax']['lim']['x'][0], fig['ax']['lim']['x'][1]], [0.1354, 0.1354], '--', color='#ea4702', linewidth=1.2, zorder=0)
                ax.text(0.025, 0.237, r"$95.4\%~{\rm C.L.}$", fontsize=12, transform=ax.transAxes)
                ax.plot([fig['ax']['lim']['x'][0], fig['ax']['lim']['x'][1]], [0.60583, 0.60583], '--', color='#ea4702', linewidth=1.2, zorder=0)
                ax.text(0.025, 0.595, r"$68.3\%~{\rm C.L.}$", fontsize=12, transform=ax.transAxes)
                ax.set_ylim(-0.16, 1.15)                 
                if self.cf.has_option(fig['section'], "BestPoint"):
                    for item in fig['colorset']['1dpdf']['bestpoint']:
                        ax.plot( [fig['var']['BestPoint']['x'], fig['var']['BestPoint']['x']], [-0.069, -0.041 ], '-', linewidth=item['width'], color=item['color'], alpha=item['alpha'], zorder=7               )   
                    for item in fig['colorset']['1dpdf']['pdfmode']:             
                        ax.plot( [fig['ax']['pdfpara']['mode'], fig['ax']['pdfpara']['mode']], [0.001, 0.999 ], '-',linewidth=item['width'],color=item['color'],alpha=item['alpha'],zorder=7)
                        ax.plot( [fig['ax']['pdfpara']['mode'], fig['ax']['pdfpara']['mode']], [-0.09, -0.12 ], '-',linewidth=item['width'],color=item['color'],alpha=item['alpha'],zorder=7 )
                ax.fill_between(fig['ax']['pdfkdedata']['xx'], 0, fig['ax']['pdfkdedata']['pdf'], color="w", alpha=fig['colorset']['1dpdf']['line']['alpha'],zorder=1)                
                ax.plot(fig['ax']['pdfkdedata']['xx'], fig['ax']['pdfkdedata']['pdf'], '-', linewidth = fig['colorset']['1dpdf']['line']['width'], color=fig['colorset']['1dpdf']['line']['color'], alpha=fig['colorset']['1dpdf']['line']['alpha'],zorder=10)
                ax.fill_between( fig['ax']['grid']['xi'], 0, fig['ax']['grid']['PL'], color='red', step='mid', alpha=0.1, zorder=0 )
                ax.step(fig['ax']['grid']['xi'],fig['ax']['grid']['PL'],color='red',where='mid',zorder=9)  
                # axpl = interp1d(fig['ax']['grid']['xi'], fig['ax']['grid']['PL'], kind='linear')
                # plgrid = axpl(xgrid)
                # ax.fill_between( fig['ax']['grid']['xi'], -0.07, -0.04, where=fig['ax']['grid']['PL']>0.1354, color='#f8a501', step='mid', alpha=0.8 )
                # ax.fill_between( fig['ax']['grid']['xi'], -0.07, -0.04, where=fig['ax']['grid']['PL']>0.60583, color='#10a6e7', step='mid', alpha=0.7 )                
                ax.fill_between( fig['ax']['pdfkdedata']['xx'], -0.07, -0.04, where=plgrid>0.1354, color='#f8a501', step='mid', alpha=0.8 )
                ax.fill_between( fig['ax']['pdfkdedata']['xx'], -0.07, -0.04, where=plgrid>0.60583, color='#10a6e7', step='mid', alpha=0.7 )
            self.ax_setticks(fig, 'x')

            if self.cf.get(fig['section'], 'x_scale').strip().lower() == "flat":
                if not self.cf.get(fig['section'], 'x_ticks')[0:4] == 'Manu':
                    ax.set_xticks(fig['ax']['ticks']['x'])
                    ax.xaxis.set_minor_locator(AutoMinorLocator())
            elif self.cf.get(fig['section'], 'x_scale').strip().lower() == "log":
                ax.set_xscale('log')
            if self.cf.get(fig['section'], 'x_ticks')[0:4] == 'Manu':
                ax.xaxis.set_major_locator(ticker.FixedLocator(fig['ax']['ticks']['x'][0]))
                ax.set_xticklabels(fig['ax']['ticks']['x'][1])
                if self.cf.get(fig['section'], 'x_scale').strip().lower() == "flat":
                    ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.set_xlim(fig['ax']['lim']['x'][0], fig['ax']['lim']['x'][1])
            ax.yaxis.set_minor_locator(FixedLocator(np.linspace(0, 1, 26)))

            ax.tick_params( labelsize=fig['colorset']['ticks']['labelsize'],  direction=fig['colorset']['ticks']['direction'],  bottom=fig['colorset']['ticks']['bottom'],  left=fig['colorset']['ticks']['left'],  top=fig['colorset']['ticks']['top'],  right=fig['colorset']['ticks']['right'], which='both' )
            ax.tick_params(which='major', length=fig['colorset']['ticks']['majorlength'], color=fig['colorset']['ticks']['majorcolor'])
            ax.tick_params(which='minor', length=fig['colorset']['ticks']['minorlength'], color=fig['colorset']['ticks']['minorcolor'])
            ax.set_xlabel(r"{}".format(self.cf.get(fig['section'], 'x_label')), fontsize=30)
            ax.set_ylabel(r"{}".format(self.cf.get(fig['section'], 'chi2_label')), fontsize=30)
            ax.xaxis.set_label_coords(0.5, -0.068)
            plax = ax.secondary_yaxis("right")
            plax.set_ylabel(r"{}".format(self.cf.get(fig['section'], "pdf_label")), fontsize=30)
            plax.yaxis.set_major_locator(FixedLocator([]))
            plax.tick_params( labelsize=3, labelcolor="None",  direction=fig['colorset']['ticks']['direction'] )

            if self.cf.has_option(fig['section'], 'Line_draw'):
                self.drawline(fig, ax)
            
            if self.cf.has_option(fig['section'], "Text"):
                self.drawtext(fig, ax)

            if self.cf.has_option(fig['section'], "drawlegend"):
                if eval(self.cf.get(fig['section'], "drawlegend")):
                    print("\tTimer: {:.2f} Second;  Drawing default format legend ...".format(time.time()-fig['start']))
                    axlegend = fig['fig'].add_axes([0.14, 0.875, 0.72, 0.07])
                    axlegend.spines['top'].set_visible(False)
                    axlegend.spines['bottom'].set_visible(False)
                    axlegend.spines['left'].set_visible(False)
                    axlegend.spines['right'].set_visible(False)
                    axlegend.xaxis.set_major_locator(NullLocator())
                    axlegend.yaxis.set_major_locator(NullLocator())
                    axlegend.set_xlim(0, 100)
                    axlegend.set_ylim(-2.5, 2)
                    axlegend.step([2,3,5,7,9, 10], [0.5, 0.5, 1.3, 1.5, 0.2, 0.2], color='red', where='mid', zorder=0)
                    axlegend.fill_between([2,3,5,7,9, 10], 0, [0.5, 0.5, 1.3, 1.5, 0.2, 0.2], color='red', step='mid', alpha=0.1)
                    axlegend.text(12, 0.2, r"$\rm Profile~Likelihood$", fontsize=10.5)
                    axlegend.fill_between([2, 10], -0.7, -2, color='#f8a501', step='mid', alpha=0.8)
                    axlegend.fill_between([4, 8], -0.7, -2,  color='#10a6e7', step='mid', alpha=0.7)
                    axlegend.text(12, -1.9, r"$\rm Best$-$\rm fit~point,~1\sigma~\&~2\sigma~confidence~interval$", fontsize=10.5)
                    legendxi = np.linspace(52, 60, 100)
                    legendyy = 1.5 * np.exp(-(legendxi-55.0)**2/2) + 0.6 * np.exp(-(legendxi-58)**2/ 1.5 )
                    axlegend.plot(legendxi, legendyy, '-', linewidth = fig['colorset']['1dpdf']['line']['width'], color=fig['colorset']['1dpdf']['line']['color'], alpha=fig['colorset']['1dpdf']['line']['alpha'],zorder=10)
                    axlegend.fill_between(legendxi, 0, legendyy, color='w', zorder=1)
                    axlegend.fill_between(legendxi, 0, legendyy, where=legendyy>0.23, edgecolor=fig['colorset']['1dpdf']['2sigma']['edge'], facecolor=fig['colorset']['1dpdf']['2sigma']['facecolor'],  alpha=fig['colorset']['1dpdf']['2sigma']['alpha'], zorder=5)
                    axlegend.fill_between(legendxi, 0, legendyy, where=legendyy>0.72, edgecolor=fig['colorset']['1dpdf']['1sigma']['edge'], facecolor=fig['colorset']['1dpdf']['1sigma']['facecolor'],  alpha=fig['colorset']['1dpdf']['1sigma']['alpha'], zorder=6)
                    axlegend.fill_between(legendxi, -0.7, -2, where=legendyy>0.23, edgecolor=fig['colorset']['1dpdf']['2sigma']['edge'], facecolor=fig['colorset']['1dpdf']['2sigma']['facecolor'],  alpha=fig['colorset']['1dpdf']['2sigma']['alpha'], zorder=5)
                    axlegend.fill_between(legendxi, -0.7, -2, where=legendyy>0.72, edgecolor=fig['colorset']['1dpdf']['1sigma']['edge'], facecolor=fig['colorset']['1dpdf']['1sigma']['facecolor'],  alpha=fig['colorset']['1dpdf']['1sigma']['alpha'], zorder=6)
                    if self.cf.has_option(fig['section'], "BestPoint"):
                        for item in fig['colorset']['1dpdf']['pdfmode']:
                            axlegend.plot( [55.0, 55.0], [0.05, 1.45 ], '-', linewidth=item['width'], color=item['color'], alpha=item['alpha'], zorder=7)
                            axlegend.plot( [55.0, 55.0], [-0.8, -1.95], '-', linewidth=item['width'], color=item['color'], alpha=item['alpha'], zorder=7)  
                        for item in fig['colorset']['1dpdf']['bestpoint']:
                            axlegend.plot( [7.0, 7.0], [-0.8, -1.95], '-', linewidth=item['width'], color=item['color'], alpha=item['alpha'], zorder=7 )
                    axlegend.text(62, 0.2, r"$\rm Posterior~PDF$", fontsize=10.5)
                    axlegend.text(62, -1.9, r"$\rm PDF~mode, ~1\sigma~\&~2\sigma~credible~region$", fontsize=10.5)

            if 'save' in self.cf.get(fig['section'], 'print_mode'):
                from matplotlib.backends.backend_pdf import PdfPages
                fig['fig'] = plt
                fig['file'] = "{}/{}".format(self.figpath, fig['name'])
                fig['fig'].savefig("{}.pdf".format(fig['file']), format='pdf')
                self.compress_figure_to_PS(fig['file'])
                print("\tTimer: {:.2f} Second;  Figure {} saved as {}".format(time.time()-fig['start'], fig['name'], "{}/{}.pdf".format(self.figpath, fig['name'])))
            if 'show' in self.cf.get(fig['section'], 'print_mode'):
                plt.show()



        elif fig['type'] == "TernaryRGB_Scatter":
            print("\n=== Ploting Figure : {} ===".format(fig['name']))
            fig['start'] = time.time()
            self.load_ternary_configure(fig, "TernaryRGB")
            self.makecanvas(fig)
            self.init_ternary(fig, 'axt')
            self.init_ternary(fig, 'axr')
            self.init_ternary(fig, 'axg')
            self.init_ternary(fig, 'axb')

            self.basic_selection(fig)
            self.getTernaryRGBData(fig)
            fig['ax']['axt'].scatter(fig['var']['axdata']['x'], fig['var']['axdata']['y'], marker='^', color=fig['var']['axdata']['c'], s=5, zorder=1, alpha=1)
            fig['ax']['axr'].scatter(fig['var']['axdata']['x'], fig['var']['axdata']['y'], marker='^', color=fig['var']['axdata']['r'], s=1, zorder=1, alpha=0.8)
            fig['ax']['axg'].scatter(fig['var']['axdata']['x'], fig['var']['axdata']['y'], marker='^', color=fig['var']['axdata']['g'], s=1, zorder=1, alpha=0.8)
            fig['ax']['axb'].scatter(fig['var']['axdata']['x'], fig['var']['axdata']['y'], marker='^', color=fig['var']['axdata']['b'], s=1.5, zorder=1, alpha=1)

            self.set_Ternary_label(fig, 'axt')

            if 'save' in self.cf.get(fig['section'], 'print_mode'):
                from matplotlib.backends.backend_pdf import PdfPages
                fig['fig'] = plt
                fig['file'] = "{}/{}".format(self.figpath, fig['name'])
                fig['fig'].savefig("{}.pdf".format(fig['file']), format='pdf')
                self.compress_figure_to_PS(fig['file'])
                print("\tTimer: {:.2f} Second;  Figure {} saved as {}".format(time.time()-fig['start'], fig['name'], "{}/{}.pdf".format(self.figpath, fig['name'])))
            if 'show' in self.cf.get(fig['section'], 'print_mode'):
                plt.show()
        elif fig['type'] == "Ternary_Scatter":
            print("\n=== Ploting Figure : {} ===".format(fig['name']))
            fig['start'] = time.time()
            self.load_ternary_configure(fig, "Ternary_Default")
            self.makecanvas(fig)


    def find_critical_prob(self, data, prob):
        norm = data['pdf'].sum()
        temp_pdf = data.sort_values(by=['pdf'], ascending=True)
        temp_pdf = temp_pdf.reset_index()
        temp_pdf.pop("index")
        idx = {
            'min':  0,
            'upp':  temp_pdf.shape[0]-1,
            'id':   temp_pdf.shape[0]//2,
            'flag': True
        }
        while idx['flag']:
            if idx['upp'] - idx['min'] < 2:
                idx['flag'] = False 
            elif temp_pdf['pdf'][idx['id']:].sum() / norm > prob:
                idx['min'] = idx['id']
                idx['id'] = (idx['upp']+idx['min'])//2
            else:
                idx['upp'] = idx['id']
                idx['id'] = (idx['upp']+idx['min'])//2
        return temp_pdf['pdf'][idx['id']]

    def set_Ternary_label(self, fig, ax):
        def setlabel(pa, pb, pc, label, sty):
            posx = 0.5*(pa[0]+pb[0])
            posy = 0.5*(pa[1]+pb[1])
            posx = posx + sty['label']['offline']*(posx - pc[0])
            posy = posy + sty['label']['offline']*(posy - pc[1])
            fig['cv'].text(
                posx,
                posy,
                r"${}$".format(self.cf.get(fig['section'], "{}_label".format(label))),
                fontsize=sty['label']['fontsize'],
                color=sty['label']['color'],
                rotation= -1.0* sty['label']['rotation'][label],
                horizontalalignment="center",
                verticalalignment="center"
            )

        axs = "{}size".format(ax)
        pointA = [fig['cf'][axs]['x'], fig['cf'][axs]['y']]
        pointB = [fig['cf'][axs]['x'] + fig['cf'][axs]['width'], fig['cf'][axs]['y']]
        pointC = [fig['cf'][axs]['x']+0.5*fig['cf'][axs]['width'], fig['cf'][axs]['y']+fig['cf'][axs]['height']]
        axsty  = fig['cf'][fig['cf'][axs]["axisstyle"]]
        setlabel(pointA, pointB, pointC, "bottom", axsty)
        setlabel(pointB, pointC, pointA, "right", axsty)
        setlabel(pointC, pointA, pointB, "left", axsty)

    def load_ternary_configure(self, fig, label):
        with open("{}/ternary_config.json".format(pwd), 'r') as f1:
            data = json.load(f1)
            for style in data:
                if style['label'] == style['label']:
                    fig['cf'] = style
                    fig['fig'].set_figheight(fig['cf']['figureSize']['height'])
                    fig['fig'].set_figwidth(fig['cf']['figureSize']['width'])
                    break

    def makecanvas(self, fig):
        fig['fig'].set_figheight(fig['cf']['figureSize']['height'])
        fig['fig'].set_figwidth(fig['cf']['figureSize']['width'])
        fig['cv'] = fig['fig'].add_axes([0., 0., 1., 1.])
        fig['cv'].axis("off")
        fig['cv'].set_xlim(0, 1)
        fig['cv'].set_ylim(0, 1)
        fig['ax'] = {}
    
    def init_ternary(self, fig, ax):
        def ticks(axs, spl, epl, info):
            axsty = fig['cf'][fig['cf'][axs]['axisstyle']]['ticks']
            if eval(axsty['switch']):
                xl = np.linspace(spl[0], epl[0], fig['cf'][axs]["multiple"]+1)
                yl = np.linspace(spl[1], epl[1], fig['cf'][axs]["multiple"]+1)
                from math import sin, cos, radians
                for ii in range(fig['cf'][axs]["multiple"]+1):
                    fig['cv'].plot(
                        [xl[ii], xl[ii]+axsty['length']*cos(radians(axsty['angle'][info]))],
                        [yl[ii], yl[ii]+axsty['length']*sin(radians(axsty['angle'][info]))],
                        '-',
                        linewidth   = axsty['width'],
                        color       = axsty['color'],
                        solid_capstyle = axsty['end'],
                        zorder=10000
                    )
                    fig['cv'].text( 
                        xl[ii]+axsty['length']*cos(radians(axsty['angle'][info])) + axsty["labelshift"][info][0], 
                        yl[ii]+axsty['length']*sin(radians(axsty['angle'][info])) + axsty["labelshift"][info][1],
                        r"${:.1f}$".format(fig['cf'][axs]['scale'][0] + ii/fig['cf'][axs]["multiple"] *fig['cf'][axs]['scale'][1]),
                        fontsize=axsty['fontsize'],
                        rotation=axsty['fontangle'][info],
                        horizontalalignment=axsty["labelshift"][info][2],
                        verticalalignment=axsty["labelshift"][info][3]
                    )
        def gridline(axs, pa, pb, pc, info):
            axsty = fig['cf'][fig['cf'][axs]['axisstyle']]['gridline']
            if eval(axsty['switch']):
                xd = np.linspace(pa[0], pb[0], fig['cf'][axs]['multiple']+1)
                yd = np.linspace(pa[1], pb[1], fig['cf'][axs]['multiple']+1)
                xe = np.linspace(pc[0], pb[0], fig['cf'][axs]['multiple']+1)
                ye = np.linspace(pc[1], pb[1], fig['cf'][axs]['multiple']+1)
            for ii in range(fig['cf'][axs]['multiple']-1):
                plt.plot(
                    [xd[ii+1], xe[ii+1]],
                    [yd[ii+1], ye[ii+1]],
                    axsty[info]['linestyle'],
                    linewidth   = axsty[info]['linewidth'],
                    color       = axsty[info]['color'],
                    zorder      = 9999,
                    transform   = fig['cv'].transAxes,
                    alpha       = axsty['alpha']
                )
        
        axs = "{}size".format(ax)
        fig['ax'][ax] = fig['fig'].add_axes([
            fig['cf'][axs]['x'],
            fig['cf'][axs]['y'],
            fig['cf'][axs]['width'],
            fig['cf'][axs]['height']
        ])
        fig['ax'][ax].axis('off')

        fig['cv'].plot(
            [fig['cf'][axs]['x'], fig['cf'][axs]['x']+fig['cf'][axs]['width'], fig['cf'][axs]['x']+0.5*fig['cf'][axs]['width'], fig['cf'][axs]['x'], fig['cf'][axs]['x']+0.5*fig['cf'][axs]['width']],
            [fig['cf'][axs]['y'], fig['cf'][axs]['y'], fig['cf'][axs]['y']+fig['cf'][axs]['height'], fig['cf'][axs]['y'],fig['cf'][axs]['y'] ], 
            '-',
            linewidth=fig['cf'][fig['cf'][axs]["axisstyle"]]['axisline']['width'],
            color=fig['cf'][fig['cf'][axs]["axisstyle"]]['axisline']['color'],
            solid_joinstyle='miter',
            transform=fig['cv'].transAxes,
            zorder=10000
        )
        pointA = [fig['cf'][axs]['x'], fig['cf'][axs]['y']]
        pointB = [fig['cf'][axs]['x'] + fig['cf'][axs]['width'], fig['cf'][axs]['y']]
        pointC = [fig['cf'][axs]['x']+0.5*fig['cf'][axs]['width'], fig['cf'][axs]['y']+fig['cf'][axs]['height']]
        ticks(axs, pointC, pointA, "left")
        ticks(axs, pointB, pointC, "right")
        ticks(axs, pointA, pointB, "bottom")
        gridline(axs, pointC, pointA, pointB, "left")
        gridline(axs, pointB, pointC, pointA, "right")
        gridline(axs, pointA, pointB, pointC, "bottom")
        fig['ax'][ax].set_xlim(fig['cf'][axs]['scale'][0],fig['cf'][axs]['scale'][1] )
        fig['ax'][ax].set_ylim(fig['cf'][axs]['scale'][0],fig['cf'][axs]['scale'][1] )

    def scatter_classify_data(self, fig, bo):
        x_sel = self.var_symbol(bo)
        for x in x_sel:
            bo = bo.replace("_{}".format(x), "fig['data']['{}']".format(x))
        if "&FC_" in bo:
            for ii, func in enumerate(self.funcs):
                bo = bo.replace(func['name'], "self.funcs[{}]['expr']".format(ii))
        fig['classify'] ={}
        fig['classify']['data'] = fig['data'][eval(bo)].reset_index()
        x_info = self.cf.get(fig['section'], 'x_variable')
        if x_info[0: 3] == '&Eq':
            x_info = x_info[4:]
            x_sel = self.var_symbol(x_info)
            for x in x_sel:
                x_info = x_info.replace("_{}".format(x), "fig['classify']['data']['{}']".format(x))
            if "&FC_" in x_info:
                for ii, func in enumerate(self.funcs):
                    x_info = x_info.replace(func['name'], "self.funcs[{}]['expr']".format(ii))
            fig['classify']['x'] = eval(x_info)
        elif x_info in fig['classify']['data'].columns.values:
            fig['classify']['x'] = fig['classify']['data'][x_info]
        y_info = self.cf.get(fig['section'], 'y_variable')
        if y_info[0: 3] == '&Eq':
            y_info = y_info[4:]
            x_sel = self.var_symbol(y_info)
            for x in x_sel:
                y_info = y_info.replace("_{}".format(x), "fig['classify']['data']['{}']".format(x))
            if "&FC_" in y_info:
                for ii, func in enumerate(self.funcs):
                    y_info = y_info.replace(func['name'], "self.funcs[{}]['expr']".format(ii))
            fig['classify']['y'] = eval(y_info)
        elif y_info in fig['classify']['data'].columns.values:
            fig['classify']['y'] = fig['classify']['data'][y_info]
        if self.cf.has_option(fig['section'], 'c_variable'):
            c_info = self.cf.get(fig['section'], 'c_variable')
            if c_info[0: 3] == '&Eq':
                c_info = c_info[4:]
                x_sel = self.var_symbol(c_info)
                for x in x_sel:
                    c_info = c_info.replace("_{}".format(x), "fig['classify']['data']['{}']".format(x))
                if "&FC_" in c_info:
                    for ii, func in enumerate(self.funcs):
                        c_info = c_info.replace(func['name'], "self.funcs[{}]['expr']".format(ii))
                fig['classify']['c'] = eval(c_info)
            elif c_info in fig['classify']['data'].columns.values:
                fig['classify']['c'] = fig['classify']['data'][c_info]

    def get_Linestyle(self, style):
        style_file = os.path.join(self.cf.get('PLOT_CONFI', 'path'), self.cf.get("COLORMAP", "StyleSetting"))
        style = style.strip()
        if (not style[0] == '&') or (not os.path.exists(style_file)):
            print("\tLine info Error: Unvaliable line format {} \n\t Default Line Format used".format(style))
            return {'width':    2., 'color':    '#b71c1c', 'style':    '-', 'alpha':    1.0, 'marker':   None, 'markersize':   5}
        else:
            style = style.split('_')
            style = {
                'name':     style[0].replace("&", '').strip(),
                'label':    style[1].strip()
            }
            with open(style_file, 'r') as f1:
                default_style = json.load(f1)
            style_tag = False
            for n, v in default_style.items():
                if style['name'] == n:
                    for item in v:
                        if style['label'] == item['label']:
                            style = item['LineStyle']
                            style_tag = True
                            break
            if style_tag:
                return style
            else:
                print("\tLine info Error: Unvaliable line format {} \n\t Default Line Format used".format(style))
                return {'width':    2., 'color':    '#b71c1c', 'style':    '-', 'alpha':    1.0, 'marker':   None, 'markersize':   5}

    def drawline(self, fig, ax):
        def get_variable(info):
            info = info.replace('{', '').replace('}', '').strip()
            info = info.split('|')
            info = {
                'varname':  info[0].strip(),
                'vardata':  np.linspace(float(info[1].strip()), float(info[2].strip()), int(info[3].strip()))
            }
            return info
        
        def get_data(expr, var):
            for ii, fc in enumerate(self.funcs):
                expr = expr.replace(fc['name'], "self.funcs[{}]['expr']".format(ii))
            y = []
            for ii in var['vardata']:
                y.append(eval(expr.replace('_{}'.format(var['varname']), str(ii))))
            return np.array(y)

        def get_function(var, func):
            var = get_variable(var)
            decode = re.compile(r'[{](.*?)[}]', re.S)
            func = re.findall(decode, func)
            for ii, fc in enumerate(func):
                line = fc.split(':')
                func[ii] = {
                    'varname':  line[0].strip(),
                    'vardata':  get_data(line[1], var)
                }
            func.append(var)
            return func 

        def get_line_info(line):
            info = line.split(',')
            if info[0].strip() == 'parametric':
                res = {}
                res['method'] = info.pop(0)
                res['var'] = info.pop(0)
                res['style'] = self.get_Linestyle(info.pop(-1))
                res['Func'] = ','.join(info)
                res['var'] = get_function(res['var'], res['Func'])
                xtag, ytag = False, False
                for ii, var in enumerate(res['var']):
                    if var['varname'] == 'x':
                        xx = var['vardata']
                        xtag = True
                    if var['varname'] == 'y':
                        yy = var['vardata']
                        ytag = True
                if xtag and ytag:
                    res['data'] = pd.DataFrame({
                        'x':  xx,
                        'y':  yy
                    })           
                elif not xtag:
                    print("\tLine Info Error: No x coordinate founded in Line Setting\n\t{}".format(line)) 
                elif not ytag:
                    print("\tLine Info Error: No y coordinate founded in Line Setting\n\t{}".format(line)) 
                return res
            elif info[0].strip() == "Equation":
                res = {}
                res['method'] = info.pop(0)
                res['var'] = info.pop(0)
                res['style'] = self.get_Linestyle(info.pop(-1))
                res['Func'] = ','.join(info)
                res['var'] = get_function(res['var'], res['Func'])
                xtag, ytag = False, False
                for ii, var in enumerate(res['var']):
                    if var['varname'] == 'x':
                        xx = var['vardata']
                        xtag = True
                    if var['varname'] == 'y':
                        yy = var['vardata']
                        ytag = True
                if xtag and ytag:
                    res['data'] = pd.DataFrame({
                        'x':  xx,
                        'y':  yy
                    })           
                elif not xtag:
                    print("\tLine Info Error: No x coordinate founded in Line Setting\n\t{}".format(line)) 
                elif not ytag:
                    print("\tLine Info Error: No y coordinate founded in Line Setting\n\t{}".format(line)) 
                return res
            else:
                print("Line Drawing Error: No such line function methed {}\n\t-> {}".format(info[0], line))
                sys.exit(1)

        def draw(lineinfo):
            if lineinfo['style']['marker'] == 'None':
                ax.plot(lineinfo['data']['x'], lineinfo['data']['y'], linewidth=lineinfo['style']['width'], color=lineinfo['style']['color'], linestyle=lineinfo['style']['style'], zorder=3000)

        fig['lineinfo'] = self.cf.get(fig['section'], 'Line_draw').split('\n')
        for ii, line in enumerate(fig['lineinfo']):
            draw(get_line_info(line))

    def get_TextStyle(self, style):
        style_file = os.path.join(self.cf.get('PLOT_CONFI', 'path'), self.cf.get("COLORMAP", "StyleSetting"))
        style = style.strip()
        if style[0] != '&' and type(eval(style)) == dict:
            style = eval(style)
            kys = ['FontSize', 'color', 'alpha']
            style_tag = True
            for k in kys:
                if 'FontSize' not in style.keys():
                    style_tag = False
            if style_tag:
                return style
            else:
                print("\tLine info Error: Unvaliable Text format {} \n\t Default Line Text Format used".format(style))
                return {"FontSize": 20, "color": "#b71c1c", "alpha": 1.0}
        elif (not style[0] == '&') or (not os.path.exists(style_file)):
            print("\tLine info Error: Unvaliable Text format {} \n\t Default Line Text Format used".format(style))
            return {"FontSize": 20, "color": "#b71c1c", "alpha": 1.0}
        else:
            style = style.split('_')
            style = {
                'name':     style[0].replace("&", '').strip(),
                'label':    style[1].strip()
            }
            with open(style_file, 'r') as f1:
                default_style = json.load(f1)
            style_tag = False
            for n, v in default_style.items():
                if style['name'] == n:
                    for item in v:
                        if style['label'] == item['label']:
                            style = item['TextStyle']
                            style_tag = True
                            break
            if style_tag:
                return style
            else:
                print("\tLine info Error: Unvaliable Text format {} \n\t Default Line Text Format used".format(style))
                return {"FontSize": 20, "color": "#b71c1c", "alpha": 1.0} 

    def drawtext(self, fig, ax):
        def get_text_info(line):
            decode = re.compile(r'[(](.*?)[)]', re.S)
            pos = re.findall(decode, line)[0]
            line = line.replace('({})'.format(pos), '').strip()
            pos = pos.split(',')
            line = line.lstrip('|')
            line = line.split('|')
            res = {
                'pos':      [float(pos[0].strip()), float(pos[1].strip())],
                'rotation': float(line.pop(0).strip()),
                'style':    self.get_TextStyle(line.pop(-1).strip()),
                'text':     '|'.join(line).strip()
            }
            return res
        
        def draw(info):
            ax.text(info['pos'][0], info['pos'][1], r"{}".format(info['text']), fontsize=info['style']['FontSize'], color=info['style']['color'], alpha=info['style']['alpha'], rotation=info['rotation'], horizontalalignment='left', verticalalignment='bottom', zorder=1999) 

        fig['textinfo'] = self.cf.get(fig['section'], 'Text').split('\n')
        for ii, line in enumerate(fig['textinfo']):
            draw(get_text_info(line))

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
                    label.append(r"{}".format(it.split(',')[1].strip()))
                fig['ax']['ticks'][aa] = tuple([tk, label])
            else:
                tick = tick.split(',')
                for ii in range(len(tick)):
                    tick[ii] = tick[ii].strip()
                fig['ax']['ticks'][aa] = np.linspace(float(tick[0]), float(tick[1]), int(tick[2]))
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

    def Get1DStatData(self, fig):
        if self.cf.has_option(fig['section'], "stat_variable"):
            fig['var'] = {}
            self.get_variable_data(fig, 'x', self.cf.get(fig['section'], 'x_variable'))
            for line in self.cf.get(fig['section'], 'stat_variable').split('\n'):
                if line.split(",")[0].strip() == 'CHI2':
                    self.get_variable_data(fig, 'CHI2', ','.join(line.split(",")[1:]).strip())
                if line.split(',')[0].strip() == 'PDF':
                    self.get_variable_data(fig, 'PDF', ','.join(line.split(',')[1:]).strip())
            if 'CHI2' in fig['var'].keys() and 'PDF' in fig['var'].keys():
                fig['var']['data'] = pd.DataFrame({
                    "x":    fig['var']['x'],
                    "CHI2": fig['var']['CHI2'],
                    "PDF":  fig['var']['PDF']    
                })
                fig['var'].pop('x')
                fig['var'].pop("PDF")
                fig['var'].pop('CHI2')
                fig['var']['type'] = ["x", "PDF", "CHI2"]
            elif 'CHI2' in fig['var'].keys():
                fig['var']['data'] = pd.DataFrame({
                    "x":    fig['var']['x'],
                    "CHI2": fig['var']['CHI2']
                })
                fig['var'].pop('x')
                fig['var'].pop('CHI2')
                fig['var']['type'] = ['x', 'CHI2']
            elif 'PDF' in fig['var'].keys():
                fig['var']['data'] = pd.DataFrame({
                    "x":    fig['var']['x'],
                    "PDF":  fig['var']['PDF']    
                })
                fig['var'].pop('x')
                fig['var'].pop("PDF")
                fig['var']['type'] = ['x', 'PDF']

    def getTernaryRGBData(self, fig):
        def var_normalize(a, b, c, d, e):
            res = pd.DataFrame({
                'left'  :   a,
                'right' :   b,
                'r'     :   c,
                'g'     :   d,
                'b'     :   e
            })
            res['sum']      = res['left'] + res['right'] + res['r'] + res['g'] + res['b']
            res['left']     = res['left']/res['sum']
            res['right']    = res['right']/res['sum']
            res['r']        = res['r']/res['sum']
            res['g']        = res['g']/res['sum']
            res['b']        = res['b']/res['sum']
            res['bottom']   = res['r'] + res['g'] + res['b']
            maxrg = max(max(res['r']), max(res['g']))
            res['r']        = np.power(res['r']/maxrg, 0.5)
            res['g']        = np.power(res['g']/maxrg, 0.5)
            res['b']        = np.power(res['b']/max(res['b']), 0.25)
            
            return res 

        if self.cf.has_option(fig['section'], 'left_variable') & self.cf.has_option(fig['section'], 'right_variable') & self.cf.has_option(fig['section'], 'r_variable') & self.cf.has_option(fig['section'], 'g_variable') &self.cf.has_option(fig['section'], 'b_variable'):
            fig['var'] = {}
            self.get_variable_data(fig, 'left', self.cf.get(fig['section'], 'left_variable'))
            self.get_variable_data(fig, 'right', self.cf.get(fig['section'], 'right_variable'))
            self.get_variable_data(fig, 'r', self.cf.get(fig['section'], 'r_variable'))
            self.get_variable_data(fig, 'g', self.cf.get(fig['section'], 'g_variable'))
            self.get_variable_data(fig, 'b', self.cf.get(fig['section'], 'b_variable'))

            fig['var']['oridata'] = var_normalize(fig['var']['left'], fig['var']['right'], fig['var']['r'], fig['var']['g'], fig['var']['b'])
            fig['var']['axdata'] = pd.DataFrame({
                'x':    fig['var']['oridata']['bottom'] + 0.5*fig['var']['oridata']['right'],
                'y':    fig['var']['oridata']['right']
            })
            fig['var']['axdata']['c'] = fig['var']['oridata'].apply(lambda  x: tuple([x['r'], x['g'], x['b']]), axis=1)           
            fig['var']['axdata']['r'] = fig['var']['oridata'].apply(lambda  x: tuple([x['r'], 0.3*x['r'], 0.]), axis=1)           
            fig['var']['axdata']['g'] = fig['var']['oridata'].apply(lambda  x: tuple([0.3*x['g'], x['g'], 0.]), axis=1)           
            fig['var']['axdata']['b'] = fig['var']['oridata'].apply(lambda  x: tuple([0.3*x['b'], 0.42*x['b'], x['b']]), axis=1)           

    def Get2DData(self, fig):
        fig['var'] = {}
        self.get_variable_data(fig, 'x', self.cf.get(fig['section'], 'x_variable'))
        self.get_variable_data(fig, 'y', self.cf.get(fig['section'], 'y_variable'))
        fig['var']['data'] = pd.DataFrame({
            'x':    fig['var']['x'],
            'y':    fig['var']['y']
        })

    def Get3DData(self, fig):
        fig['var'] = {}
        self.get_variable_data(fig, 'x', self.cf.get(fig['section'], 'x_variable'))
        self.get_variable_data(fig, 'y', self.cf.get(fig['section'], 'y_variable'))
        self.get_variable_data(fig, 'c', self.cf.get(fig['section'], 'c_variable'))
        fig['var']['data'] = pd.DataFrame({
            'x':    fig['var']['x'],
            'y':    fig['var']['y'],
            'c':    fig['var']['c']
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
            fig['var'][name] = fig['data'][varinf]
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
                data_dir = os.path.join(self.cf.get('PLOT_CONFI', 'path'), self.cf.get(item, 'file'))
                fun['data'] = pd.read_csv(data_dir)
                fun['expr'] = interp1d(fun['data']['x'], fun['data']['y'])
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
            # print("Total Data is -> {} rows".format(self.data.shape[0]))
            if "*Bool*" not in self.data.columns:
                bool_list = np.ones(self.data.shape[0], dtype=np.bool)
                self.data['*Bool*'] = bool_list
                bo = bo + "& self.data['*Bool*']"
            else:
                bool_list = np.ones(self.data.shape[0], dtype=np.bool)
                self.data['abc**cba**bool**loob**alphabeta'] = bool_list
                bo = bo + "& self.data['abc**cba**bool**loob**alphabeta']"
            fig['data'] = self.data[eval(bo)].reset_index()
            print("Selected Data is -> {} rows".format(fig['data'].shape[0]))






