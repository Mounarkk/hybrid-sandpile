# coding: 8859
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


import argparse
import pandas as pds
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import os
from pathlib import Path
import sys
import textwrap
from matplotlib.lines import Line2D



def openfile(path="./data/perf/data.csv", sepa=";"):
    try:
        return pds.read_csv(path, sep=sepa)
    except FileNotFoundError:
        print("File not found: ", path, file=sys.stderr)
        sys.exit(1)

# Donne tous les champs de df qui ne sont pas list�s


def complementaryCols(listeAttr, df):
    return [i for i in list(df.columns) if i not in listeAttr]


def constantCols(df):
    return df.columns[df.eq(df.iloc[0]).all()].tolist()


def shuffle(u, v, before=" ", equal="="):
    s = ""
    for i in zip(u, v):
        s += before + str(i[0]) + equal + str(i[1])
    return s


def factorizeLegendColRow(axisCol, args, axisAttr, df):
    if axisCol == None:
        return
    # list the attr that are functionally dependent on axisCol
    funcDeps = []
    for col in list(df.columns):
        if not col in axisAttr:
            projection = df[[axisCol, col]].groupby(axisCol).nunique()
            if projection[projection[col] > 1].empty:
                funcDeps += [col]
    if funcDeps != []:
        df["tmp"] = df[funcDeps].apply(
            lambda row: shuffle(funcDeps, row.values.tolist(), "  ", " = "), axis=1)
        for col in funcDeps:
            del df[col]
        df[axisCol] = df[axisCol].map(str) + " |" + df["tmp"]
        del df["tmp"]


def computeTitleAndLegend(args, df):
    axisAttr = [args.x,  args.col, args.row, args.heaty] + args.all_y
    constantParameters = constantCols(df)
    if len(constantParameters) == 0:
        title = (u'Courbe de {y} en fonction de {x}').format(
            x=args.x, y=args.all_y)
    else:
        title = (u'{cons}').format(x=args.x, y=args.all_y,
                                   cons=textForConstantParameters(df, constantParameters))
        for col in constantParameters:
            if not col in axisAttr:
                del df[col]

    factorizeLegendColRow(args.col, args, axisAttr, df)
    factorizeLegendColRow(args.row, args, axisAttr, df)

    attr = complementaryCols(['time', 'ref'] + axisAttr, df)

    if attr != []:
        df['legend'] = df[attr].apply(
            lambda row: shuffle(attr, row.values.tolist()), axis=1)
        for col in attr :
            del df[col]
    return title


# Cree les nom avec toutes les valeurs non utilisees pour le reste du graphique

def textForConstantParameters(df, constantParameters):
    string = ""
    # Donne les valeurs des constantes dans l'ordre des colonnes
    dataConst = [df[i].iloc[0] for i in constantParameters]
    for i in range(len(constantParameters)):
        if dataConst[i] != "none" and str(dataConst[i]) != "nan":
            string = string + \
                str(constantParameters[i]) + "=" + str(dataConst[i]) + " "
    return string


def computeSpeedUpAttr(df, args):  # Automatise la creation du speedup
    group = ['machine', 'size', 'kernel', 'iterations', 'arg']
    group = [
        attr for attr in group if attr not in args.delete and attr in list(df.columns)]
    if args.RefTimeVariants == [] and args.RefTimeTiling == [] :
        refDF = df[df.threads == 1].reset_index(drop=True)
    elif args.RefTimeTiling == []:
               refDF = df[df.variant.isin(args.RefTimeVariants)
                   & df.threads == 1].reset_index(drop=True)
    elif args.RefTimeVariants == []:
               refDF = df[df.tiling.isin(args.RefTimeTiling)
                   & df.threads == 1].reset_index(drop=True)    
    else:
        refDF = df[df.variant.isin(args.RefTimeVariants)
                   &  df.tiling.isin(args.RefTimeTiling)
                   & df.threads == 1].reset_index(drop=True)
    if refDF.empty:
        sys.exit("No row with OMP_NUM_THREADS=1 to compute speedUP")

    for i in complementaryCols(group + ['time'], refDF):
        del refDF[i]

    refDF = refDF.loc[refDF.groupby(
        group).time.idxmin()].reset_index(drop=True)
    refDF = refDF.rename(columns={'time': 'refTime'})

    df = df.merge(refDF, how='inner')

    df['speedup'] = df['refTime'] / df['time']

    if args.noRefTime:
        del df['refTime']
    else:
        df['refTime'] = df['refTime'] // 1000
    return df


def heatFacet(*args, **kwargs):
    data = kwargs['data']
    data = data.pivot(index=args[1], columns=args[0], values=args[2])
    print("------------------------------------")
    print(data)
    print("------------------------------------")
    m = data.max().max()
    if "time (ms)" == args[2]:
        fmt = '.2f' if m < 10 else '.1f' if m < 100 else '.0f'
        g = sns.heatmap(data, cmap='rocket_r', annot=True,
                        fmt=fmt, annot_kws={"fontsize": 8})
    else:
        fmt = '.2f' if m < 1 else '.1f' if m < 10 else '.0f'
        g = sns.heatmap(data, cmap='rocket', annot=True,
                        fmt=fmt, annot_kws={"fontsize": 8})
    g.invert_yaxis()
    plt.yticks(rotation=0)


axes ={}
def twin_lineplot(x,y,**kwargs):
    ax = plt.gca()
    mini=kwargs.pop('mini')
    maxi=kwargs.pop('maxi')
    if 'label' in  kwargs.keys() :
        kwargs.pop('label')
    #if ax.xaxis not in axes.keys(): # utilité du test ???
    axes[ax.axis] = ax.twinx()
    axes[ax.axis].set(ylim=(mini,maxi) )
    sns.lineplot(x=x,y=y,ax=axes[ax.axis],label='_nolegend_',**kwargs)
    axes[ax.axis].set_ylabel("")
    axes[ax.axis].grid(visible=None)
    axes[ax.axis].tick_params(axis='y',labelsize=8)

def multiple_lineplots(args,df,g):
 
        linestyle_str = [ 'solid', 'dotted', 'dashed', 'dashdot',(0, (3, 1, 1, 1))]
        for  i in range(len(args.y)):
            g.map(sns.lineplot, args.x, args.y[i],err_style="bars", linestyle=linestyle_str[i])

        for i in range(len(args.y2)):
            mini = min(0,df[args.y2].min().min())
            maxi = df[args.y2].max().max()
            g.map(twin_lineplot, args.x, args.y2[i], err_style="bars", linestyle=linestyle_str[len(args.y) + i],mini=mini,maxi=maxi)

        g.set_axis_labels(x_var=args.x, y_var=args.y[0] if args.ylabel == None else args.ylabel, clear_inner=True)#


        g.add_legend()
        l = g._legend.legendHandles

        for ax in g.axes.flat:
            if ax.texts:
                txt = ax.texts[0]
                ax.text(txt.get_unitless_position()[0]+0.1, txt.get_unitless_position()[1],
                txt.get_text(),
                transform=ax.transAxes,
                va='center',fontsize='medium',
                rotation=-90)
                ax.texts[0].remove()

        for i in range(len(args.all_y)):
            g._legend.legendHandles.append(mlines.Line2D([], [], color='black', linestyle=linestyle_str[i], label= " " + args.all_y[i]))
            g._legend.texts.append(plt.text(0,0," " + args.all_y[i],visible=False))

        if sns.__version__ >= '0.11.2' :
            if len(g._legend.legendHandles) == len(args.all_y): 
                sns.move_legend(g,"lower center", bbox_to_anchor=[0.5,-0.009], ncol= len(args.all_y), title=None, frameon=False)
            else:
                sns.move_legend(g,"center right")
        return g


def easyPlotDataFrame(df, args):
    with sns.plotting_context("paper", font_scale=args.font_scale):
        pc_easyPlotDataFrame(df, args)

def pc_easyPlotDataFrame(df, args):
    title = computeTitleAndLegend(args, df)

    legend = "legend" if "legend" in list(df.columns) else None

    if not args.no_sort:
        df = df.sort_values(by=args.all_y[0], ascending=False)

    if (args.plottype == 'lineplot'): 
        g = sns.FacetGrid(df, row=args.row, col=args.col, hue=legend, sharex='col', sharey='row',
                          height=args.height, margin_titles=True, legend_out=not args.legendInside, aspect=args.aspect)
        g = multiple_lineplots(args,df,g)
    elif (args.plottype == 'heatmap'):
        df = df.groupby(complementaryCols(args.y, df), as_index=False).mean()
        g = sns.FacetGrid(df, row=args.row, col=args.col,
                          hue=legend, height=args.height, aspect=args.aspect)
        g = g.map_dataframe(heatFacet, args.x, args.heaty, args.y[0])

        # actually catplot raise an exception whenever args.kind in ["swarm","strip"] 
        # instead we map their associated function thanks to FacetGrid  
        # seaborn 0.12.2  matplotlib 3.6.3 & 3.7.1
    elif  args.kind in ["swarm","strip"] :
        funToMpap = sns.stripplot if args.kind == "strip" else sns.swarmplot
        g = sns.FacetGrid(df, row=args.row, col=args.col, sharex='col', sharey='row',
                            margin_titles=True, legend_out=not args.legendInside,
                          hue=legend, height=args.height, aspect=args.aspect)
        g = g.map_dataframe(funToMpap, args.x , args.y[0])
        g.add_legend()

    else:
        g = sns.catplot(data=df, x=args.x, y=args.y[0], row=args.row, col=args.col, hue=legend,
                        kind=args.kind, sharex='col', sharey='row',
                        height=args.height, margin_titles=True, legend_out=not args.legendInside, aspect=args.aspect, markers=Line2D.filled_markers)

    if args.comment != None:
        title += args.comment
    if not args.hideParameters:
        g.fig.suptitle(title, wrap=True, y=0.999) # wrap does not perform well whenever y=1 !
        plt.subplots_adjust(top=args.adjustTop)
    else:
        print(title)

    if args.yscale == "log2":
        plt.yscale("log", base=2)
    elif args.yscale != "linear":
        g.set(yscale=args.yscale)

    if args.xscale == "log2":
        plt.xscale("log", base=2)
    elif args.xscale != "linear":
        g.set(xscale=args.xscale)

    if args.plottype == 'heatmap':
        g.tight_layout(rect=[0, 0.03, 1, args.adjustTop])
    else:
        g.tight_layout()
    return g


perfAttr = ["time", "l1hits", "l2hits", "l3hits", "dramhits"]

def parseArguments(argv, computed_attr={}):
    global parser, args, perfAttr
    parser = argparse.ArgumentParser(
        argv,
        description='''Process performance plots.
        The labels of the options are similar to those of
        easypap for all that relates to data selection,
        and to those of seaborn for all aspects of graphic
        layout https://seaborn.pydata.org/introduction.html''')

    all = ["size", "iterations", "kernel", "variant", "tiling", "threads",
           "nb_tiles", "schedule", "label", "machine", "tile_size", "tileh", "tilew", "arg", "places"]

    perfAttr += list(computed_attr.keys())

    parser.add_argument("-x", "-heatx", choices=all +
                        ["custom"], default="threads")
    parser.add_argument("-heaty", choices=all+["custom"], default=None)
    parser.add_argument("-y", "-y1",
                        action='store', nargs='+',
                        choices= perfAttr + ["speedup", "throughput", "efficiency"], 
                        default=["speedup"])

    parser.add_argument("--ylabel",
                        action='store', nargs='?', 
                        default=None)

    parser.add_argument("-y2",
                        action='store', nargs='+',
                        choices= perfAttr + [ "speedup", "throughput", "efficiency"]+list(computed_attr.keys()) ,
                        default=[])



    parser.add_argument('-rtv', '--RefTimeVariants',
                        action='store', nargs='+',
                        help="list of variants to take into account to compute the speedUP RefTimes",
                        default=[])

    parser.add_argument('-rtt', '--RefTimeTiling',
                        action='store', nargs='+',
                        help="list of tiling functions to take into account to compute the speedUP RefTimes",
                        default=[])

    parser.add_argument("-C", "--col", choices=all+["custom"], default=None)
    parser.add_argument("-R", "--row", choices=all+["custom"], default=None)

    parser.add_argument('-of', '--output',
                        action='store', nargs='?',
                        help='Filename to output the plot (supported formats: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff)',
                        default='plot.pdf')

    parser.add_argument('-if', '--input',
                        action='store', nargs='?',
                        help="Data's filename",
                        const=os.getcwd() + "/data/perf/data.csv",
                        default=os.getcwd() + "/data/perf/data.csv")

    parser.add_argument('-k', '--kernel',
                        action='store', nargs='+',
                        help="list of kernels to plot",
                        default=[])

    parser.add_argument('-t', '--threads',
                        action='store', nargs='+', type=int,
                        help="list of numbers of threads to plot",
                        default=[])

    parser.add_argument('--delete',
                        action='store', nargs='+',
                        help="delete a column before proceeding data",
                        choices=all,
                        default=[]
                        )

    parser.add_argument('-v', '--variant',
                        action='store', nargs='+',
                        help="list of variants to plot",
                        default=[])

    parser.add_argument('-wt', '--tiling',
                        action='store', nargs='+',
                        help="list of tile functions to plot",
                        default=[])

    parser.add_argument('-th', '--tileh',
                        action='store', nargs='+', type=int,
                        help="list of tile heights to plot",
                        default=[])

    parser.add_argument('-tw', '--tilew',
                        action='store', nargs='+', type=int,
                        help="list of tile widths to plot",
                        default=[])

    parser.add_argument('-nt', '--nb_tiles',
                        action='store', nargs='+', type=int,
                        help="list of nb_tiles to plot",
                        default=[])


#    parser.add_argument('-ts', '--tile',
#                        action='store', nargs='*',type=int,
#                        help="print tile sizes rather than nb_tiles / list of tiles to plot",
#                        default=None)

    parser.add_argument('-m', '--machine',
                        action='store', nargs='+',
                        help="list of machines to plot",
                        default=[])

    parser.add_argument('-i', '--iterations',
                        action='store', nargs='+', type=int,
                        help="list of iterations to plot",
                        default=[])

    parser.add_argument('-sc', '--schedule',
                        action='store', nargs='+',
                        help="list of schedule policies to plot",
                        default=[])

    parser.add_argument('--places',
                        action='store', nargs='+',
                        help="list of schedule policies to plot",
                        default=[])

    parser.add_argument('--arg',
                        action='store', nargs='+',
                        help="list of arg value to plot",
                        default=[])

    parser.add_argument('-s', '--size',
                        action='store', nargs='+', type=int,
                        help="list of sizes to plot",
                        default=[])

    parser.add_argument('-lb', '--label',
                        action='store', nargs='+',
                        help="list of labels to plot",
                        default=[])

    parser.add_argument('--height',
                        action='store',
                        type=int,
                        help="to set the height of each subgraph",
                        default=4)

    parser.add_argument('--hideParameters',
                        action='store_true',
                        help="to hide constant parameters",
                        default=False)

    parser.add_argument('--legendInside',
                        action='store_true',
                        help="to print the legend inside the graph",
                        default=False)

    parser.add_argument("--no_sort",
                        action='store_true',
                        help="sort data following y",
                        default=False)

    parser.add_argument('--adjustTop',
                        action='store',
                        type=float,
                        help="to adjust the space for the suptitle",
                        default=1)

    parser.add_argument('--aspect',
                        action='store',
                        type=float,
                        help="to adjust the ratio length/height",
                        default=1.1)

    parser.add_argument('--font_scale',
                        action='store',
                        type=float,
                        help="to adjust the font of the title and the legend",
                        default=1.0)

    parser.add_argument('--noRefTime',
                        action='store_true',
                        help="do not print reftime in legend",
                        default=False)

    parser.add_argument('--yscale',
                        choices=["linear", "log", "log2", "symlog", "logit"],
                        action='store',
                        default="linear")

    parser.add_argument('--xscale',
                        choices=["linear", "log", "log2", "symlog", "logit"],
                        action='store',
                        default="linear")

    parser.add_argument('--plottype',
                        choices=['lineplot', 'catplot', 'heatmap'],
                        action='store',
                        default="lineplot")

    parser.add_argument('--kind',
                        choices=["strip", "swarm", "box", "violin",
                                 "boxen", "point", "bar", "count"],
                        help="kind of barplot (see sns catplot)",
                        action='store',
                        default="swarm")

    parser.add_argument('--comment',
                        action='store',
                        help="add some text to the title",
                        default=None)

    args = parser.parse_args()
    args.computed_attr = computed_attr
    return args


def substitute_attr_y(args,src,dst):
    if src in args.y :
        i = args.y.index(src)
        args.y = args.y[:i]+[dst]+ args.y[i+1:]
    if src in args.y2 :
        i = args.y2.index(src)
        args.y2 = args.y2[:i]+[dst]+ args.y2[i+1:]


def getDataFrame(args):
    # Lecture du fichier d'experiences:
    df = openfile(args.input, sepa=";")

    if args.kernel != []:
        df = df[df.kernel.isin(args.kernel)].reset_index(drop=True)

    if args.arg != []:
        df = df[df.arg.isin(args.arg)].reset_index(drop=True)

    if args.iterations != []:
        df = df[df.iterations.isin(args.iterations)].reset_index(drop=True)

    if args.size != []:
        df = df[df["size"].isin(args.size)].reset_index(drop=True)

    if args.machine != []:
        df = df[df.machine.isin(args.machine)].reset_index(drop=True)

    df = df.assign(**args.computed_attr)

    if args.delete != []:
        for attr in args.delete:
            del df[attr]

    args.all_y =  args.y + args.y2

    if  "speedup" in args.all_y:
        df = computeSpeedUpAttr(df, args)

    if "efficiency" in args.all_y:
        if  "speedup" not in args.all_y :
            df = computeSpeedUpAttr(df, args)
            df['efficiency'] = df['speedup'] / df['threads']
            del df['speedup']
        else :
            df['efficiency'] = df['speedup'] / df['threads']

    if "throughput" in args.all_y:
        substitute_attr_y(args, 'throughput','throughput (MPixel / s)')
        df['throughput (MPixel / s)'] = (df['size'] ** 2) * df['iterations'] / df['time']

    for attr in perfAttr :
        if attr not in args.all_y:
            del df[attr]

    if "time" in args.all_y :
        df['time'] = df['time'] / 1000
        df.rename(columns={'time': 'time (ms)'}, inplace=True)
        substitute_attr_y(args,'time',"time (ms)")

    args.all_y = args.y + args.y2

    if args.label != []:
        df = df[df.label.isin(args.label)].reset_index(drop=True)

    if args.schedule != []:
        df = df[df.schedule.isin(args.schedule)].reset_index(drop=True)

    if args.places != []:
        df = df[df.places.isin(args.places)].reset_index(drop=True)

    if args.threads != []:
        df = df[df.threads.isin(args.threads)].reset_index(drop=True)

    if args.variant != []:
        df = df[df.variant.isin(args.variant)].reset_index(drop=True)

    if args.tiling != []:
        df = df[df.tiling.isin(args.tiling)].reset_index(drop=True)

    if args.tileh != []:
        df = df[df.tileh.isin(args.tileh)].reset_index(drop=True)

    if args.tilew != []:
        df = df[df.tilew.isin(args.tilew)].reset_index(drop=True)

    if 'tileh' not in args.delete and 'tilew' not in args.delete:
        if args.nb_tiles == []:
            if not ('tileh' in [args.col, args.row, args.x, args.heaty] or 'tilew' in [args.col, args.row, args.x,  args.heaty]):
                df['tile_size'] = df.tileh.map(
                    str) + '$\\times$' + df.tilew.map(str)
                del df['tileh']
                del df['tilew']
        else:
            df['nb_tileh'] = df['size'] // df['tileh']
            df['nb_tilew'] = df['size'] // df['tilew']
            df = df[df.nb_tilew.isin(args.nb_tiles)].reset_index(drop=True)
            # df = df[df.nb_tileh.isin(args.nb_tiles)].reset_index(drop=True)
            df = df[df.nb_tileh == df.nb_tilew].reset_index(drop=True)
            df['nb_tiles'] = df['nb_tileh']
            del df['tileh']
            del df['tilew']
            del df['nb_tileh']
            del df['nb_tilew']

    if df.empty:
        sys.exit("No data")

    # remove empty columns
    df = df.dropna(axis=1, how='all')
    return df.replace({None: "none"})


def savePlot(fig):
    plt.savefig(args.output, format=Path(args.output).suffix[1:])