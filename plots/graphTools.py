# coding: 8859
import warnings

# warnings.filterwarnings("error")
# warnings.simplefilter(action='ignore', category=FutureWarning)

from updateParamaters import updateFunParameters, searchParameter

import argparse
import pandas as pds
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import os
from pathlib import Path
import sys
import textwrap
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator


def verbose(text, arg):
    global args
    if args.verbose:
        print("--------------" + text + "-------------------")
        print(arg)


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
            lambda row: shuffle(funcDeps, row.values.tolist(), "  ", " = "), axis=1
        )
        for col in funcDeps:
            del df[col]
        df[axisCol] = df[axisCol].map(str) + " |" + df["tmp"]
        del df["tmp"]


def decalerLabels(g):
    # Récupérez l'axe x actuel
    for ax in g.axes.flat:
        positions_ticks = ax.get_xticks()
        ax.xaxis.set_major_locator(FixedLocator(positions_ticks))
        # Créez une liste personnalisée des étiquettes de ticks
        xtick_labels = [label.get_text() for label in ax.get_xticklabels()]
        # Décalez vers le bas les étiquettes de ticks sur les positions impaires
        for i in range(1, len(xtick_labels), 2):
            xtick_labels[i] = "\n" + xtick_labels[i]
        # Remplacez les étiquettes de ticks actuelles par les étiquettes personnalisées
        ax.set_xticklabels(xtick_labels)


def computeTitleAndLegend(args, df):
    axisAttr = [args.x, args.col, args.row, args.heaty] + args.all_y
    constantParameters = constantCols(df)
    if len(constantParameters) == 0:
        title = ("Courbe de {y} en fonction de {x}").format(x=args.x, y=args.all_y)
    else:
        title = ("{cons}").format(
            x=args.x,
            y=args.all_y,
            cons=textForConstantParameters(df, constantParameters),
        )
        for col in constantParameters:
            if not col in axisAttr:
                del df[col]

    factorizeLegendColRow(args.col, args, axisAttr, df)
    factorizeLegendColRow(args.row, args, axisAttr, df)

    attr = complementaryCols(["time", "ref"] + axisAttr, df)

    if attr != []:
        df["legend"] = df[attr].apply(
            lambda row: shuffle(attr, row.values.tolist()), axis=1
        )
        for col in attr:
            del df[col]
    return title


# Cree les nom avec toutes les valeurs non utilisees pour le reste du graphique


def textForConstantParameters(df, constantParameters):
    string = ""
    # Donne les valeurs des constantes dans l'ordre des colonnes
    dataConst = [df[i].iloc[0] for i in constantParameters]
    for i in range(len(constantParameters)):
        if dataConst[i] != "none" and str(dataConst[i]) != "nan":
            string = string + str(constantParameters[i]) + "=" + str(dataConst[i]) + " "
    return string


def computeSpeedUpAttr(df, args):  # Automatise la creation du speedup
    group = ["machine", "size", "kernel", "iterations", "arg"]
    group = [
        attr for attr in group if attr not in args.delete and attr in list(df.columns)
    ]
    if args.RefTimeVariants == [] and args.RefTimeTiling == []:
        refDF = df[df.threads == 1].reset_index(drop=True)
    elif args.RefTimeTiling == []:
        refDF = df[df.variant.isin(args.RefTimeVariants) & df.threads == 1].reset_index(
            drop=True
        )
    elif args.RefTimeVariants == []:
        refDF = df[df.tiling.isin(args.RefTimeTiling) & df.threads == 1].reset_index(
            drop=True
        )
    else:
        refDF = df[
            df.variant.isin(args.RefTimeVariants)
            & df.tiling.isin(args.RefTimeTiling)
            & df.threads
            == 1
        ].reset_index(drop=True)
    if refDF.empty:
        sys.exit("No row with OMP_NUM_THREADS=1 to compute speedUP")

    for i in complementaryCols(group + ["time"], refDF):
        del refDF[i]

    refDF = refDF.loc[
        refDF.groupby(group).time.idxmin()
    ]  # .reset_index(drop=True, inplace=True)
    refDF = refDF.rename(columns={"time": "refTime"})
    verbose("Speed-up references ", refDF)

    df = df.merge(refDF, how="inner")

    df["speedup"] = df["refTime"] / df["time"]

    if args.noRefTime:
        del df["refTime"]
    else:
        df["refTime"] = df["refTime"] // 1000
    return df


def heatFacet(*args, **kwargs):
    data = kwargs["data"]
    data = data.pivot(index=args[1], columns=args[0], values=args[2])
    verbose("heatmap data", data)
    m = data.max().max()
    if "time (ms)" == args[2]:
        fmt = ".2f" if m < 10 else ".1f" if m < 100 else ".0f"
        g = sns.heatmap(
            data, cmap="rocket_r", annot=True, fmt=fmt, annot_kws={"fontsize": 8}
        )
    else:
        fmt = ".2f" if m < 1 else ".1f" if m < 10 else ".0f"
        g = sns.heatmap(
            data, cmap="rocket", annot=True, fmt=fmt, annot_kws={"fontsize": 8}
        )
    g.invert_yaxis()
    plt.yticks(rotation=0)


axes = {}


def twin_lineplot(x, y, **kwargs):
    ax = plt.gca()
    mini = kwargs.pop("mini")
    maxi = kwargs.pop("maxi")
    if "label" in kwargs.keys():
        kwargs.pop("label")
    # if ax.xaxis not in axes.keys(): # utilité du test ???
    axes[ax.axis] = ax.twinx()
    axes[ax.axis].set(ylim=(mini, maxi))

    if args.y2scale == "log2":
        axes[ax.axis].set_yscale("log", base=2)
    elif args.y2scale != "linear":
        axes[ax.axis].set_yscale(args.y2scale)

    sns.lineplot(x=x, y=y, ax=axes[ax.axis], label="_nolegend_", **kwargs)
    axes[ax.axis].set_ylabel(y.name)
    axes[ax.axis].grid(visible=None)
    axes[ax.axis].tick_params(axis="y", labelsize=8)


def add_legend_and_labels(g, args, linestyle_code):
    g.set_axis_labels(
        x_var=args.x,
        y_var=args.y[0] if args.ylabel == None else args.ylabel,
        clear_inner=True,
    )  #
    g.add_legend()
    l = g._legend.legendHandles
    for ax in g.axes.flat:
        if ax.texts:
            txt = ax.texts[0]
            ax.text(
                txt.get_unitless_position()[0] + 0.1,
                txt.get_unitless_position()[1],
                txt.get_text(),
                transform=ax.transAxes,
                va="center",
                fontsize="medium",
                rotation=-90,
            )
            ax.texts[0].remove()


linestyle_code = [
    (0, ()),  # solid
    (0, (1, 1)),  # dotted
    (0, (5, 1)),  # dashed
    (0, (3, 1, 1, 1)),  # dashdot
    (0, (3, 1, 1, 1, 1, 1)),  # dashdotdot
    (0, (1, 10)),  # loosely dotted
    (0, (5, 10)),  # loosely dashed
    (0, (3, 10, 1, 10)),  # loosely dashdotted
    (0, (3, 10, 1, 10, 1, 10)),  # loosely dashdotdotted
]


def single_entry_lineplots(args, df, g):
    warnings.simplefilter(action="ignore", category=RuntimeWarning)
    warnings.simplefilter(action="ignore", category=FutureWarning)

    # Avoid "RuntimeWarning: All-NaN axis encountered"
    # Indeed using the parameter err_style='bars' generates warnings when there is only one value
    g.map(
        sns.lineplot,
        args.x,
        "value",
        linestyle="solid",
        err_style="bars",
        marker=Line2D.filled_markers[1],
    )
    warnings.simplefilter(action="default", category=FutureWarning)
    warnings.simplefilter(action="default", category=RuntimeWarning)

    linestyle_code = [(0, ())]
    add_legend_and_labels(g, args, linestyle_code)

    if sns.__version__ >= "0.11.2":
        sns.move_legend(g, "center right")

    return g


def single_entry_2_scales_lineplots(args, df, g):
    custom_palette = sns.hls_palette(len(args.y) + len(args.y2))
    warnings.simplefilter(action="ignore", category=RuntimeWarning)
    warnings.simplefilter(action="ignore", category=FutureWarning)

    # Avoid "RuntimeWarning: All-NaN axis encountered"
    # Indeed using the parameter err_style='bars' generates warnings when there is only one value
    for i in range(len(args.y)):
        g.map(
            sns.lineplot,
            args.x,
            args.y[i],
            color=custom_palette[i],
            err_style="bars",
            marker=Line2D.filled_markers[1],
        )
    delta = (df[args.y2].max().max() - df[args.y2].min().min()) / 20
    mini = df[args.y2].min().min() - delta
    maxi = df[args.y2].max().max() + delta

    for i in range(len(args.y2)):
        g.map(
            twin_lineplot,
            args.x,
            args.y2[i],
            err_style="bars",
            marker=Line2D.filled_markers[14],
            color=custom_palette[len(args.y) + i],
            mini=mini,
            maxi=maxi,
        )
    warnings.simplefilter(action="default", category=RuntimeWarning)
    warnings.simplefilter(action="default", category=FutureWarning)

    add_legend_and_labels(g, args, linestyle_code)

    if len(args.all_y) > 1:
        for i in range(len(args.all_y)):
            g._legend.legendHandles.append(
                mlines.Line2D(
                    [], [], color=custom_palette[i], label=" " + args.all_y[i]
                )
            )
            g._legend.texts.append(plt.text(0, 0, " " + args.all_y[i], visible=False))

    sns.move_legend(g, "center right")
    return g


def multiple_entries_lineplots(args, df, g):
    warnings.simplefilter(action="ignore", category=RuntimeWarning)
    warnings.simplefilter(action="ignore", category=FutureWarning)

    # Avoid "RuntimeWarning: All-NaN axis encountered"
    # Indeed using the parameter err_style='bars' generates warnings when there is only one value
    for i in range(len(args.y)):
        g.map(
            sns.lineplot,
            args.x,
            args.y[i],
            linestyle=linestyle_code[i],
            err_style="bars",
            marker=Line2D.filled_markers[1],
        )
    delta = (df[args.y2].max().max() - df[args.y2].min().min()) / 20
    mini = df[args.y2].min().min() - delta
    maxi = df[args.y2].max().max() + delta
    for i in range(len(args.y2)):
        g.map(
            twin_lineplot,
            args.x,
            args.y2[i],
            err_style="bars",
            marker=Line2D.filled_markers[14],
            linestyle=linestyle_code[len(args.y) + i],
            mini=mini,
            maxi=maxi,
        )
    warnings.simplefilter(action="default", category=RuntimeWarning)
    warnings.simplefilter(action="default", category=FutureWarning)

    add_legend_and_labels(g, args, linestyle_code)

    if len(args.all_y) > 1:
        for i in range(len(args.all_y)):
            g._legend.legendHandles.append(
                mlines.Line2D(
                    [],
                    [],
                    color="black",
                    linestyle=linestyle_code[i],
                    label=" " + args.all_y[i],
                )
            )
            g._legend.texts.append(plt.text(0, 0, " " + args.all_y[i], visible=False))

    sns.move_legend(g, "center right")
    return g


def easyPlotDataFrame(df, args):
    with sns.plotting_context("paper", font_scale=args.fontScale):
        return pc_easyPlotDataFrame(df, args)


def pc_easyPlotDataFrame(df, args):
    title = computeTitleAndLegend(args, df) + "(" + str(df.shape[0]) + " exp.)"
    verbose("Constant parameters", title)

    if not args.noSort:
        df = df.sort_values(by=args.all_y[0], ascending=False)

    verbose("Data frame with legend", df)

    legend = "legend" if "legend" in list(df.columns) else None
    nbLegendEntries = 1 if legend is None else len(df[legend].unique())

    if args.plottype == "lineplot":
        if nbLegendEntries == 1 and args.y2 == []:
            df = pds.melt(
                df,
                var_name="legend",
                value_name="value",
                id_vars=[x for x in [args.x, args.col, args.row] if x is not None],
                value_vars=args.y,
            )
            legend = "legend"
            verbose("Data frame melted", df)
        
        kwargs, args.unknown_args = updateFunParameters(
            sns.FacetGrid,
            args.unknown_args,
            sharex=args.row != None,
            sharey=args.col != None,
            margin_titles=True,
            legend_out=True,
        )
        verbose("FacetGrit kwargs", kwargs)
        g = sns.FacetGrid(df, hue=legend, **kwargs)
        if nbLegendEntries == 1 and args.y2 == []:
            g = single_entry_lineplots(args, df, g)
        elif nbLegendEntries == 1:
            g = single_entry_2_scales_lineplots(args, df, g)
        else:
            g = multiple_entries_lineplots(args, df, g)

    elif args.plottype == "heatmap":
        df = df.groupby(complementaryCols(args.y, df), as_index=False).mean()
        kwargs, args.unknown_args = updateFunParameters(
            sns.FacetGrid,
            args.unknown_args,
            sharex=args.row != None,
            sharey=args.col != None,
            margin_titles=True,
            legend_out=True,
        )
        verbose("FacetGrit kwargs", kwargs)

        g = sns.FacetGrid(df, hue=legend, **kwargs)
        g = g.map_dataframe(heatFacet, args.x, args.heaty, args.y[0])

    else:
        kwargs, args.unknown_args = updateFunParameters(
            sns.catplot,
            args.unknown_args,
            kind="swarm",
            margin_titles=True,
            legend_out=True,
            sharex=args.row != None,
            sharey=args.col != None,
        )
        verbose("catplot kwargs", kwargs)

        g = sns.catplot(
            data=df,
            x=args.x,
            y=args.y[0],
            hue=legend,
            palette=random.shuffle(sns.hls_palette(nbLegendEntries)),
            **kwargs
        )

    if args.comment != None:
        title += args.comment
    if not args.hideParameters:
        g.fig.suptitle(
            title, wrap=True, y=0.999
        )  # wrap does not perform well whenever y=1 !
        plt.subplots_adjust(top=args.adjustTop)
    else:
        print(title)

    for ax in g.axes.flat:
        if args.yscale == "log2":
            ax.set_yscale("log", base=2)
        elif args.yscale != None:
            ax.set_yscale(args.yscale)
        if args.xscale == "log2":
            ax.set_xscale("log", base=2)
        elif args.xscale != None:
            ax.set_xscale(args.xscale)

    if args.plottype == "heatmap":
        g.tight_layout(rect=[0, 0.03, 1, args.adjustTop])
    elif nbLegendEntries == 1 and args.y2 != []:
        g.tight_layout(rect=[0, 0.03, min(0.6 + (args.aspect * 0.07), 0.95), 1])
    else:
        g.tight_layout()

    if args.unknown_args != []:
        print("Warning unused arguments :", args.unknown_args, file=sys.stderr)
    return g


def listColumns(csvFileName):
    import csv

    try:
        with open(csvFileName, "r") as csvFile:
            reader = csv.reader(csvFile, delimiter=";")
            columns = next(reader)
        return columns
    except FileNotFoundError:
        print(csvFileName, ": file not found.", file=sys.stderr)
        sys.exit(1)


perfAttr = None


class CustomHelpFormatter(argparse.HelpFormatter):
    def _split_lines(self, text, width):
        if text.startswith("R|"):
            return text[2:].splitlines()
        return super()._split_lines(text, width)


def parseArguments(argv, derived_perf={}, derived_attr={}):
    global parser, args, perfAttr
    ### first get the input filename
    parseOnlyInputFile = argparse.ArgumentParser(argv, add_help=False)
    parseOnlyInputFile.add_argument(
        "-if",
        "--input",
        action="store",
        nargs="?",
        help="Data's filename",
        default=os.getcwd() + "/data/perf/data.csv",
    )
    parseOnlyInputFile.add_argument( # => -i is not an abbrev. of -if 
        "-i", action="store", type=int, nargs="+", default=[]
    )

    args, unknown_args = parseOnlyInputFile.parse_known_args()

    all = (
        ["nb_tiles", "tile_size"]
        + list(derived_attr.keys())
        + listColumns(args.input)
        + list(derived_perf.keys())
    )

    index_time = all.index(
        "time"
    )  # All attributes located after 'time' must refer to measurements.
    perfAttr = all[index_time:]

    parser = argparse.ArgumentParser(
        argv,
        formatter_class=CustomHelpFormatter,
        description="""Process performance plots.
        The labels of the options are similar to those of
        easypap for all that relates to data selection,
        and to those of seaborn for all aspects of graphic
        layout https://seaborn.pydata.org/introduction.html""",
    )

    files = parser.add_argument_group("Files")
    plot = parser.add_argument_group(
        """Three kinds of graphics are offered: lineplot, catplot and heatmap. 
    
Catplot and lineplot graphs can be customized through arguments that correspond to those of seaborn's catplot and, in the case of lineplot, facetgrid methods. 
Arguments must be passed in the form of an "arg=value" assignment. For example, to create a multi-plot grid where graphics share the same size on each row and the same variant on each column, simply define the "row=size col=variant" variables.
Please note that it might be necessary to use -- to separate a list of arguments placed before another option. For example:  --size 512 1024 -- row=none.

The facetgrid's arguments [https://seaborn.pydata.org/generated/seaborn.facegrid.html] : 
col_wrap=None, sharex=True, sharey=True, height=3, aspect=1, palette=None, row_order=None, col_order=None, hue_order=None, hue_kws=None, dropna=False, legend_out=True, despine=True, margin_titles=False, xlim=None, ylim=None.

Catplot's arguments [https://seaborn.pydata.org/generated/seaborn.catplot.html]:
 row=None, col=None, kind=strip, estimator=mean, errorbar=('ci', 95), n_boot=1000, units=None, seed=None, order=None, hue_order=None, row_order=None, col_order=None, col_wrap=None, height=5, aspect=1, log_scale=None, native_scale=False, formatter=None, orient=None, color=None, palette=None, hue_norm=None, legend=auto, legend_out=True, sharex=True, sharey=True, margin_titles=False"""
    )
    data = parser.add_argument_group("Data selection")
    presentation = parser.add_argument_group("Display")
    delete = parser.add_argument_group("Delete : to delete attributes")
    axes = parser.add_argument_group("Axes")

    files.add_argument(
        "-if",
        "--input",
        action="store",
        nargs="?",
        help="Data's filename",
        default=os.getcwd() + "/data/perf/data.csv",
    )

    delete.add_argument("--delete", action="store", nargs="*", choices=all, default=[])

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="turn on verbose mode",
        default=False,
    )

    axes.add_argument("-x", "-heatx", choices=all, default="threads")
    axes.add_argument("-heaty", choices=all, default=None)
    axes.add_argument(
        "-y",
        action="store",
        nargs="*",
        choices=perfAttr + ["speedup", "throughput", "efficiency"],
        default=["speedup"],
    )

    axes.add_argument("--ylabel", action="store", nargs="?", default=None)

    axes.add_argument(
        "-y2",
        action="store",
        nargs="*",
        choices=perfAttr + ["speedup", "throughput", "efficiency"],
        default=[],
    )

    axes.add_argument(
        "-rtv",
        "--RefTimeVariants",
        action="store",
        nargs="*",
        help="list of variants to take into account to compute the speedUP RefTimes",
        default=[],
    )

    axes.add_argument(
        "-rtt",
        "--RefTimeTiling",
        action="store",
        nargs="*",
        help="list of tiling functions to take into account to compute the speedUP RefTimes",
        default=[],
    )

    files.add_argument(
        "-of",
        "--output",
        action="store",
        nargs="?",
        help="Filename to output the plot (supported formats: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff)",
        default=os.path.splitext(args.input)[0] + ".pdf",
    )

    int_arguments = [
        ("-i", "--iterations"),
        ("-s", "--size"),
        ("-t", "--threads"),
        ("-th", "--tileh"),
        ("-tw", "--tilew"),
        ("-nt", "--nb_tiles"),
    ]

    for opt in int_arguments:
        data.add_argument(*opt, action="store", type=int, nargs="+", default=[])

    str_arguments = [
        ("-k", "--kernel"),
        ("-v", "--variant"),
        ("-wt", "--tiling"),
        ("-m", "--machine"),
        ("-sc", "--schedule"),
        ("-lb", "--label"),
        ("--arg",),
        ("--places",),
    ]

    for opt in str_arguments:
        data.add_argument(*opt, action="store", nargs="+", default=[])

    presentation.add_argument(
        "--hideParameters",
        action="store_true",
        help="to hide constant parameters",
        default=False,
    )

    axes.add_argument(
        "--noSort", action="store_true", help="sort data following y", default=False
    )

    presentation.add_argument(
        "--adjustTop",
        action="store",
        type=float,
        help="to adjust the space for the suptitle",
        default=1,
    )

    presentation.add_argument(
        "--fontScale",
        action="store",
        type=float,
        help="to adjust the font of the title and the legend",
        default=1.0,
    )

    presentation.add_argument(
        "--noRefTime",
        action="store_true",
        help="do not print reftime in legend",
        default=False,
    )

    for opt in ["--yscale", "--y2scale"]:
        axes.add_argument(
            opt,
            choices=["linear", "log", "log2", "symlog", "logit"],
            action="store",
            default=None,
        )

    axes.add_argument(
        "--xscale",
        choices=["linear", "log", "log2", "symlog", "logit"],
        action="store",
        default=None,
    )

    plot.add_argument(
        "--plottype",
        choices=["lineplot", "catplot", "heatmap"],
        action="store",
        default="lineplot",
    )

    presentation.add_argument(
        "--comment", action="store", help="add some text to the title", default=None
    )

    args = args, unknown_args = parser.parse_known_args()

    args.derived_perf = derived_perf
    args.derived_attr = derived_attr
    args.unknown_args = unknown_args
    args.row = searchParameter("row", unknown_args)
    args.col = searchParameter("col", unknown_args)
    args.aspect = searchParameter("aspect", unknown_args, 1)
    return args


def substitute_attr_y(args, src, dst):
    if src in args.y:
        i = args.y.index(src)
        args.y = args.y[:i] + [dst] + args.y[i + 1 :]
    if src in args.y2:
        i = args.y2.index(src)
        args.y2 = args.y2[:i] + [dst] + args.y2[i + 1 :]


def selectRows(filters, df):
    condition = pds.Series(True, index=df.index)
    for column, values in filters.items():
        if values:
            condition &= df[column].isin(values)
    df = df.loc[condition].reset_index(drop=True)
    return df


def getDataFrame(args):
    # Lecture du fichier d'experiences:
    df = openfile(args.input, sepa=";")

    verbose("Initial data frame", df)
    # Do not delete row that may contain sequential best perf
    filters = {
        "kernel": args.kernel,
        "arg": args.arg,
        "iterations": args.iterations,
        "size": args.size,
        "machine": args.machine,
    }
    df = selectRows(filters, df)
    df = df.assign(**args.derived_attr)
    df = df.assign(**args.derived_perf)

    if args.delete != []:
        for attr in args.delete:
            del df[attr]

    args.all_y = args.y + args.y2

    if "speedup" in args.all_y:
        df = computeSpeedUpAttr(df, args)

    if "efficiency" in args.all_y:
        if "speedup" not in args.all_y:
            df = computeSpeedUpAttr(df, args)
            df["efficiency"] = df["speedup"] / df["threads"]
            del df["speedup"]
        else:
            df["efficiency"] = df["speedup"] / df["threads"]

    if "throughput" in args.all_y:
        substitute_attr_y(args, "throughput", "throughput (MPixel / s)")
        df["throughput (MPixel / s)"] = (
            (df["size"] ** 2) * df["iterations"] / df["time"]
        )

    for attr in perfAttr:
        if attr not in args.all_y:
            del df[attr]

    if "time" in args.all_y:
        df["time"] = df["time"] / 1000
        df.rename(columns={"time": "time (ms)"}, inplace=True)
        substitute_attr_y(args, "time", "time (ms)")

    args.all_y = args.y + args.y2

    filters = {
        "label": args.label,
        "schedule": args.schedule,
        "places": args.places,
        "threads": args.threads,
        "variant": args.variant,
        "tiling": args.tiling,
        "tileh": args.tileh,
        "tilew": args.tilew,
    }
    df = selectRows(filters, df)

    if "tileh" not in args.delete and "tilew" not in args.delete:
        if args.nb_tiles == []:
            if not (
                "tileh" in [args.col, args.row, args.x, args.heaty]
                or "tilew" in [args.col, args.row, args.x, args.heaty]
            ):
                df["tile_size"] = (
                    df.tileh.astype(str) + "$\\times$" + df.tilew.astype(str)
                )
                del df["tileh"]
                del df["tilew"]
        else:
            df["nb_tileh"] = df["size"] // df["tileh"]
            df["nb_tilew"] = df["size"] // df["tilew"]
            df[df.nb_tilew.isin(args.nb_tiles)].reset_index(drop=True, inplace=True)
            df[df.nb_tileh == df.nb_tilew].reset_index(drop=True, inplace=True)
            df["nb_tiles"] = df["nb_tileh"]
            del df["tileh"]
            del df["tilew"]
            del df["nb_tileh"]
            del df["nb_tilew"]

    if df.empty:
        sys.exit("No data")

    # remove empty columns
    df = df.dropna(axis=1, how="all")
    verbose("Selected data frame", df)

    return df.replace({None: "none"})


def savePlot(fig):
    legend = fig._legend

    plt.savefig(
        args.output, format=Path(args.output).suffix[1:], bbox_extra_artists=(legend)
    )
    print("Graph saved in", args.output)
