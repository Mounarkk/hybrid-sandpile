import argparse
import pandas as pds
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import textwrap
from matplotlib.backends.backend_pdf import PdfPages

sns.set(style="darkgrid")


def openfile(path="./plots/data/perf_data.csv", sepa=";"):
    try:
        return pds.read_csv(path, sep=sepa)
    except FileNotFoundError:
        print("File not found: ", path, file=sys.stderr)
        sys.exit(1)

# Donne tous les champs de df qui ne sont pas listés


def complementaryCols(listeAttr, df):
    return [i for i in list(df.columns) if i not in listeAttr]


def constantCols(df):
    return df.columns[df.eq(df.iloc[0]).all()].tolist()


def allData(name, df):  # Donne toutes les valeurs possibles de la colonne "name" dans "df"
    return df[name].unique()


def shuffle(u, v, sepa=" "):
    s = ""
    for i in zip(u, v):
        s += str(i[0]) + "=" + str(i[1]) + sepa
    return s

# Cree les nom avec toutes les valeurs non utilisees pour le reste du graphique


def creationLegende(datasForGrapheNames, df):

    attr = complementaryCols(['time', 'ref'] + datasForGrapheNames +
                             [i for i in list(df.columns) if df[i].nunique() == 1], df)

    if attr == []:
        attr = ['kernel']

    df['legend'] = df[attr].apply(
        lambda row: shuffle(attr, row.values.tolist()), axis=1)

# Donne les valeurs constantes de "df" a travers un dictionnaire


def texteParametresConstants(df, texte=""):
    const = constantCols(df)
    string = ""
    # Donne les valeurs des constantes dans l'ordre des colonnes
    dataConst = [df[i][0] for i in const]
    for i in range(len(const)):
        string = string + str(const[i]) + "=" + str(dataConst[i]) + " "
    return textwrap.fill(texte + " " + string, width=90)


def nombreParametresConstants(df):
    return len(constantCols(df))


def creationRefSpeedUp(df):  # Automatise la creation du speedup
    df['ref'] = 0
    for i in (allData('dim', df)):
        for ite in (allData('iterations', df)):
            for ker in (allData('kernel', df)):
                valPerf = df[(df.threads == 1) & (df.dim == i) & (
                    df.iterations == ite) & (df.kernel == ker)]['time']
                df.loc[(df.dim == i) & (df.iterations == ite) &
                       (df.kernel == ker), 'ref'] = valPerf.min()
    df['speedup'] = df['ref'] / df['time']


def creerGraphique(df,
                   x='threads',
                   y='time',
                   col='dim',
                   row='iterations',
                   plottype='lineplot',
                   yscale='linear',
                   height=5,
                   showTitle=True):

    if (df['label'] == 'unlabelled').all:
        del df['label']
    constNum = nombreParametresConstants(df)
    datasForGrapheNames = [x, y, col, row]
    creationLegende(datasForGrapheNames, df)

    df = df.sort_values(by=y, ascending=False)

    g = sns.FacetGrid(df, row=row, col=col, hue="legend",
                      height=height, margin_titles=True)
    if (plottype == 'lineplot'):
        g.map(sns.lineplot, x, y, err_style="bars",
              marker="o")
    elif (plottype == 'barplot'):
        g.map(sns.barplot, x, y)
    else:
        print("Chose between 'lineplot' and 'barplot'")
        return 0
    g.set(yscale=yscale)
    g.add_legend()
    if showTitle:
        plt.subplots_adjust(top=0.85)
        if constNum == 0:
            titre = (u'Courbe de {y} en fonction de {x}').format(x=x, y=y)
        else:
            titre = (u'Courbe de {y} en fonction de {x}\n {cons}').format(
                x=x, y=y, cons=texteParametresConstants(df, "Paramètre :" if constNum == 1 else "Paramètres :"))
        g.fig.suptitle(titre)
    return g


def parserArguments(argv):
    global parser, args
    parser = argparse.ArgumentParser(
        argv, description='Process performance plots')

    parser.add_argument(
        "-x", choices=["threads", "dim", "iterations", "grain"], default="threads")
    parser.add_argument(
        "-y", choices=["time", "speedup", "throughput"], default="speedup")
    parser.add_argument(
        "-C", "--col", choices=["dim", "iterations", "kernel", "variant", "grain"], default=None)
    parser.add_argument(
        "-R", "--row", choices=["dim", "iterations", "kernel", "variant", "grain"], default=None)

    parser.add_argument('-of', '--output',
                        action='store', nargs='?',
                        help='Filename to output the plot',
                        const='plot.pdf',
                        default='plot.pdf')

    parser.add_argument('-if', '--input',
                        action='store', nargs='?',
                        help="Data's filename",
                        const=os.getcwd() + "/plots/data/perf_data.csv",
                        default=os.getcwd() + "/plots/data/perf_data.csv")

    parser.add_argument('-k', '--kernel',
                        action='store', nargs='+',
                        help="list of kernels to plot",
                        default="")

    parser.add_argument('-t', '--threads',
                        action='store', nargs='+',
                        help="list of numbers of threads to plot",
                        default="")

    parser.add_argument('-v', '--variant',
                        action='store', nargs='+',
                        help="list of variants to plot",
                        default="")

    parser.add_argument('-g', '--grain',
                        action='store', nargs='+',
                        help="list of grains to plot",
                        default="")

    parser.add_argument('-i', '--iterations',
                        action='store', nargs='+',
                        help="list of iterarations to plot",
                        default="")

    parser.add_argument('-d', '--dim',
                        action='store', nargs='+',
                        help="list of sizes to plot",
                        default="")

    parser.add_argument('-lb', '--label',
                        action='store', nargs='+',
                        help="list of labels to plot",
                        default="")

    parser.add_argument('--height',
                        action='store',
                        type=int,
                        help="to set the height of each subgraph",
                        default=4)

    parser.add_argument('--showTitle',
                        action='store_true',
                        help="to print title",
                        default=False)

    parser.add_argument('--yscale',
                        choices=["linear", "log", "symlog", "logit"],
                        action='store',
                        default="linear")

    parser.add_argument('--plottype',
                        choices=['lineplot', 'barplot'],
                        action='store',
                        default="lineplot")

    args = parser.parse_args()

    return args


def lireDataFrame(args):

    # Lecture du fichier d'experiences:
    df = openfile(args.input, sepa=";")

    if args.kernel != "":
        df = df[df.kernel.isin(args.kernel)].reset_index(drop=True)

    if args.iterations != "":
        df = df[df.iterations.isin(args.iterations)].reset_index(drop=True)

    if args.dim != "":
        df = df[df.dim.isin(args.dim)].reset_index(drop=True)

    if args.y == "speedup":
        creationRefSpeedUp(df)

    if args.label != "":
        df = df[df.label.isin(args.label)].reset_index(drop=True)

    if args.threads != "":
        df = df[df.threads.isin(args.threads)].reset_index(drop=True)

    if args.variant != "":
        df = df[df.variant.isin(args.variant)].reset_index(drop=True)

    if args.grain != "":
        df = df[df.grain.isin(args.grain)].reset_index(drop=True)

    return df


def engeristrerGraphique(fig):
    pp = PdfPages(args.output)
    plt.savefig(pp, format='pdf')
    pp.close()
