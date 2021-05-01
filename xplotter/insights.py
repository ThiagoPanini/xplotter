"""
---------------------------------------------------
---------------- MODULE: Insights -----------------
---------------------------------------------------
In this module, the user will find some useful
functions for formatting matplotlib and seaborn
plots, like customizing borders and putting labels
on data

Table of Contents
---------------------------------------------------
1. Initial setup
    1.1 Importing libraries
    1.2 Auxiliar functions
2. Custom graphical analysis
    2.1 Donut and pie charts
    2.2 Simple and percentual countplots
    2.3 Distribution plot
    2.4 Aggregation approach
    2.5 General overview and correlation matrix
    2.6 Multiple plots at one function
    2.7 Line charts for evolution plots
---------------------------------------------------
"""

# Author: Thiago Panini
# Data: 29/04/2021


"""
---------------------------------------------------
---------------- 1. INITIAL SETUP -----------------
             1.1 Importing libraries
---------------------------------------------------
"""

# Standard libraries
import os
import pandas as pd
import numpy as np
from datetime import datetime
from math import ceil
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

from xplotter.formatter import make_autopct, format_spines, AnnotateBars


"""
---------------------------------------------------
---------------- 1. INITIAL SETUP -----------------
             1.2 Auxiliar functions
---------------------------------------------------
"""

# Saving figures generated from matplotlib
def save_fig(fig, output_path, img_name, tight_layout=True, dpi=300):
    """
    Saves figures created from matplotlib/seaborn

    Parameters
    ----------
    :param fig: figure created using matplotlib [type: plt.figure]
    :param output_file: path for image to be saved (path + filename in png format) [type: string]
    :param tight_layout: flag for tighting figure layout before saving it [type: bool, default=True]
    :param dpi: image resolution [type: int, default=300]

    Return
    ------
    This function returns nothing besides the image saved on the given path

    Application
    ---------
    fig, ax = plt.subplots()
    save_fig(fig, output_file='image.png')
    """

    # Searching for the existance of the directory
    if not os.path.isdir(output_path):
        print(f'Directory {output_path} not exists. Creating a directory on the given path')
        try:
            os.makedirs(output_path)
        except Exception as e:
            print(f'Error on creating the directory {output_path}. Exception: {e}')
            return
    
    # Tighting layout
    if tight_layout:
        fig.tight_layout()
    
    try:
        output_file = os.path.join(output_path, img_name)
        fig.savefig(output_file, dpi=300)
    except Exception as e:
        print(f'Error on saving image. Exception: {e}')
        return



"""
---------------------------------------------------
---------- 2. CUSTOM GRAPHICAL ANALYSIS -----------
            2.1 Donut and pie charts
---------------------------------------------------
"""

def plot_donut_chart(df, col, **kwargs):
    """
    Creates a custom donut chart for a specific column
    
    Parameters
    ----------
    :param df: dataset used for plotting [type: pd.DataFrame]
    :param col: column name to be plotted [type: string]
    :param **kwargs: additional parameters
        :arg figsize: figure dimension [type: tuple, default=(8, 8)]
        :arg ax: matplotlib axis in case of external figure defition [type: mpl.Axes, default=None]
        :arg circle_radius: central circle radius of the chart [type: float, default=0.8]
        :arg circle_radius_color: central circle color of the chart [type: string, default='white']
        :arg label_names: custom labels [type: dict, default=value_counts().index]
        :arg top: filter the top N categories on the chart [type: int]
        :arg colors: color list for customizing the chart [type: list]
        :arg text: text string to be put on central circle of the chart [type: string, default=f'Total: \n{sum(values)}']
        :arg title: chart title [type: string, default=f'Donut Chart for Feature {col}']
        :arg autotexts_size: label size from the numerical value [type: int, default=14]
        :arg autotexts_color: label color from the numerical value [type: int, default='black']
        :arg texts_size: label size from the chart [type: int, default=14]
        :arg texts_color: label color from the chart [type: int, default='black']
        :arg save: flag for saving the image created [type: bool, default=None]
        :arg output_path: path for image to be saved [type: string, default='output/']
        :arg img_name: filename for image to be saved [type: string, default=f'{col}_donutchart.png']
    
    Return
    ------
    This function returns nothing besides the custom donut chart

    Application
    -----------
    plot_donut_chart(df=df, col='categorical_column', label_names={1: 'Class 1', 2: 'Class 2'})
    """
    
    # Validating column name on the given dataset
    if col not in df.columns:
        print(f'There is no column {col} on the given dataset')
        return

    # Returning values and labels for plotting
    counts = df[col].value_counts()
    values = counts.values
    labels = counts.index
    if 'label_names' in kwargs:
        try:
            labels = labels.map(kwargs['label_names'])
        except Exception as e:
            print(f'Error on mapping the dictionary label_names on column {col}. Exception: {e}')

    # Verifying top_n category filter
    if 'top' in kwargs and kwargs['top'] > 0:
        values = values[:-kwargs['top']]
        labels = labels[:-kwargs['top']]
    
    # Chart colors
    color_list = ['darkslateblue', 'crimson', 'lightseagreen', 'lightskyblue', 'lightcoral', 'silver']
    colors = kwargs['colors'] if 'colors' in kwargs else color_list[:len(labels)]

    # Setting up parameters
    figsize = kwargs['figsize'] if 'figsize' in kwargs else (8, 8)
    ax = kwargs['ax'] if 'ax' in kwargs else None
    circle_radius = kwargs['circle_radius'] if 'circle_radius' in kwargs else 0.8
    circle_radius_color = kwargs['circle_radius_color'] if 'circle_radius_color' in kwargs else 'white'

    # Plotting the donut chart
    center_circle = plt.Circle((0, 0), circle_radius, color=circle_radius_color)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors, startangle=90, autopct=make_autopct(values))
    ax.add_artist(center_circle)

    # Setting up central text
    text = kwargs['text'] if 'text' in kwargs else f'Total: \n{sum(values)}'
    text_kwargs = dict(size=20, fontweight='bold', va='center')
    ax.text(0, 0, text, ha='center', **text_kwargs)
    
    # Axis title
    title = kwargs['title'] if 'title' in kwargs else f'Donut Chart for Feature \n{col}'
    ax.set_title(title, size=16, color='dimgrey')

    # Customizing graphic created
    autotexts_size = kwargs['autotexts_size'] if 'autotexts_size' in kwargs else 14
    autotexts_color = kwargs['autotexts_color'] if 'autotexts_color' in kwargs else 'black'
    texts_size = kwargs['texts_size'] if 'texts_size' in kwargs else 14
    texts_color = kwargs['texts_color'] if 'texts_stexts_colorize' in kwargs else 'black'

    # Customizing labels
    plt.setp(autotexts, size=autotexts_size, color=autotexts_color)
    plt.setp(texts, size=texts_size, color=texts_color)

    # Saving image
    if 'save' in kwargs and bool(kwargs['save']):
        output_path = kwargs['output_path'] if 'output_path' in kwargs else 'output/'
        img_name = kwargs['img_name'] if 'img_name' in kwargs else f'{col}_donutchart.png'
        save_fig(fig=fig, output_path=output_path, img_name=img_name)

def plot_pie_chart(df, col, **kwargs):
    """
    Creates a custom pie chart for a specific column
    
    Parameters
    ----------
    :param df: dataset used for plotting [type: pd.DataFrame]
    :param col: column name to be plotted [type: string]
    :param **kwargs: additional parameters
        :arg figsize: figure dimension [type: tuple, default=(8, 8)]
        :arg ax: matplotlib axis in case of external figure defition [type: mpl.Axes, default=None]
        :arg circle_radius: central circle radius of the chart [type: float, default=0.8]
        :arg circle_radius_color: central circle color of the chart [type: string, default='white']
        :arg label_names: custom labels [type: dict, default=value_counts().index]
        :arg top: filter the top N categories on the chart [type: int]
        :arg colors: color list for customizing the chart [type: list]
        :arg title: chart title [type: string, default=f'Donut Chart for Feature {col}']
        :arg autotexts_size: label size from the numerical value [type: int, default=14]
        :arg autotexts_color: label color from the numerical value [type: int, default='black']
        :arg texts_size: label size from the chart [type: int, default=14]
        :arg texts_color: label color from the chart [type: int, default='black']
        :arg save: flag for saving the image created [type: bool, default=None]
        :arg output_path: path for image to be saved [type: string, default='output/']
        :arg img_name: filename for image to be saved [type: string, default=f'{col}_piechart.png']
    
    Return
    ------
    This function returns nothing besides the custom pie chart

    Application
    -----------
    plot_pie_chart(df=df, col='categorical_column', label_names={1: 'Class 1', 2: 'Class 2'})
    """

    # Validating column name on the given dataset
    if col not in df.columns:
        print(f'There is no column {col} on the given dataset')
        return

    # Returning labels ans values
    counts = df[col].value_counts()
    values = counts.values
    labels = counts.index
    if 'label_names' in kwargs:
        try:
            labels = labels.map(kwargs['label_names'])
        except Exception as e:
            print(f'Error on mapping the dict label_names on column {col}. Exception: {e}')

    # Filtering top N categories if applicable
    if 'top' in kwargs and kwargs['top'] > 0:
        values = values[:-kwargs['top']]
        labels = labels[:-kwargs['top']]
    
    # Colors for the chart
    color_list = ['darkslateblue', 'crimson', 'lightseagreen', 'lightskyblue', 'lightcoral', 'silver']
    colors = kwargs['colors'] if 'colors' in kwargs else color_list[:len(labels)]

    # Setting up parameters
    figsize = kwargs['figsize'] if 'figsize' in kwargs else (8, 8)
    ax = kwargs['ax'] if 'ax' in kwargs else None
    explode = kwargs['explode'] if 'explode' in kwargs else (0,) * len(labels)
    shadow = kwargs['shadow'] if 'shadow' in kwargs else False

    # Plotting pie chart
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors, autopct=make_autopct(values), 
                                      startangle=90, explode=explode, shadow=shadow)
    
    # Defining title
    title = kwargs['title'] if 'title' in kwargs else f'Pie Chart for Feature \n{col}'
    ax.set_title(title, size=16, color='dimgrey')

    # Customizing the chart
    autotexts_size = kwargs['autotexts_size'] if 'autotexts_size' in kwargs else 14
    autotexts_color = kwargs['autotexts_color'] if 'autotexts_color' in kwargs else 'white'
    texts_size = kwargs['texts_size'] if 'texts_size' in kwargs else 14
    texts_color = kwargs['texts_color'] if 'texts_stexts_colorize' in kwargs else 'black'

    # Setting up labels
    plt.setp(autotexts, size=autotexts_size, color=autotexts_color)
    plt.setp(texts, size=texts_size, color=texts_color)

    # Saving image if applicable
    if 'save' in kwargs and bool(kwargs['save']):
        output_path = kwargs['output_path'] if 'output_path' in kwargs else 'output/'
        img_name = kwargs['img_name'] if 'img_name' in kwargs else f'{col}_piechart.png'
        save_fig(fig=fig, output_path=output_path, img_name=img_name)

def plot_double_donut_chart(df, col1, col2, **kwargs):
    """
    Creates a "double" custom donut chart for two columns of a giben dataset
    
    Parameters
    ----------
    :param df: dataset used for plotting [type: pd.DataFrame]
    :param col1: outter column name to be plotted on the external part of the donut [type: string]
    :param col1: inner column name to be plotted on the internal part of the donut [type: string]
    :param **kwargs: additional parameters
        :arg label_names_col1: label names for outter column of the donut [type: string]
        :arg label_names_col2: label names for inner column of the donut [type: string]
        :arg colors1: color list for outter column of the donut [type: list]
        :arg colors2: color list for inner column of the donut [type: list]
        :arg figsize: figure dimension [type: tuple, default=(8, 8)]
        :arg ax: matplotlib axis in case of external figure defition [type: mpl.Axes, default=None]
        :arg circle_radius: central circle radius of the chart [type: float, default=0.55]
        :arg circle_radius_color: central circle color of the chart [type: string, default='white']
        :arg text: central text on the donut [type: string, default='']   
        :arg title: chart title [type: string, default=f'Donut Chart for Feature {col}']
        :arg autotexts_size: label size from the numerical value [type: int, default=14]
        :arg autotexts_color: label color from the numerical value [type: int, default='black']
        :arg texts_size: label size from the chart [type: int, default=14]
        :arg texts_color: label color from the chart [type: int, default='black']
        :arg save: flag for saving the image created [type: bool, default=None]
        :arg output_path: path for image to be saved [type: string, default='output/']
        :arg img_name: filename for image to be saved [type: string, default=f'{col}_piechart.png']
    
    Return
    ------
    This function returns nothing besides plotting the custom double donut chart

    Application
    -----------
    plot_pie_chart(df=df, col='categorical_column', label_names={1: 'Class 1', 2: 'Class 2'})
    """
    
    # Validating column name on the given dataset
    if col1 not in df.columns:
        print(f'There is no column {col1} on the given dataset')
        return
    if col2 not in df.columns:
        print(f'There is no column {col2} on the given dataset')
        return

    # Returning labels ans values
    first_layer_donut = df.groupby(col1).count().iloc[:, 0]
    first_layer_values = first_layer_donut.values
    second_layer_donut = df.groupby([col1, col2]).count().iloc[:, 0]
    second_layer_values = second_layer_donut.values
    col2_index = df.groupby(col2).count().iloc[:, 0].index
    
    # Creating a DataFrame with outter layer data
    second_layer_df = pd.DataFrame(second_layer_donut.index.values)
    second_layer_df['first_index'] = second_layer_df[0].apply(lambda x: x[0])
    second_layer_df['second_index'] = second_layer_df[0].apply(lambda x: x[1])
    second_layer_df['values'] = second_layer_donut.values

    # Returning labels and mapping into the categories
    if 'label_names_col1' in kwargs:
        try:
            labels_col1 = first_layer_donut.index.map(kwargs['label_names_col1'])
        except Exception as e:
            print(f'Error on mapping the dict label_names on column {col1}. Exception: {e}')
    else:
        labels_col1 = first_layer_donut.index

    if 'label_names_col2' in kwargs:
        try:
            labels_col2 = second_layer_df['second_index'].map(kwargs['label_names_col2'])
        except Exception as e:
            print(f'Error on mapping the dict label_names on column {col2}. Exception: {e}')
    else:
        labels_col2 = second_layer_df['second_index']
    
    # Colors for the chart
    color_list = ['darkslateblue', 'crimson', 'lightseagreen', 'silver', 'lightskyblue', 'lightcoral']
    colors1 = kwargs['colors1'] if 'colors1' in kwargs else color_list[:len(label_names_col1)]
    colors2 = kwargs['colors2'] if 'colors2' in kwargs else color_list[-len(col2_index):]

    # Setting up parameters
    figsize = kwargs['figsize'] if 'figsize' in kwargs else (8, 8)
    ax = kwargs['ax'] if 'ax' in kwargs else None
    circle_radius = kwargs['circle_radius'] if 'circle_radius' in kwargs else 0.55
    circle_radius_color = kwargs['circle_radius_color'] if 'circle_radius_color' in kwargs else 'white'

    # Plotting a donut chart twice
    center_circle = plt.Circle((0, 0), circle_radius, color=circle_radius_color)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    wedges1, texts1, autotexts1 = ax.pie(first_layer_values, colors=colors1, startangle=90, 
                                         autopct=make_autopct(first_layer_values), pctdistance=1.20)
    wedges2, texts2, autotexts2 = ax.pie(second_layer_values, radius=0.75, colors=colors2, startangle=90, 
                                         autopct=make_autopct(second_layer_values), pctdistance=0.55)
    ax.add_artist(center_circle)

    # Setting central text information
    text = kwargs['text'] if 'text' in kwargs else ''
    text_kwargs = dict(size=20, fontweight='bold', va='center')
    ax.text(0, 0, text, ha='center', **text_kwargs)
    
    # Setting chart title
    title = kwargs['title'] if 'title' in kwargs else f'Double Donut Chart for Features \n{col1} and {col2}'
    ax.set_title(title, size=16, color='dimgrey')

    # Customizing the graph
    autotexts_size = kwargs['autotexts_size'] if 'autotexts_size' in kwargs else 14
    autotexts_color = kwargs['autotexts_color'] if 'autotexts_color' in kwargs else 'black'
    texts_size = kwargs['texts_size'] if 'texts_size' in kwargs else 14
    texts_color = kwargs['texts_color'] if 'texts_stexts_colorize' in kwargs else 'black'

    # Setting up the data labels on the cart
    plt.setp(autotexts1, size=autotexts_size, color=autotexts_color)
    plt.setp(texts1, size=texts_size, color=texts_color)
    plt.setp(autotexts2, size=autotexts_size, color=autotexts_color)
    plt.setp(texts2, size=texts_size, color=texts_color)
    
    # Setting and positioning graph legend
    custom_lines = []
    for c1 in colors1:
        custom_lines.append(Line2D([0], [0], color=c1, lw=4))
    for c2 in colors2:
        custom_lines.append(Line2D([0], [0], color=c2, lw=4))
    all_labels = list(labels_col1) + list(np.unique(labels_col2))
    ax.legend(custom_lines, labels=all_labels, fontsize=12, loc='upper left')

    # Saving image if applicable
    if 'save' in kwargs and bool(kwargs['save']):
        output_path = kwargs['output_path'] if 'output_path' in kwargs else 'output/'
        img_name = kwargs['img_name'] if 'img_name' in kwargs else f'{col1}_{col2}_donutchart.png'
        save_fig(fig=fig, output_path=output_path, img_name=img_name)


"""
---------------------------------------------------
---------- 2. CUSTOM GRAPHICAL ANALYSIS -----------
        2.2 Simple and percentual countplots
---------------------------------------------------
"""

# Simple countplot
def plot_countplot(df, col, **kwargs):
    """
    Creates a simple countplot using a dataset and a column name
    
    Parameters
    ----------
    :param df: dataset used for plotting [type: pd.DataFrame]
    :param col: column to be used as target of the countplot [type: string]
    :param **kwargs: additional parameters
        :arg top: filter the top N categories on the chart [type: int, default=-1]
        :arg figsize: figure dimension [type: tuple, default=(10, 7)]
        :arg ax: matplotlib axis in case of external figure defition [type: mpl.Axes, default=None]
        :arg hue: breaks the chart into another category (seaborn hue function arg) [type: string, default=None]
        :arg palette: color palette to be used on the chart [type: string, default='rainbow']
        :arg order: sorts categories (seaborn order function arg) [type: bool, default=True]
        :arg orient: horizontal or vertical orientation [type: string, default='h']
        :arg title: chart title [type: string, default=f'Countplot for Feature {col}']
        :arg size_title: title size [type: int, default=16]
        :arg size_labels: label size [type: int, default=14]
        :arg label_names: custom labels [type: dict, default=value_counts().index]    
        :arg save: flag for saving the image created [type: bool, default=None]
        :arg output_path: path for image to be saved [type: string, default='output/']
        :arg img_name: filename for image to be saved [type: string, default=f'{col}_countplot.png']

    Return
    ------
    This function returns nothing besides plotting the countplot chart

    Application
    -----------
    plot_countplot(df=df, col='column')
    """
    
    # Validating column name on the given dataset
    if col not in df.columns:
        print(f'There is no column {col} on the given dataset')
        return
    
    # Filtering categories if applicable
    top = kwargs['top'] if 'top' in kwargs else -1
    if top > 0:
        cat_count = df[col].value_counts()
        top_categories = cat_count[:top].index
        df = df[df[col].isin(top_categories)]
        
    # Setting up parameters
    figsize = kwargs['figsize'] if 'figsize' in kwargs else (10, 7)
    ax = kwargs['ax'] if 'ax' in kwargs else None
    hue = kwargs['hue'] if 'hue' in kwargs else None
    palette = kwargs['palette'] if 'palette' in kwargs else 'rainbow'
    order = df[col].value_counts().index if 'order' in kwargs and bool(kwargs['order']) else None
    orient = kwargs['orient'] if 'orient' in kwargs and kwargs['orient'] in ['h', 'v'] else 'v'
        
    # Setting chart orientation
    if orient == 'h':
        x = None
        y = col
    else:
        x = col
        y = None
    
    # Creating figure and applying countplot function
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    sns.countplot(data=df, ax=ax, x=x, y=y, hue=hue, order=order, palette=palette)

    # Returning customization parameters
    title = kwargs['title'] if 'title' in kwargs else f'Countplot for Feature {col}'
    size_title = kwargs['size_title'] if 'size_title' in kwargs else 16
    size_labels = kwargs['size_labels'] if 'size_labels' in kwargs else 14
    label_names = kwargs['label_names'] if 'label_names' in kwargs else None
    
    # Customizing chart
    ax.set_title(title, size=size_title, pad=20)
    format_spines(ax, right_border=False)

    # Inserting percentage and changing labels wherever orient is equel to 'v'
    ncount = len(df)
    if x:
        # Data labels
        for p in ax.patches:
            x = p.get_bbox().get_points()[:, 0]
            y = p.get_bbox().get_points()[1, 1]
            try:
                ax.annotate('{}\n{:.1f}%'.format(int(y), 100. * y / ncount), (x.mean(), y), 
                            ha='center', va='bottom', size=size_labels)
            except ValueError as ve: # Error by zero division by non existent values
                continue
        
        # Label names
        if 'label_names' in kwargs:
            labels_old = ax.get_xticklabels()
            labels = [l.get_text() for l in labels_old]
            try:
                # Converting text before mapping
                if type(list(kwargs['label_names'].keys())[0]) is int:
                    labels = [int(l) for l in labels]
                elif type(list(kwargs['label_names'].keys())[0]) is float:
                    labels = [float(l) for l in labels]
                
                # Mapping custom labels
                labels = pd.DataFrame(labels)[0].map(kwargs['label_names'])
                ax.set_xticklabels(labels)
            except Exception as e:
                print(f'Error on mapping the dict label_names on column {col}. Exception: {e}')
    
    # Inserting percentage and changing labels wherever orient is equel to 'h'
    else:
        # Data labels
        for p in ax.patches:
            x = p.get_bbox().get_points()[1, 0]
            y = p.get_bbox().get_points()[:, 1]
            try:
                ax.annotate('{} ({:.1f}%)'.format(int(x), 100. * x / ncount), (x, y.mean()), 
                            va='center', size=size_labels)
            except ValueError as ve: # Error by zero division by non existent values
                continue

        # Label names
        if 'label_names' in kwargs:
            labels_old = ax.get_yticklabels()
            labels = [l.get_text() for l in labels_old]
            try:
                # Converting text before mapping
                if type(list(kwargs['label_names'].keys())[0]) is int:
                    labels = [int(l) for l in labels]
                elif type(list(kwargs['label_names'].keys())[0]) is float:
                    labels = [float(l) for l in labels]
                
                # Mapping custom labels
                labels = pd.DataFrame(labels)[0].map(kwargs['label_names'])
                ax.set_yticklabels(labels)
            except Exception as e:
                print(f'Error on mapping the dict label_names on column {col}. Exception: {e}')

    # Saving image if applicable
    if 'save' in kwargs and bool(kwargs['save']):
        output_path = kwargs['output_path'] if 'output_path' in kwargs else 'output/'
        img_name = kwargs['img_name'] if 'img_name' in kwargs else f'{col}_countplot.png'
        save_fig(fig=fig, output_path=output_path, img_name=img_name)

# Percentual countplot using crosstab and grouped barchart
def plot_pct_countplot(df, col, hue, **kwargs):
    """
    Creates a percentage countplot (grouped bar chart) using a dataset and a column name
    
    Parameters
    ----------
    :param df: dataset used for plotting [type: pd.DataFrame]
    :param col: column to be used as target of the countplot [type: string]
    :param **kwargs: additional parameters
        :arg top: filter the top N categories on the chart [type: int, default=-1]
        :arg figsize: figure dimension [type: tuple, default=(10, 7)]
        :arg ax: matplotlib axis in case of external figure defition [type: mpl.Axes, default=None]   
        :arg palette: color palette to be used on the chart [type: string, default='rainbow']       
        :arg orient: horizontal or vertical orientation [type: string, default='h']
        :arg title: chart title [type: string, default=f'Countplot for Feature {col}']
        :arg size_title: title size [type: int, default=16]
        :arg label_names: custom labels [type: dict, default=value_counts().index]    
        :arg save: flag for saving the image created [type: bool, default=None]
        :arg output_path: path for image to be saved [type: string, default='output/']
        :arg img_name: filename for image to be saved [type: string, default=f'{col}_countplot.png']

    Return
    -------
    This function returns nothing besides plotting the countplot chart

    Application
    -----------
    plot_countplot(df=df, col='column')
    """
    
    # Validating column name on the given dataset
    if col not in df.columns:
        print(f'There is no column {col} on the given dataset')
        return
    
    # Validating hue column on the given dataset
    if hue not in df.columns:
        print(f'There is no column {hue} on the given dataset')
        return
    
    # Filtering categories if applicable
    top = kwargs['top'] if 'top' in kwargs else -1
    if top > 0:
        cat_count = df[col].value_counts()
        top_categories = cat_count[:top].index
        df = df[df[col].isin(top_categories)]
        
    # Setting up parameters
    figsize = kwargs['figsize'] if 'figsize' in kwargs else (10, 7)
    ax = kwargs['ax'] if 'ax' in kwargs else None
    palette = kwargs['palette'] if 'palette' in kwargs else 'rainbow'
    kind = 'bar' if 'orient' in kwargs and kwargs['orient'] == 'v' else 'barh'
    title = kwargs['title'] if 'title' in kwargs else f'Grouped Percentage Countplot for Feature \n{col}'
    size_title = kwargs['size_title'] if 'size_title' in kwargs else 16
    
    # Grouping column using crosstab function
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    col_to_hue = pd.crosstab(df[col], df[hue])
    col_to_hue.div(col_to_hue.sum(1).astype(float), axis=0).plot(kind=kind, stacked=True, ax=ax, 
                                                                 colormap=palette)
    
    # Customizing title
    ax.set_title(title, size=size_title, pad=20)

    # Setting labels if kind is equal to barh chart
    if kind == 'barh':
        if 'label_names' in kwargs:
            labels_old = ax.get_xticklabels()
            labels = [l.get_text() for l in labels_old]
            try:
                # Converting text before mapping
                if type(list(kwargs['label_names'].keys())[0]) is int:
                    labels = [int(l) for l in labels]
                elif type(list(kwargs['label_names'].keys())[0]) is float:
                    labels = [float(l) for l in labels]
                
                # Mapping custom labels
                labels = pd.DataFrame(labels)[0].map(kwargs['label_names'])
                ax.set_xticklabels(labels)
            except Exception as e:
                print(f'Error on mapping labels on column {col}. Exception: {e}')
    
    # Setting labels if kind is equal to barh chart
    else:
        if 'label_names' in kwargs:
            labels_old = ax.get_yticklabels()
            labels = [l.get_text() for l in labels_old]
            try:
                # Converting text before mapping
                if type(list(kwargs['label_names'].keys())[0]) is int:
                    labels = [int(l) for l in labels]
                elif type(list(kwargs['label_names'].keys())[0]) is float:
                    labels = [float(l) for l in labels]
                
                # Mapping custom labels
                labels = pd.DataFrame(labels)[0].map(kwargs['label_names'])
                ax.set_yticklabels(labels)
            except Exception as e:
                print(f'Error on mapping labels on column {col}. Exception: {e}')

    # Saving image if applicable
    if 'save' in kwargs and bool(kwargs['save']):
        output_path = kwargs['output_path'] if 'output_path' in kwargs else 'output/'
        img_name = kwargs['img_name'] if 'img_name' in kwargs else f'{col}_{hue}_pctcountplot.png'
        save_fig(fig=fig, output_path=output_path, img_name=img_name)


"""
---------------------------------------------------
---------- 2. CUSTOM GRAPHICAL ANALYSIS -----------
               2.3 Distribution plot
---------------------------------------------------
"""

# Distribution plot from a numerical feature
def plot_distplot(df, col, kind='dist', **kwargs):
    """
    Creates a custom distribution plot based on a numeric column
    
    Parameters
    ----------
    :param df: dataset used for plotting [type: pd.DataFrame]
    :param col: numeric column to be its distribution plotted [type: string]
    :param kind: kind of distribution plot [type: string, default='dist']
        *options for this parameter: ['dist', 'kde', 'box', 'boxen', 'strip']
    :param **kwargs: additional parameters
        :arg hue: breaks the chart into another category (seaborn hue function arg) [type: string, default=None]
        :arg figsize: figure dimension [type: tuple, default=(10, 7)]
        :arg ax: matplotlib axis in case of external figure defition [type: mpl.Axes, default=None]
        :arg hist: plots histogram bars on the chart (seaborn's parameter) [type: bool, default=False]
        :arg kde: plots kde line on the chart (seaborn's parameter) [type: bool, default=True]
        :arg rug: plots rug at the bottom of the the chart (seaborn's parameter) [type: bool, default=False]
        :arg shade: fills the area below distribution curve (seaborn's parameter) [type: bool, default=True]
        :arg color: color of the distribution line [type: string, default='darkslateblue']
        :arg palette: color palette to be used on the chart [type: string, default='rainbow']
        :arg title: chart title [type: string, default=f'{kind.title()}plot for Feature {col}']
        :arg size_title: title size [type: int, default=16]
        :arg color_list: list of colors to be used on the chart if applicable [type: list]
            *default: ['darkslateblue', 'crimson', 'cadetblue', 'mediumseagreen', 'salmon', 'lightskyblue', 'darkgray']
        :arg save: flag for saving the image created [type: bool, default=None]
        :arg output_path: path for image to be saved [type: string, default='output/']
        :arg img_name: filename for image to be saved [type: string, default=f'{col}{hue}_{kind}plot.png']

    Return
    ------
    This function returns nothing besides plotting the distribution chart

    Application
    ---------
    plot_distplot(df=df, col='column_name')
    """

    # Searching if the columns on parameters are on the given dataset
    hue = kwargs['hue'] if 'hue' in kwargs else None
    if col not in df.columns:
        print(f'There is no column {col} on the given dataset')
        return
    if hue is not None and hue not in df.columns:
        print(f'There is no column {hue} on the given dataset')
        return
    
    # Validating kind of the chart to be plotted
    possible_kinds = ['dist', 'kde', 'box', 'boxen', 'strip']
    if kind not in possible_kinds:
        print(f'Invalid "kind" parameter. Please choose between: {possible_kinds}')

    # Extracting chart parameters
    figsize = kwargs['figsize'] if 'figsize' in kwargs else (10, 7)
    ax = kwargs['ax'] if 'ax' in kwargs else None
    hist = kwargs['hist'] if 'hist' in kwargs else False
    kde = kwargs['kde'] if 'kde' in kwargs else True
    rug = kwargs['rug'] if 'rug' in kwargs else False
    shade = kwargs['shade'] if 'shade' in kwargs else True
    color = kwargs['color'] if 'color' in kwargs else 'darkslateblue'
    palette = kwargs['palette'] if 'palette' in kwargs else 'rainbow'
    title = kwargs['title'] if 'title' in kwargs else f'{kind.title()}plot for Feature {col}'
    size_title = kwargs['size_title'] if 'size_title' in kwargs else 16
    list_of_colors = ['darkslateblue', 'crimson', 'cadetblue', 'mediumseagreen', 
                      'salmon', 'lightskyblue', 'cornflowerblue']
    color_list = kwargs['color_list'] if 'color_list' in kwargs else list_of_colors
    c = 0

    # Setting seaborn style
    sns.set(style='white', palette='muted', color_codes=True)

    # Building figure and axis if applicable
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Distplot
    if kind == 'dist':
        if hue is not None:
            for cat in df[hue].value_counts().index:
                color = color_list[c]
                sns.distplot(df[df[hue]==cat][col], ax=ax, hist=hist, kde=kde, rug=rug, 
                            label=cat, color=color)
                c += 1
        else:
            sns.distplot(df[col], ax=ax, hist=hist, kde=kde, rug=rug, color=color)
    # Kdeplot        
    elif kind == 'kde':
        if hue is not None:
            for cat in df[hue].value_counts().index:
                color = color_list[c]
                sns.kdeplot(df[df[hue]==cat][col], ax=ax, shade=shade, label=cat, color=color)
                c += 1
        else:
            sns.kdeplot(df[col], ax=ax, shade=shade, color=color)
    # Boxplot
    elif kind == 'box':
        if hue is not None:
            sns.boxplot(x=hue, y=col, data=df, ax=ax, palette=palette)
        else:
            sns.boxplot(y=col, data=df, ax=ax, palette=palette)
    # Boxenplot
    elif kind == 'boxen':
        if hue is not None:
            sns.boxenplot(x=hue, y=col, data=df, ax=ax, palette=palette)
        else:
            sns.boxenplot(y=col, data=df, ax=ax, palette=palette)
    # Stripplot
    elif kind == 'strip':
        if hue is not None:
            sns.stripplot(x=hue, y=col, data=df, ax=ax, palette=palette)
        else:
            sns.stripplot(y=col, data=df, ax=ax, palette=palette)
            
    # Changing labels (in case of kind=box, boxen or strip)
    if 'label_names' in kwargs and hue is not None and kind in ['box', 'boxen', 'strip']:
        labels_old = ax.get_xticklabels()
        labels = [l.get_text() for l in labels_old]
        try:
            # Converting texts before mapping
            if type(list(kwargs['label_names'].keys())[0]) is int:
                labels = [int(l) for l in labels]
            elif type(list(kwargs['label_names'].keys())[0]) is float:
                labels = [float(l) for l in labels]

            # Mapping custom labels
            labels = pd.DataFrame(labels)[0].map(kwargs['label_names'])
            ax.set_xticklabels(labels)
        except Exception as e:
            print(f'Error on mapping labels on column {col}. Exception: {e}')
            
    # Customizing chart
    format_spines(ax=ax, right_border=False)
    ax.set_title(title, size=size_title)
    if kind in ['dist', 'kde'] and hue is not None:
        ax.legend(title=hue)
        
    # Saving image if applicable
    if 'save' in kwargs and bool(kwargs['save']):
        output_path = kwargs['output_path'] if 'output_path' in kwargs else 'output/'
        img_name = kwargs['img_name'] if 'img_name' in kwargs else f'{col}{hue}_{kind}plot.png'
        save_fig(fig=fig, output_path=output_path, img_name=img_name)


"""
---------------------------------------------------
---------- 2. CUSTOM GRAPHICAL ANALYSIS -----------
             2.4 Aggregation approach
---------------------------------------------------
"""

# Simple aggregation plot based on a group and a value columns
def plot_aggregation(df, group_col, value_col, aggreg, **kwargs):
    """
    Plots a custom aggregate chart into a bar style
    
    Parameters
    ----------
    :param df: dataset used for plotting [type: pd.DataFrame]
    :param group_col: group column used on aggregation [type: string]
    :param value_col: value column with values to be aggregated [type: string]
    :param aggreg: name reference of aggregation to be used [type: string]
    :param **kwargs: additional parameters 
        :arg hue: breaks the chart into another category (seaborn hue function arg) [type: string, default=None]
        :arg top: filter the top N categories on the chart [type: int, default=-1]
        :arg figsize: figure dimension [type: tuple, default=(10, 7)]
        :arg ax: matplotlib axis in case of external figure defition [type: mpl.Axes, default=None]
        :arg palette: color palette to be used on the chart [type: string, default='rainbow']
        :arg label_names: custom labels [type: dict, default=value_counts().index]
        :arg order: sorts the categories (seaborn order function arg) [type: bool, default=True]
        :arg orient: horizontal or vertical orientation [type: string, default='h']
        :arg title: chart title [type: string, default=f'{aggreg.title()} of {value_col} by {group_col}']
        :arg size_title: title size [type: int, default=16]
        :arg size_labels: label size [type: int, default=14]
        :arg n_dec: number of decimal places on number formating [type: int, default=2]
        :arg save: flag for saving the image created [type: bool, default=None]
        :arg output_path: path for image to be saved [type: string, default='output/']
        :arg img_name: filename for image to be saved 
            [type: string, default={value_col}{hue}_{aggreg}plot_by{group_col}.png']

    Return
    ------
    This function returns nothing besides plotting the summarized bar chart

    Application
    -----------
    plot_aggregation(df=df, group_col='group', value_col='value', aggreg='agg')
    """
    
    # Searching if the parameters columns are on the dataset
    hue = kwargs['hue'] if 'hue' in kwargs else None
    df_columns = df.columns
    if group_col not in df_columns:
        print(f'There is no column {group_col} in the given dataset')
        return
    if value_col not in df_columns:
        print(f'There is no column {value_col} in the given dataset')
        return
    if hue is not None and hue not in df_columns:
        print(f'There is no column {hue} in the given dataset')
        return

    # Validating and transforming the aggreg parameter
    aggreg = aggreg.lower()

    # Applying the aggregation based on parameters
    try:
        if hue is not None:
            df_group = df.groupby(by=[group_col, hue], as_index=False).agg({value_col: aggreg})
        else:
            df_group = df.groupby(by=group_col, as_index=False).agg({value_col: aggreg})
        df_group[group_col] = df_group[group_col].astype(str)
    except AttributeError as ae:
        print(f'Error on applying aggregation with aggreg {aggreg}. Excepion: {ae}')
        
    # Filtering top N categories if applicable
    if 'top' in kwargs and kwargs['top'] > 0:
        df_group = df_group[:kwargs['top']]
        
    # Setting up chart parameters
    figsize = kwargs['figsize'] if 'figsize' in kwargs else (10, 7)
    ax = kwargs['ax'] if 'ax' in kwargs else None
    palette = kwargs['palette'] if 'palette' in kwargs else 'rainbow'
    
    # Setting up labels
    if 'label_names' in kwargs:
        try:
            df_group[group_col] = df_group[group_col].map(kwargs['label_names'])
        except Exception as e:
            print(f'Error on mapping the dict label_names on column {group_col}. Exception: {e}')
    order = kwargs['order'] if 'order' in kwargs else None
        
    # Chart orientation
    orient = kwargs['orient'] if 'orient' in kwargs and kwargs['orient'] in ['h', 'v'] else 'v'
    if orient == 'v':
        x = group_col
        y = value_col
    else:
        x = value_col
        y = group_col
        
    # Returning parameters for chart customization
    title = kwargs['title'] if 'title' in kwargs else f'{aggreg.title()} of {value_col} by {group_col}'
    size_title = kwargs['size_title'] if 'size_title' in kwargs else 16
    size_labels = kwargs['size_labels'] if 'size_labels' in kwargs else 14
    n_dec = kwargs['n_dec'] if 'n_dec' in kwargs else 2
    
    # Building up a barchart
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=x, y=y, data=df_group, ax=ax, hue=hue, palette=palette, ci=None, orient=orient, order=order)
    
    # Setting title and formating borders
    ax.set_title(title, size=size_title, pad=20)
    format_spines(ax, right_border=False)

    # Inserting percentage label
    if orient == 'h':
        AnnotateBars(n_dec=n_dec, font_size=size_labels, color='black').horizontal(ax)
    else:
        AnnotateBars(n_dec=n_dec, font_size=size_labels, color='black').vertical(ax)
            
    # Saving image if applicable
    if 'save' in kwargs and bool(kwargs['save']):
        output_path = kwargs['output_path'] if 'output_path' in kwargs else 'output/'
        img_name = kwargs['img_name'] if 'img_name' in kwargs else f'{value_col}{hue}_{aggreg}plot_by{group_col}.png'
        save_fig(fig=fig, output_path=output_path, img_name=img_name)

# A complete report using countplot, aggregation, distplot and statistical parameters
def plot_cat_aggreg_report(df, cat_col, value_col, aggreg='mean', **kwargs):
    """
    Creates a joint analysis between count, aggregation and distribution by union functions likes
    plot_countplot(), plot_aggregation() and plot_distplot(). This plot_cat_aggreg_report() is really
    useful for extract rich insights from the data and visualize it at one place
    
    Parameters
    ----------
    :param df: dataset used for plotting [type: pd.DataFrame]
    :param cat_col: categorical column used on analysis [type: string]
    :param value_col: value column with values to be aggregated [type: string]
    :param aggreg: name reference of aggregation to be used [type: string, default='mean']
    :param **kwargs: additional parameters 
        :arg figsize: figure dimension [type: tuple, default=(17, 5)]
        :arg hue: breaks the chart into another category (seaborn hue function arg) [type: string, default=None]
        :arg palette: color palette to be used on the chart [type: string, default='rainbow']
        :arg order: sorts categories into a defined order [type: list, default=None]
        :arg orient: horizontal or vertical orientation [type: string, default='v']
        :arg dist_kind: kind of distribution chart [type: string, default='dist']
        :arg title1: title for countplot chart [type: string, default=f'Countplot for {cat_col}']
        :arg title2: title for aggregation chart [type: string, default=f'{value_col} {aggreg.title()} by {cat_col}']
        :arg title3: head title for statistical axis [type: string, default='Statistical Analysis']
        :arg title4: title for distplot chart [type: string, default=f'{kind.title()}plot for {value_col}']
        :arg size_title: title size [type: int, default=16]
        :arg size_labels: label size [type: int, default=12]
        :arg top: filter the top N categories on the chart [type: int, default=-1]
        :arg desc_text_x_pos: initial x position of description text [type: float, default=.50]
        :arg desc_text_y_pos: initial y position of description text [type: float, default=.75]
        :arg desc_text: description text [type: string]
            *default: 'A statistical approac for {value_col}\nusing the data available']
        :arg desc_text_font: size of description text [type: int, default=12]
        :arg stat_title_x_pos: initial x position of statistical titles [type: float, default=.17]
        :arg stat_title_y_pos: initial y position of statistical titles [type: float]
            *default: desc_text_y_pos - 0.2
        :arg stat_title_mean: statistical mean title [type: string, default='Mean']
        :arg stat_title_median: statistical median title [type: string, default='Median']
        :arg stat_title_std: statistical standard deviation title [type: string, default='Std']
        :arg stat_title_font: size of statistical titles [type: int, default=14]
        :arg inc_x_pos: step on x axis for separating statistical elements [type: int, default=18]
        :arg stat_x_pos: initial x position of statistical values [type: float, default=.27]
        :arg stat_y_pos: initial y position of statistical values [type: float]
            *default: stat_title_y_pos - 0.22
        :arg hist: plots histogram bars on the chart (seaborn's parameter) [type: bool, default=True]
        :arg kde: plots kde line on the chart (seaborn's parameter) [type: bool, default=True]
        :arg rug: plots rug at the bottom of the the chart (seaborn's parameter) [type: bool, default=False]
        :arg shade: fills the area below distribution curve (seaborn's parameter) [type: bool, default=True]
        :arg color: color of the distribution line [type: string, default='darkslateblue']
        :arg palette: color palette to be used on the chart [type: string, default='rainbow']
        :arg save: flag indicativo de salvamento da imagem gerada [type: bool, default=None]
        :arg output_path: caminho de output da imagem a ser salva [type: string, default='output/']
        :arg img_name: nome do arquivo .png a ser gerado [type: string, default=f'{cat_col}_{value_col}_{aggreg}plot.png']

    Return
    ------
    This functions returns nothing besides the complete analysis chart based on given parameters

    Application
    -----------
    plot_cat_aggreg_report(df=df, cat_col="categoric_column", value_col="numeric_column")
    """

    # Verifying if cat_col and value_col are on dataset columns
    if cat_col not in df.columns:
        print(f'Column {cat_col} is not on the dataset. Please change "cat_col" parameter.')
        return
    elif value_col not in df.columns:
        print(f'Column {value_col} is not on the dataset. Please change "value_col" parameter.')
        return
    
    # Extraindo parâmetros adicionais da função
    figsize = kwargs['figsize'] if 'figsize' in kwargs else (17, 5)
    hue = kwargs['hue'] if 'hue' in kwargs else None
    palette = kwargs['palette'] if 'palette' in kwargs else 'rainbow'
    order = df[col].value_counts().index if 'order' in kwargs and bool(kwargs['order']) else None
    orient = kwargs['orient'] if 'orient' in kwargs and kwargs['orient'] in ['h', 'v'] else 'v'
    dist_kind = kwargs['dist_kind'] if 'dist_kind' in kwargs else 'dist'
    title1 = kwargs['title1'] if 'title1' in kwargs else f'Volumetria de dados por {cat_col}'
    title2 = kwargs['title2'] if 'title2' in kwargs else f'{aggreg.title()} de {value_col} por {cat_col}'
    title3 = kwargs['title3'] if 'title3' in kwargs else 'Parâmetros Estatísticos'
    title4 = kwargs['title4'] if 'title4' in kwargs else f'{dist_kind.title()}plot para {value_col}'
    size_title = kwargs['size_title'] if 'size_title' in kwargs else 16
    size_labels = kwargs['size_labels'] if 'size_labels' in kwargs else 12
    top = kwargs['top'] if 'top' in kwargs else -1
    
    # Validating distribution kind
    possible_kinds = ['dist', 'kde', 'box', 'boxen', 'strip']
    if dist_kind not in possible_kinds:
        print(f'Invalid dist_kind parameter. Possible options: {possible_kinds}')
        return
    
    # Building figure
    fig = plt.figure(constrained_layout=True, figsize=figsize)

    # Axis definition using GridSpec
    gs = GridSpec(2, 3, figure=fig)

    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[:, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 2])
    
    # Graph 01 - countplot for cat_col
    plot_countplot(df=df, col=cat_col, ax=ax1, hue=hue, palette=palette, order=order, title=title1,
                   orient=orient, size_title=size_title, size_labels=size_labels, top=top)
    
    # Graph 02 - aggregation plot for cat_col and value_col
    order_labels = [item.get_text() for item in ax1.get_xticklabels()]
    plot_aggregation(df=df, group_col=cat_col, value_col=value_col, ax=ax2, aggreg=aggreg, title=title2,
                     size_title=size_title, size_labels=size_labels, orient=orient, order=order_labels)
    
    # Graph 03 - statistical analysis of value_col
    describe = df[value_col].describe()
    mean = int(round(describe['mean'], 0))
    median = int(round(describe['50%'], 0))
    std = int(round(describe['std'], 0))
    
    # Extracting parameters for the third block
    len_mean = len(str(mean))
    len_median = len(str(median))
    
    # Handling description text positioning
    desc_text_x_pos = kwargs['desc_text_x_pos'] if 'desc_text_x_pos' in kwargs else 0.50
    desc_text_y_pos = kwargs['desc_text_y_pos'] if 'desc_text_y_pos' in kwargs else 0.75
    tmp_desc_text = f'A statistical approac for {value_col}\nusing the data available'
    desc_text = kwargs['desc_text'] if 'desc_text' in kwargs else tmp_desc_text
    desc_text_font = kwargs['desc_text_font'] if 'desc_text_font' in kwargs else 12
    
    # Handling positioning of statistical text titles
    stat_title_x_pos = kwargs['stat_title_x_pos'] if 'stat_title_x_pos' in kwargs else 0.17
    stat_title_y_pos = kwargs['stat_title_y_pos'] if 'stat_title_y_pos' in kwargs else desc_text_y_pos-.2
    stat_title_mean = kwargs['stat_title_mean'] if 'stat_title_mean' in kwargs else 'Média'
    stat_title_median = kwargs['stat_title_median'] if 'stat_title_median' in kwargs else 'Mediana'
    stat_title_std = kwargs['stat_title_std'] if 'stat_title_std' in kwargs else 'Desv Pad'
    stat_title_font = kwargs['stat_title_font'] if 'stat_title_font' in kwargs else 14
    inc_x_pos = kwargs['inc_x_pos'] if 'inc_x_pos' in kwargs else 18
    
    # Handling positioning of statistical values
    stat_x_pos = kwargs['stat_x_pos'] if 'stat_x_pos' in kwargs else .17
    stat_y_pos = kwargs['stat_y_pos'] if 'stat_y_pos' in kwargs else stat_title_y_pos-.22
    
    # Plotting description text on ax3 axis
    ax3.text(desc_text_x_pos, desc_text_y_pos, desc_text, fontsize=desc_text_font, ha='center', 
               color='black')
    
    # Plotting titles of statistical parameters
    ax3.text(stat_title_x_pos, stat_title_y_pos, stat_title_mean, fontsize=stat_title_font, 
               ha='center', color='black', style='italic')
    stat_title_x_pos += len_mean/inc_x_pos
    ax3.text(stat_title_x_pos, stat_title_y_pos, stat_title_median, fontsize=stat_title_font, 
               ha='center', color='black', style='italic')
    stat_title_x_pos += len_median/inc_x_pos
    ax3.text(stat_title_x_pos, stat_title_y_pos, stat_title_std, fontsize=stat_title_font, 
               ha='center', color='black', style='italic')
    
    # Plotting statistical parameters
    ax3.text(stat_x_pos, stat_y_pos, mean, fontsize=stat_title_font, ha='center', color='white', style='italic', weight='bold',
             bbox=dict(facecolor='navy', alpha=0.5, pad=10, boxstyle='round, pad=.7'))
    stat_x_pos += len_mean/inc_x_pos
    ax3.text(stat_x_pos, stat_y_pos, median, fontsize=stat_title_font, ha='center', color='white', style='italic', weight='bold',
             bbox=dict(facecolor='navy', alpha=0.5, pad=10, boxstyle='round, pad=.7'))
    stat_x_pos += len_median/inc_x_pos
    ax3.text(stat_x_pos, stat_y_pos, std, fontsize=stat_title_font, ha='center', color='white', style='italic', weight='bold',
             bbox=dict(facecolor='navy', alpha=0.5, pad=10, boxstyle='round, pad=.7'))
     
    # Formatting axis
    ax3.axis('off')
    ax3.set_title(title3, size=16, weight='bold', pad=20)
    
    # Graph 04 - distplot for value_col    
    
    # Extracting additional parameters
    hist = kwargs['hist'] if 'hist' in kwargs else True
    kde = kwargs['kde'] if 'kde' in kwargs else True
    rug = kwargs['rug'] if 'rug' in kwargs else False
    shade = kwargs['shade'] if 'shade' in kwargs else True
    color = kwargs['color'] if 'color' in kwargs else 'darkslateblue'
    palette = kwargs['palette'] if 'palette' in kwargs else 'rainbow'
    
    # Plotting distribution chart using plot_distplot() based on kind parameter
    if dist_kind in ['dist', 'kde']:
        plot_distplot(df=df, col=value_col, ax=ax4, kind=dist_kind, hue=None, hist=hist, kde=kde, rug=rug,
                      shade=shade, color=color, palette=palette, title=title4, size_title=size_title)
    else:
        plot_distplot(df=df, col=value_col, ax=ax4, kind=dist_kind, hue=cat_col, hist=hist, kde=kde, rug=rug,
                      shade=shade, color=color, palette=palette, title=title4, size_title=size_title)
    
    # Tighting layout
    plt.tight_layout()

    # Saving image if applicable
    if 'save' in kwargs and bool(kwargs['save']):
        output_path = kwargs['output_path'] if 'output_path' in kwargs else 'output/'
        img_name = kwargs['img_name'] if 'img_name' in kwargs else f'{cat_col}_{value_col}_{aggreg}plot.png'
        save_fig(fig=fig, output_path=output_path, img_name=img_name)


"""
---------------------------------------------------
---------- 2. CUSTOM GRAPHICAL ANALYSIS -----------
     2.5 General overview and correlation matrix
---------------------------------------------------
"""

# An overview from the data and its attributes
def data_overview(df, **kwargs):
    """
    Extract useful information parameters of a given dataset to offers a general overview from the data
    
    Parameters
    ----------
    :param df: dataset used for content analysis [type: pd.DataFrame]
    :param **kwargs: additional parameters
        :arg corr: flag for applying correlation analysis [type: bool, default=False]
        :arg corr_method: correlation method in case of correlation analysis to be applied [type: string, default='pearson'] 
            *options: ['pearson', 'kendall', 'spearman']
        :arg target: column to be used as target of correlation analysis [type: string, default=None]
        :arg thresh_corr: threshold for filtering corr >= threshold [type: float, default=None]
        :arg thresh_null: threshold for filtering qtd_null >= threshold [type: float, default=0]
        :arg thresh_pct_null: percentual threshold para filtering pct_null >= threshold [type: float, default=0]
        :arg sort: overview attribute to be sorted on the final dataset [type: string, defaul='qtd_null']
        :arg ascending: sorting flag [type: bool, default=False]

    Return
    ------
    :return df_overview: dataset with general parameters of each column of the original data
    
    Application
    -----------
    df_overview = data_overview(df=df, corr=True, target='target')
    """
    
    # Returning null data on the given dataset
    df_null = pd.DataFrame(df.isnull().sum()).reset_index()
    df_null.columns = ['feature', 'qtd_null']
    df_null['pct_null'] = df_null['qtd_null'] / len(df)
    
    # Returning dtypes and total of categories entries from categorical attributes
    df_null['dtype'] = df_null['feature'].apply(lambda x: df[x].dtype)
    df_null['qtd_cat'] = [len(df[col].value_counts()) if df[col].dtype == 'object' else 0 for col in 
                          df_null['feature'].values]
    
    # Returning additional parameters of the function
    corr = kwargs['corr'] if 'corr' in kwargs else False
    corr_method = kwargs['corr_method'] if 'corr_method' in kwargs else 'pearson'
    target = kwargs['target'] if 'target' in kwargs else None
    thresh_corr = kwargs['thresh_corr'] if 'thresh_corr' in kwargs else None
    thresh_null = kwargs['thresh_null'] if 'thresh_null' in kwargs else 0
    thresh_pct_null = kwargs['thresh_pct_null'] if 'thresh_pct_null' in kwargs else 0   
    sort = kwargs['sort'] if 'sort' in kwargs else 'qtd_null'
    ascending = kwargs['sort_ascending'] if 'sort_ascending' in kwargs else False
    
    # Validating correlation parameters
    if corr and target is None:
        print(f"When corr=True it's also needed to define 'target' parameter")
        return
    
    if corr and target is not None:
        # Extracting correlation
        target_corr = pd.DataFrame(df.corr(method=corr_method)[target])
        target_corr = target_corr.reset_index()
        target_corr.columns = ['feature', f'target_{corr_method}_corr']
        
        # Joining data and filtering based on correlation threshold
        df_overview = df_null.merge(target_corr, how='left', on='feature')
        if thresh_corr is not None:
            df_overview = df_overview[df_overview[f'target_{corr_method}_corr'] > thresh_corr]
            
    else:
        # Correlation analysis won't be applied
        df_overview = df_null
        
    # Filtering null data based on null thresholds
    df_overview = df_overview.query('pct_null >= @thresh_null')
    df_overview = df_overview.query('qtd_null >= @thresh_pct_null')
    
    # Sorting the final dataset
    df_overview.sort_values(by=sort, ascending=ascending, inplace=True)
    df_overview.reset_index(drop=True, inplace=True)
    
    return df_overview

# A beautiful correlation matrix
def plot_corr_matrix(df, corr_col, corr='positive', **kwargs):
    """
    Plots a beautiful and custom correlation matrix for a given dataset and a target column
    
    Parameters
    ----------
    :param df: dataset used for plotting [type: pd.DataFrame]
    :param corr_col: column to be used as the target on correlation analysis [type: string]
    :param corr: kind of correlation (positive or negative) [type: string, default='positive']
    :param **kwargs: additional parameters
        :arg n_vars: number of features to be plotted [type: int, default=len(num_features)]
        :arg title: chart title [type: string]
            *default: f'Top {n_vars} Features with {corr.title()} Correlation \nwith Target {corr_col}'
        :arg figsize: figure dimension [type: tuple, default=(10, 10)]
        :arg ax: matplotlib axis in case of external figure defition [type: mpl.Axes, default=None]
        :arg size_title: title size [type: int, default=16]
        :arg cbar: flag for showing a right sided color bar [type: bool, default=True]
        :arg annot: flag for showing labels on matrix cells [type: bool, default=True]
        :arg square: flag for set matrix dimension as a square [type: bool, default=True]
        :arg fmt: labels formatting of the matrix [type: string, default='.2f']
        :arg save: flag for saving the image created [type: bool, default=None]
        :arg output_path: path for image to be saved [type: string, default='output/']
        :arg img_name: filename for image to be saved [type: string, default=f'{corr_col}_{kind[:3]}_corr_mx.png']
    
    Return
    ------
    This function returns nothing besides plotting the correlation matrix chart

    Application
    -----------
    plot_corr_matrix(df=df, corr_col='target')
    """
    
    # Creating a correlation matrix for the DataFrame
    corr_mx = df.corr()
    
    # Extracting numerical features from the dataset and filtering n_vars if applicable
    num_features = [col for col, dtype in df.dtypes.items() if dtype != 'object']
    n_vars = kwargs['n_vars'] if 'n_vars' in kwargs else len(num_features)
    title = f'Top {n_vars} Features with {corr.title()} Correlation \nwith Target {corr_col}'

    # Associating parameters based on type of correlation
    if corr == 'positive':
        # Returning top n_vars columns with positive correlation and setting cmap
        corr_cols = list(corr_mx.nlargest(n_vars+1, corr_col)[corr_col].index)
        cmap = kwargs['cmap'] if 'cmap' in kwargs else 'YlGnBu'
        
    elif corr == 'negative':
        # Returning top n_vars columns with negative correlation and setting cmap
        corr_cols = list(corr_mx.nsmallest(n_vars+1, corr_col)[corr_col].index)
        corr_cols = [corr_col] + corr_cols[:-1]
        cmap = kwargs['cmap'] if 'cmap' in kwargs else 'magma'
        
    # Creating a correlation array using np.corrcoef
    corr_data = np.corrcoef(df[corr_cols].values.T)

    # Setting up general parameters of the chart
    figsize = kwargs['figsize'] if 'figsize' in kwargs else (10, 10)
    ax = kwargs['ax'] if 'ax' in kwargs else None
    size_title = kwargs['size_title'] if 'size_title' in kwargs else 16
    cbar = kwargs['cbar'] if 'cbar' in kwargs else True
    annot = kwargs['annot'] if 'annot' in kwargs else True
    square = kwargs['square'] if 'square' in kwargs else True
    fmt = kwargs['fmt'] if 'fmt' in kwargs else '.2f'
    
    # Plotting the matrix as heatmap function
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_data, ax=ax, cbar=cbar, annot=annot, square=square, fmt=fmt, cmap=cmap,
                yticklabels=corr_cols, xticklabels=corr_cols)
    ax.set_title(title, size=size_title, color='black', pad=20)

    # Saving image if applicable
    if 'save' in kwargs and bool(kwargs['save']):
        output_path = kwargs['output_path'] if 'output_path' in kwargs else 'output/'
        img_name = kwargs['img_name'] if 'img_name' in kwargs else f'{corr_col}_{kind[:3]}_corr_mx.png'
        save_fig(fig=fig, output_path=output_path, img_name=img_name)


"""
---------------------------------------------------
---------- 2. CUSTOM GRAPHICAL ANALYSIS -----------
        2.6 Multiple plots at one function
---------------------------------------------------
"""

# Multiple distplots
def plot_multiple_distplots(df, col_list, n_cols=3, kind='dist', **kwargs):
    """
    Plots custom distribution charts for multiple columns at once using the col_list parameter
    
    Parameters
    ----------
    :param df: dataset used for plotting [type: pd.DataFrame]
    :param col_list: list with numerical columns to be used on analysis [type: list]
    :param n_cols: figure columns configured for this multiple graph [type: int, default=3]
    :param kind: kind of distribution plot [type: string, default='dist']
        *options: ['dist', 'kde', 'box', 'boxen', 'strip']
    :param **kwargs: additional parameters
        :arg figsize: figure dimension [type: tuple, default=(10, n_rows * 5)]
            *where n_rows = ceil(len(col_list) / n_cols)
        :arg hue: breaks the chart into another category (seaborn hue function arg) [type: string, default=None]
        :arg hist: plots histogram bars on the chart (seaborn's parameter) [type: bool, default=False]
        :arg kde: plots kde line on the chart (seaborn's parameter) [type: bool, default=True]
        :arg rug: plots rug at the bottom of the the chart (seaborn's parameter) [type: bool, default=False]
        :arg shade: fills the area below distribution curve (seaborn's parameter) [type: bool, default=True]
        :arg color: color of the distribution line [type: string, default='darkslateblue']
        :arg palette: color palette to be used on the chart [type: string, default='rainbow']
        :arg size_title: title size [type: int, default=16]
        :arg save: flag for saving the image created [type: bool, default=None]
        :arg output_path: path for image to be saved [type: string, default='output/']
        :arg img_name: filename for image to be saved [type: string, default=f'multiple_{kind}plot.png']    

    Return
    ------
    This function returns nothing besides plotting the multiple distribution charts

    Application
    ---------
    plot_multiple_distplots(df=df, col_list['colA', 'colB', 'colC'])
    """
    
    # Validating kind of the chart to be plotted
    possible_kinds = ['dist', 'kde', 'box', 'boxen', 'strip']
    if kind not in possible_kinds:
        print(f'Invalid "kind" parameter. Please choose between: {possible_kinds}')
    
    # Computing figure parameters for axis structure
    n_rows = ceil(len(col_list) / n_cols)
    figsize = kwargs['figsize'] if 'figsize' in kwargs else (17, n_rows * 5)
    i, j = 0, 0
    
    # Extracting additional parameters for the charts
    hist = kwargs['hist'] if 'hist' in kwargs else True
    kde = kwargs['kde'] if 'kde' in kwargs else True
    rug = kwargs['rug'] if 'rug' in kwargs else False
    shade = kwargs['shade'] if 'shade' in kwargs else True
    color = kwargs['color'] if 'color' in kwargs else 'darkslateblue'
    palette = kwargs['palette'] if 'palette' in kwargs else 'rainbow'
    size_title = kwargs['size_title'] if 'size_title' in kwargs else 16
    
    # Creating a figure with multiple axis and iterating over them
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)
    for col in col_list:
        # Indexing axis
        try:
            ax = axs[i, j]
        except IndexError as ie:
            # For one-dimensional figure, it's necessary to index axis as a vector and not as a matrix
            ax = axs[j]
        title = f'{kind.title()} for {col}'
        
        # Plotting distribution plot with all parameters extracted
        plot_distplot(df=df, col=col, ax=ax, kind=kind, hue=hue, hist=hist, kde=kde, rug=rug,
                      shade=shade, color=color, palette=palette, title=title, size_title=size_title)
        
        # Stepping up index
        j += 1
        if j >= n_cols:
            j = 0
            i += 1
            
    # Special case: empty figures at the end of iteration
    i, j = 0, 0
    for n_plots in range(n_rows * n_cols):

        # If axis index is greater than number of features, hides the axis border
        if n_plots >= len(col_list):
            try:
                axs[i][j].axis('off')
            except TypeError as e:
                axs[j].axis('off')

        # Stepping up index
        j += 1
        if j == n_cols:
            j = 0
            i += 1 
    
    # Tighting layout
    plt.tight_layout()

    # Saving image if applicable
    if 'save' in kwargs and bool(kwargs['save']):
        output_path = kwargs['output_path'] if 'output_path' in kwargs else 'output/'
        img_name = kwargs['img_name'] if 'img_name' in kwargs else f'multiple_{kind}plot.png'
        save_fig(fig=fig, output_path=output_path, img_name=img_name)
  
# Multiple distplots with scatterplots (regplot)
def plot_multiple_dist_scatterplot(df, col_list, y_col, dist_kind='dist', scatter_kind='reg', 
                                   **kwargs):
    """
    Plots a rich graph that joins a distribution and a scatterplot
    
    Parameters
    ----------
    :param df: dataset used for plotting [type: pd.DataFrame]
    :param col_list: list with numerical columns to be used on analysis [type: list]
    :param y_col: y columns to be used on y axis of the scatterplot [type: string]
    :param dist_kind: kind of distribution plot [type: string, default='dist']
        *options: ['dist', 'kde', 'box', 'boxen', 'strip']
    :param scatter_kind: kind of scatter correlation plot [type: string, default='reg']
        *options: ['scatter', 'reg']
    :param **kwargs: additional parameters  
        :arg figsize: figure dimension [type: tuple, default=(10, n_rows * 5)]
            *where n_rows = ceil(len(col_list) / n_cols)
        :arg title1: distribution plot title [type: string, default=f'{dist_kind.title()} for Feature {col}']
        :arg title2: scatterplot title [type: string, default=f'{scatter_kind.title()}plot Between {col} and {y_col}']
        :arg hue: breaks the chart into another category (seaborn hue function arg) [type: string, default=None]
        :arg hist: plots histogram bars on the chart (seaborn's parameter) [type: bool, default=False]
        :arg kde: plots kde line on the chart (seaborn's parameter) [type: bool, default=True]
        :arg rug: plots rug at the bottom of the the chart (seaborn's parameter) [type: bool, default=False]
        :arg shade: fills the area below distribution curve (seaborn's parameter) [type: bool, default=True]
        :arg color: color of the distribution line [type: string, default='darkslateblue']
        :arg palette: color palette to be used on the chart [type: string, default='rainbow']
        :arg size_title: title size [type: int, default=16]
        :arg alpha: transparency of scatterplot circles [type: float, default=.7]
        :arg save: flag for saving the image created [type: bool, default=None]
        :arg output_path: path for image to be saved [type: string, default='output/']
        :arg img_name: filename for image to be saved [type: string, default=f'{dist_kind}_{scatter_kind}plot.png']
    
    Return
    ------
    This function returns nothing besides plotting the dist and scatter combination for multiple columns

    Application
    -----------
    plot_multiple_dist_scatterplot(df=df, col_list['colA', 'colB'], y_col='numeric_col')
    """
    
    # Validating kinds in dist and scatter plot configuration
    possible_dist_kinds = ['dist', 'kde', 'box', 'boxen', 'strip']
    possible_scatter_kinds = ['scatter', 'reg']
    if dist_kind not in possible_dist_kinds:
        print(f'Invalid dist_kind parameter. Possible options: {possible_kinds}')
        return
    if scatter_kind not in possible_scatter_kinds:
        print(f'Invalid scatter_kind parameter. Possible options: {possible_scatter_kinds}')
        return
    
    # Validating y_col on list of dataset columns
    if y_col not in list(df.columns):
        print(f'There is no {y_col} on the given daset. Please change "y_col" parameter')
        return
    
    # Computing figure parameters
    n_rows = len(col_list)
    figsize = kwargs['figsize'] if 'figsize' in kwargs else (17, n_rows * 5)
    i = 0
    
    # Extracting parameters for plotting the graphs
    hue = kwargs['hue'] if 'hue' in kwargs else None
    hist = kwargs['hist'] if 'hist' in kwargs else True
    kde = kwargs['kde'] if 'kde' in kwargs else True
    rug = kwargs['rug'] if 'rug' in kwargs else False
    shade = kwargs['shade'] if 'shade' in kwargs else True
    color = kwargs['color'] if 'color' in kwargs else 'darkslateblue'
    palette = kwargs['palette'] if 'palette' in kwargs else 'rainbow'
    size_title = kwargs['size_title'] if 'size_title' in kwargs else 16
    
    # Creating figure and iteration over the multiple axis
    fig, axs = plt.subplots(nrows=n_rows, ncols=2, figsize=figsize)
    for col in col_list:
        # Indexing axis on the first column (j = 0) for distribution chart
        try:
            ax = axs[i, 0]
        except IndexError as ie:
            # For one-dimensional figure, it's necessary to index axis as a vector and not as a matrix
            ax = axs[0]
        title1 = kwargs['title1'] if 'title1' in kwargs else f'{dist_kind.title()} for Feature {col}'
        
        # Plotting a distribution chart using plot_distplot xplotter function
        plot_distplot(df=df, col=col, ax=ax, kind=dist_kind, hue=hue, hist=hist, kde=kde, rug=rug,
                      shade=shade, color=color, palette=palette, title=title1, size_title=size_title)
        
        # Indexing axis on the second column (j = 1) for scatterplot
        try:
            ax2 = axs[i, 1]
        except IndexError as ie:
            # For one-dimensional figure, it's necessary to index axis as a vector and not as a matrix
            ax2 = axs[1]
        alpha = kwargs['alpha'] if 'alpha' in kwargs else .7
        
        # Scatterplot
        if scatter_kind == 'scatter':
            sns.scatterplot(x=col, y=y_col, data=df, color=color, ax=ax2)
            
        # Regplot
        if scatter_kind == 'reg':
            sns.regplot(x=col, y=y_col, data=df, color=color, ax=ax2)
        
        # Stepping up i index
        i += 1
        
        # Customizing chart
        format_spines(ax2, right_border=False)
        title2 = kwargs['title2'] if 'title2' in kwargs else f'{scatter_kind.title()}plot Between {col} and {y_col}'
        ax2.set_title(title2, size=size_title)
        if dist_kind in ['dist', 'kde'] and hue is not None:
            ax.legend(title=hue)

    # Tighting layout        
    plt.tight_layout()

    # Saving image if applicable
    if 'save' in kwargs and bool(kwargs['save']):
        output_path = kwargs['output_path'] if 'output_path' in kwargs else 'output/'
        img_name = kwargs['img_name'] if 'img_name' in kwargs else f'{dist_kind}_{scatter_kind}plot.png'
        save_fig(fig=fig, output_path=output_path, img_name=img_name)

# Multiple countplots
def plot_multiple_countplots(df, col_list, n_cols=3, **kwargs):
    """
    Plots multiple formatted countplot based on a list of columns of a given dataset
    
    Parameters
    ----------
    :param df: dataset used for plotting [type: pd.DataFrame]
    :param col_list: list with numerical columns to be used on analysis [type: list]
    :param n_cols: figure columns configured for this multiple graph [type: int, default=3]
    :param **kwargs: additional parameters
        :arg figsize: figure dimension [type: tuple, default=(10, n_rows * 5)]
            *where n_rows = ceil(len(col_list) / n_cols)
        :arg hue: breaks the chart into another category (seaborn hue function arg) [type: string, default=None]
        :arg palette: color palette to be used on the chart [type: string, default='rainbow']
        :arg order: sorts categories into a defined order [type: bool, default=None]
        :arg orient: horizontal or vertical orientation [type: string, default='v']
        :arg size_title: title size [type: int, default=16]
        :arg size_labels: label size [type: int, default=14]
        :arg top: filter the top N categories on the chart [type: int, default=-1]
        :arg save: flag for saving the image created [type: bool, default=None]
        :arg output_path: path for image to be saved [type: string, default='output/']
        :arg img_name: filename for image to be saved [type: string, default=f'multiple_countplots.png']

    Return
    ------
    This function returns nothing besides plotting multiple countplot charts

    Application
    -----------
    plot_multiple_countplots(df=df, col_list=['colA', 'colB'])
    """
    
    # Setting up figure for multiple plots
    n_rows = ceil(len(col_list) / n_cols)
    figsize = kwargs['figsize'] if 'figsize' in kwargs else (17, n_rows * 5)
    i, j = 0, 0
    
    # Extracting additional parameters for the charts
    hue = kwargs['hue'] if 'hue' in kwargs else None
    palette = kwargs['palette'] if 'palette' in kwargs else 'rainbow'
    order = df[col].value_counts().index if 'order' in kwargs and bool(kwargs['order']) else None
    orient = kwargs['orient'] if 'orient' in kwargs and kwargs['orient'] in ['h', 'v'] else 'v'
    size_title = kwargs['size_title'] if 'size_title' in kwargs else 12
    size_labels = kwargs['size_labels'] if 'size_labels' in kwargs else 12
    top = kwargs['top'] if 'top' in kwargs else -1
    
    # Creating figure and iteration over the multiple axis generated
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)
    for col in col_list:
        # Indexing axis
        try:
            ax = axs[i, j]
        except IndexError as ie:
            # For one-dimensional figure, it's necessary to index axis as a vector and not as a matrix
            ax = axs[j]
        
        title = kwargs['title'] if 'title' in kwargs else f'Countplot for Feature {col}'
        
        # Building the countplot chart by calling plot_countplot function using parameters associated
        plot_countplot(df=df, col=col, ax=ax, hue=hue, palette=palette, order=order, title=title,
                       orient=orient, size_title=size_title, size_labels=size_labels, top=top)
                    
        # Stepping up index
        j += 1
        if j == n_cols:
            j = 0
            i += 1
            
    # Special case: empty figures at the end of iteration
    i, j = 0, 0
    for n_plots in range(n_rows * n_cols):

        # If axis index is greater than number of features, eliminates the axis border
        if n_plots >= len(col_list):
            try:
                axs[i][j].axis('off')
            except TypeError as e:
                axs[j].axis('off')

        # Stepping up index
        j += 1
        if j == n_cols:
            j = 0
            i += 1 
    
    # Tighting layout
    plt.tight_layout()      

    # Saving image if applicable
    if 'save' in kwargs and bool(kwargs['save']):
        output_path = kwargs['output_path'] if 'output_path' in kwargs else 'output/'
        img_name = kwargs['img_name'] if 'img_name' in kwargs else f'multiple_countplots.png'
        save_fig(fig=fig, output_path=output_path, img_name=img_name)


"""
---------------------------------------------------
---------- 2. CUSTOM GRAPHICAL ANALYSIS -----------
        2.7 Line charts for evolution plots
---------------------------------------------------
"""

# A date-like evolution plot
def plot_evolutionplot(df, x, y, agg=True, agg_functions=['count', 'sum', 'mean'], 
                          agg_type='count', **kwargs):
    """
    Plots an evolution plot in a line chart. This function is main useful when passed a date
    column referente as x parameter. Optionally, it's possible to configure de function
    to aggregate values internaly.
    
    Parameters
    ----------
    :param df: dataset used for plotting [type: pd.DataFrame]
    :param x: column reference for x axis (date representation) [type: string]
    :param y: column reference for y axis (aggregation) [type: string]
    :param agg: flag for applying aggregation proccess inside function [type: bool, default=True]
    :param agg_functions: aggregators list to be applied [type: list, default=['count', 'sum', 'mean']]
    :param agg_type: aggregation type for plotting the line chart [type: string, default='count']
    :param **kwargs: additional parameters
        :arg hue: breaks the chart into another category [type: string, default=None]
        :arg str_col: flag for changing dtype of x column into string [type: bool, default=True]
        :arg date_col: flag for changing dtype of x column into date [type: bool, default=True]
        :arg date_fmt: in case of date transformation, this parameter sets de format [type: string, default='%Y%m']
        :arg figsize: figure dimension [type: tuple, default=(17, 7)]
        :arg ax: matplotlib axis in case of external figure defition [type: mpl.Axes, default=None]
        :arg color: color of the distribution line [type: string, default='darkslateblue']
        :arg palette: color palette to be used on the chart [type: string, default='rainbow_r']
        :arg markers: flag para putting markers on line chart [type: bool, default=True]
        :arg style: seaborn style function parameter [type: string, default=None]
        :arg size: seaborn size function parameter [type: string, default=None]
        :arg sort: flag for sorting the data [type: bool, default=False]
        :arg x_rot: x axis label rotation [type: int, default=90]
        :arg title: chart title [type: string, default=f'Lineplot of {agg_type.title()} of {y_ori} by {x}']
        :arg label_data: flag for putting labels on data points [type: bool, default=True]
        :arg label_aggreg: label aggregation if it's too big [type: string, default='K']
            *options: ['', 'K', 'M', 'B']
    
    Returns
    -------
    This function returns nothing besides the evolution plot

    Application
    -----------
    plot_evolutionplot(df=df, x='date_col', y='num_col', agg_type='sum', date_col=False, x_rot=0)
    """
    
    # Defining an aggregation function to be used inside function
    def make_aggregation(df, group_col, value_col, agg_functions=['count', 'sum', 'mean'], **kwargs):
        # Grouping data using the agg_functions list
        agg_dict = {value_col: agg_functions}
        return df.groupby(by=group_col, as_index=False).agg(agg_dict)
    
    # Verifying columns x and y inside dataset columns
    y_ori = y
    if x not in df.columns:
        print(f'Column "x"={x} is not in the given dataset')
        return
    if y not in df.columns:
        print(f'Column "y"={y} is not in the given dataset')
        return
    
    # Extracting hue parameter and creating a "dummy" column with it if applicable
    hue = kwargs['hue'] if 'hue' in kwargs else None
    if hue:
        df[hue] = df[hue].astype(str)
    
    # Applying aggregation on data if applicable
    if agg:
        group_col = [x, hue] if hue else x
        new_columns = [x, hue] + agg_functions if hue else [x] + agg_functions
        df_group = make_aggregation(df=df, group_col=group_col, value_col=y, agg_functions=agg_functions)
        
        # Aggregating based no parameteres passed
        if agg_type not in agg_functions:
            print(f'Parameter "agg_type={agg_type}" is not present on "agg_functions={agg_functions}"')
            return
        
        # Updatig y colum and df according to aggregation result
        df_group.columns = new_columns
        y = agg_type
    else:
        df_group = df     
        
    # Extracting transformation parameters of the columns
    str_col = kwargs['str_col'] if 'str_col' in kwargs else True
    date_col = kwargs['date_col'] if 'date_col' in kwargs else True
    date_fmt = kwargs['date_fmt'] if 'date_fmt' in kwargs else '%Y%m'
    
    # Transforming columns (string and date)
    if str_col:
        df_group[x] = df_group[x].astype(str)
    if date_col:
        try:
            df_group[x] = df_group[x].apply(lambda st: datetime.strptime(str(st), date_fmt))
        except ValueError as ve:
            print(f'{ve}. Change "date_fmt" parameter or set "date_col=False"\n')
    
    # Extracting chart parameters
    figsize = kwargs['figsize'] if 'figsize' in kwargs else (17, 7)
    ax = kwargs['ax'] if 'ax' in kwargs else None
    color = kwargs['color'] if 'color' in kwargs else 'darkslateblue'
    palette = kwargs['palette'] if 'palette' in kwargs else 'rainbow_r'
    markers = kwargs['markers'] if 'markers' in kwargs else True
    style = kwargs['style'] if 'style' in kwargs else None
    size = kwargs['size'] if 'size' in kwargs else None
    sort = kwargs['sort'] if 'sort' in kwargs else False
    
    # Plotting graph using seaborn's lineplot function
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(x=x, y=y, data=df_group, ax=ax, hue=hue, color=color, palette=palette, 
                 markers=markers, style=style, size=size, sort=sort)
    
    # Extracting parameters for customizing the chart
    x_rot = kwargs['x_rot'] if 'x_rot' in kwargs else 90
    title = kwargs['title'] if 'title' in kwargs else f'Lineplot - {agg_type.title()} de {y_ori} por {x}'
    label_data = kwargs['label_data'] if 'label_data' in kwargs else True
    label_aggreg = kwargs['label_aggreg'] if 'label_aggreg' in kwargs else 'K'
    label_aggreg_options = ['', 'K', 'M', 'B']
    if label_aggreg not in label_aggreg_options:
        print(f'Parameter "label_aggreg" {label_aggreg} must be one of {label_aggreg_options}. Reverting to ""')
        label_aggreg = ''   
    label_aggreg_dict = {'': 1, 'K': 1000, 'M': 1000000, 'B': 1000000000}
    label_aggreg_value = label_aggreg_dict[label_aggreg]
    
    # Customizing chart
    format_spines(ax, right_border=False)
    for tick in ax.get_xticklabels():
        tick.set_rotation(x_rot)
    ax.set_title(title, size=16)
    
    # Labeling data
    if label_data:
        for x, y in zip(df_group[x], df_group[y]):
            ax.annotate(str(round(y/label_aggreg_value, 2))+label_aggreg, xy=(x, y), 
                        textcoords='data', ha='center', va='center', color='dimgrey')

    # Tighting layout
    plt.tight_layout()

    # Saving image if applicable
    if 'save' in kwargs and bool(kwargs['save']):
        output_path = kwargs['output_path'] if 'output_path' in kwargs else 'output/'
        img_name = kwargs['img_name'] if 'img_name' in kwargs else f'{x}_{y}evlplot.png'
        save_fig(fig=fig, output_path=output_path, img_name=img_name)
