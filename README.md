<h1 align="center">
  <a href="https://pypi.org/project/xplotter/"><img src="https://i.imgur.com/5XFP1Ha.png" height=400, weight=400, alt="xplotter logo"></a>
</h1>

<div align="center">
  <strong>:bar_chart: Gathering insights from data in a complete EDA process components :chart_with_upwards_trend:</strong>
</div>
<br/>

<div align="center">  
  
  ![Release](https://img.shields.io/badge/release-ok-brightgreen)
  [![PyPI](https://img.shields.io/pypi/v/xplotter?color=blueviolet)](https://pypi.org/project/xplotter/)
  ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/xplotter?color=green)
  ![PyPI - Status](https://img.shields.io/pypi/status/xplotter)

</div>
<br/>


## Table of content

- [About xplotter](#about-xplotter)
  - [Package structure](#package-structure)
  - [Features](#features)
- [Installing the package](#installing-the-package)
- [Examples](#examples)
  - [Insights module](#insights-module)
  - [Usage around the world](#usage-around-the-world)
- [Contribution](#contribution)
- [Social Media](#social-media)

___

## About xplotter

The definition is clear: `xplotter` is a python library created for making the Exploratory Data Analysis process so much easier! With `xplotter`, data scientists and data analysts can use a vast number of functions for plotting, customizing and extracting insights from data with very few lines of code. The exploratory process is a key step on every data science and business inteligence project and it's very important to understand your data before take any action. The use cases are almost infinity!

**Why use xplotter?**
* _Use functions for plotting graphs and extracting information from your data in an easy way_
* _You can explore your data faster_
* _Visualize some beautiful charts with few lines of code_
* _Save your images in a local repository_
* _Improve analytics by analysing trends and distributions of your data_
* _Rich documentation to explore_

___

### Package structure

After viewing some of benefits of using `xplotter` in a data project, it's also important to see how the package was built and how it's organized. At the moment, there are two modules on the xplotter package folder and they are explained on the table below:

| Module      | Description                                                   | Functions/Methods | Lines of Code (approx) |
| :---------: | :-----------------------------------------------------------: | :---------------: | :--------------------: |
| `formatter` | Auxiliar functions for formatting charts                      |         3         |         ~150           |
| `insights`  | Functions for exploring data in a wide range of possibilities |        14         |         ~1800          |

___

### Features

The features of xplotter package are built into useful and well-documented functions that can be used in any step of data exploration process. There will be a specific session of usage examples in this documentations but, just be clear, you can use xplotter for a simple formatting step like customizing the border axis in a matplotlib graph...

```python
from xplotter.formatter import format_spines

fig, ax = plt.subplots(figsize=(10, 7))
format_spines(ax, right_border=False)
```

...or even plot a simple and customized countplot with labels already written inside the bars...

```python
from xplotter.insights import plot_countplot

plot_countplot(df=df, col='cat_column')
```

At this moment, all the features available in the xplotter package are:


| Module      | Function/Class                   | Short Description                                                                          |
| :---------: | :------------------------------: | :----------------------------------------------------------------------------------------: |
| `formatter` | `format_spines`                  | Modify borders and axis colors of matplotlib figure                                        |
| `formatter` | `AnnotateBars`                   | Makes the process of labeling data points in a bar chart easier                            |
| `formatter` | `make_autopct`                   | Helps labeling data in a pie or donut chart                                                |
| `insights`  | `save_fig`                       | Easy way for saving figures created inside or outside xplotter                             |
| `insights`  | `plot_donutchart`                | Creates a custom donut chart for a specific categorical column                             |
| `insights`  | `plot_pie_chart`                 | Creates a custom pie chart for a specific categorical column                               |
| `insights`  | `plot_double_donut_chart`        | Creates a "double" custom donut chart for two columns of a given dataset                   |
| `insights`  | `plot_countplot`                 | Creates a simple countplot using a dataset and a column name                               |
| `insights`  | `plot_pct_countplot`             | Creates a percentage countplot (grouped bar chart) using a dataset and a column name       |
| `insights`  | `plot_distplot`                  | Creates a custom distribution plot based on a numeric column                               |
| `insights`  | `plot_aggregation`               | Plots a custom aggregate chart into a bar style                                            |
| `insights`  | `plot_cat_aggreg_report`         | A rich and complete report using count, aggregation and distribution functions             |
| `insights`  | `data_overview`                  | Extract useful information of a given dataset to offers an overview from the data          |
| `insights`  | `plot_corr_matrix`               | A beautiful and customized correlation matrix for a dataset and a target column            |
| `insights`  | `plot_multiple_distplots`        | Plots custom distribution charts for multiple columns at once using the col_list parameter |
| `insights`  | `plot_multiple_dist_scatterplot` | Plots a rich graph that joins a distribution and a scatterplot                             |
| `insights`  | `plot_multiple_countplots`       | Plots multiple formatted countplot based on a list of columns of a given dataset           |
| `insights`  | `plot_evolutionplot`             | Plots an evolution plot in a line chart                                                    |

___

## Installing the package

The last version of `xplotter` package are published and available on [PyPI repository](https://pypi.org/project/xplotter/)

> :pushpin: **Note:** as a good practice for every Python project, the creation of a <a href="https://realpython.com/python-virtual-environments-a-primer/">virtual environment</a> is needed to get a full control of dependencies and third part packages on your code. By this way, the code below can be used for creating a new venv on your OS.
> 

```bash
# Creating and activating venv on Linux
$ python -m venv <path_venv>/<name_venv>
$ source <path_venv>/<nome_venv>/bin/activate

# Creating and activating venv on Windows
$ python -m venv <path_venv>/<name_venv>
$ <path_venv>/<nome_venv>/Scripts/activate
```

With the new venv active, all you need is execute the code below using pip for installing xplotter package (upgrading pip is optional):

```bash
$ pip install --upgrade pip
$ pip install xplotter
```

The xplotter package is built in a layer above some other python packages like matplotlib, seaborn and pandas. Because of that, when installing xplotter, the pip utility will also install all dependencies linked to xplotter. The output expected on cmd or terminal are something like:

```
Installing collected packages: six, pytz, python-dateutil, pyparsing, numpy, kiwisolver, cycler, scipy, pandas, matplotlib, seaborn, xplotter
Successfully installed cycler-0.10.0 kiwisolver-1.3.1 matplotlib-3.2.1 numpy-1.20.2 pandas-1.1.5 pyparsing-2.4.7 python-dateutil-2.8.1 pytz-2021.1 scipy-1.6.3 seaborn-0.11.1 six-1.15.0 xplotter-0.0.3
```
___

## Examples

In this session, you will see some usage examples of xplotter on real problems using data. After installing the package, it's important to know how to use it and how to extract the real power of it.

### Insights Module

As you could see by this time, the insights module from xplotter have a lot of functions that can deliver something like "charts as a code". It's just a funy way to describe how the components and the tools inside this module can make life of a data explorer easier. To ilustrate that, the code below reads the titanic dataset directly from seaborn and plots a custom donut chart for _survived_ feature.

```python
from xplotter.insights import plot_donut_chart
import seaborn as sns

titanic = seaborn.load_dataset('titanic')
plot_donut_chart(df=titanic, col='survived')
```
This outputs the following figure:

<div align="center">
   <img src="https://i.imgur.com/0pia2bm.png" width=450 height=450>
</div>

Another example uses the iris dataset also read from seaborn `load_dataset()` function. In this case, we can use the `plot_distplot()` xplotter funcion for visualize the distribution of sepal_length feature for each species in a fast and direct way:

```python
iris = seaborn.load_dataset('iris')
plot_distplot(df=iris, col='sepal_length', hue='species')
```

Let's see the beautiful distribution plot the function above generated for you:

<div align="center">
   <img src="https://i.imgur.com/Q2A2mgk.png" height=450 width=700>
</div>

There are much more you can do using insights module from xplotter package. It's a hole world to be explore and a bunch of business questions that can be answeared through visualization charts easy plotted through those already coded functions. In the next topic, this documentation will share with you some of real world implementations using xplotter.
___

### Usage Around the World

For being easy and simple, xplotter can have alot of applications. One of the most famous one is the notebook <a href="https://www.kaggle.com/thiagopanini/exploring-and-predicting-survival-on-titanic/comments">Titanic Dataset Exploration</a> posted on Kaggle by <a href="https://www.kaggle.com/thiagopanini">Thiago Panini</a>. This well written notebook uses the insight module for plotting beautiful charts and building a really complete Exploratory Data Analysis proccess and, by now, it achieve a <b>bronze medal</b> with 32 upvotes by Kaggle's community and a incredible mark of more than 1,600 views!

<div align="center">
   <img src="https://i.imgur.com/q1DnARQ.png" height=90 width=900>
</div>

___

## Contribution

The xplotter python package is an open source implementation and the more people use it, the more happy the developers will be. So if you want to contribute with xplotter, please feel free to follow the best practices for implementing coding on this github repository through creating new branches, making merge requests and pointig out whenever you think there is a new topic to explore or a bug to be fixed.

Thank you very much for reaching this and it will be a pleasure to have you as xplotter user or developer.

___

## Social Media

Follow me on LinkedIn: https://www.linkedin.com/in/thiago-panini/
See my other Python packages: https://github.com/ThiagoPanini
