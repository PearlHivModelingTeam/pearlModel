# Imports
import feather
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import six
import seaborn as sns
from matplotlib import rc
#rc('text', usetex=True)

def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, colWidths=None, **kwargs):

    if ax is None:
        if colWidths is None:
            size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
            fig, ax = plt.subplots(figsize=size)
            ax.axis('off')
        else:
            y = np.array(data.shape[::-1])[-1]
            size = np.array([16, y*row_height])
            print(size)
            fig, ax = plt.subplots(figsize=size)
            ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, colWidths=colWidths, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax

pd.set_option("display.max_rows", 1001)

# Define directories
cwd = os.getcwd()
param_dir = cwd + '/../../../../../data/parameters'


with pd.HDFStore(param_dir + '/parameters.h5') as store:
    data = store['h1yy_by_age_2009']

data = data[['pct']].reset_index()
data['h1yy'] = data['h1yy'].astype(int)
data = pd.pivot_table(data, index=['group', 'age2009cat'], values='pct', columns='h1yy')
del data.columns.name
data = data.reindex(index=pd.MultiIndex.from_product([data.index.levels[0], np.arange(1, 8)], names=['Population', 'Age Category']))

data = data.round(decimals=3)
#data.to_excel('out/art_init_table.xlsx')
data = data.reset_index()

group_dict = {'het_male': ['het_black_male', 'het_hisp_male', 'het_white_male'],
              'het_female': ['het_black_female', 'het_hisp_female', 'het_white_female'],
              'idu_male': ['idu_black_male', 'idu_hisp_male', 'idu_white_male'],
              'idu_female': ['idu_black_female', 'idu_hisp_female', 'idu_white_female'],
              'msm_male': ['msm_black_male', 'msm_hisp_male', 'msm_white_male']}

colWidths = 2 * [2] + 10 * [1]
data['Race'] = data['Population'].str.split(pat='_', expand=True)[[1]]
data['Race'] = data['Race'].str.capitalize()
print(data)

for key in group_dict:
    group_names = group_dict[key]
    df = data.loc[data['Population'].isin(group_names)].copy()
    df = df[['Race', 'Age Category', 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009]]
    df.loc[df['Race']=='Hisp', 'Race'] = 'Hispanic'
    df = df.replace(np.nan, '-', regex=True)

    render_mpl_table(df, header_columns=0, col_width=2.6, header_color='#347DBE', colWidths=colWidths)
    plt.savefig(f'out/{key}.png', bbox_inches='tight', dpi=300)
