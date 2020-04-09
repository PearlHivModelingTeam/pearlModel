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
    data = store['cd4_increase']

data = data.reset_index()

data1 = data[['group', 'intercept', 'agecat', 'cd4cat349', 'cd4cat499', 'cd4cat500', 'time_from_h1yy', '_time_from_h1yy', '__time_from_h1yy', '___time_from_h1yy']].copy()
data2 = data[['group', '_timecd4cat349', '__timecd4cat349', '___timecd4cat349', '_timecd4cat499', '__timecd4cat499', '___timecd4cat499', '_timecd4cat500', '__timecd4cat500', '___timecd4cat500']].copy()

data1 = data1.rename(columns={'group': 'Population',
                              'intercept': '$\\beta_0$',
                              'agecat': '$\\beta_\mathrm{age\_cat}$',
                              'cd4cat349': '$\\beta_\mathrm{cd4\_cat\_1}$',
                              'cd4cat499': '$\\beta_\mathrm{cd4\_cat\_2}$',
                              'cd4cat500': '$\\beta_\mathrm{cd4\_cat\_3}$',
                              'time_from_h1yy': '$\\beta_\mathrm{yrs\_art}$',
                              '_time_from_h1yy': '$\\beta_\mathrm{yrs\_art\_1}$',
                              '__time_from_h1yy': '$\\beta_\mathrm{yrs\_art\_2}$',
                              '___time_from_h1yy': '$\\beta_\mathrm{yrs\_art\_3}$'})

data = data1.copy()
table_name = 'cd4_increase_table_a'
data = data.round(decimals=2)
if not os.path.isfile(f'out/{table_name}.xlsx'):
    data.to_excel(f'out/{table_name}.xlsx', index=False)
colWidths = 1 * [1.7] + 9 *[1]

render_mpl_table(data, header_columns=0, col_width=2.6, header_color='#347DBE', colWidths=colWidths)
plt.savefig(f'out/{table_name}.png', bbox_inches='tight', dpi=300)

data2 = data2.rename(columns={'group': 'Population',
                              '_timecd4cat349': '$\\beta_\mathrm{cd4\_1\_yrs\_1}$',
                              '__timecd4cat349': '$\\beta_\mathrm{cd4\_1\_yrs\_2}$',
                              '___timecd4cat349': '$\\beta_\mathrm{cd4\_1\_yrs\_3}$',
                              '_timecd4cat499': '$\\beta_\mathrm{cd4\_2\_yrs\_1}$',
                              '__timecd4cat499': '$\\beta_\mathrm{cd4\_2\_yrs\_2}$',
                              '___timecd4cat499': '$\\beta_\mathrm{cd4\_2\_yrs\_3}$',
                              '_timecd4cat500': '$\\beta_\mathrm{cd4\_3\_yrs\_1}$',
                              '__timecd4cat500': '$\\beta_\mathrm{cd4\_3\_yrs\_2}$',
                              '___timecd4cat500': '$\\beta_\mathrm{cd4\_3\_yrs\_3}$'})

data = data2.copy()
table_name = 'cd4_increase_table_b'
data = data.round(decimals=2)
if not os.path.isfile(f'out/{table_name}.xlsx'):
    data.to_excel(f'out/{table_name}.xlsx', index=False)
colWidths = 1 * [1.7] + 9 *[1]

render_mpl_table(data, header_columns=0, col_width=2.6, header_color='#347DBE', colWidths=colWidths)
plt.savefig(f'out/{table_name}.png', bbox_inches='tight', dpi=300)

