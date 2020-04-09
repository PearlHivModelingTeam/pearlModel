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
table_name = 'loss_to_follow_up_table'


with pd.HDFStore(param_dir + '/parameters.h5') as store:
    data = store['loss_to_follow_up']

data = data.reset_index()
data = data[['group', 'intercept', 'year', 'sqrtcd4n', 'haart_period', 'age', '_age', '__age', '___age']]
data = data.rename(columns={'group': 'Population',
                            'intercept': '$\\beta_0$',
                            'age': '$\\beta_\mathrm{age}$',
                            '_age': '$\\beta_\mathrm{age\_1}$',
                            '__age': '$\\beta_\mathrm{age\_2}$',
                            '___age': '$\\beta_\mathrm{age\_3}$',
                            'year': '$\\beta_\mathrm{year}$',
                            'sqrtcd4n': '$\\beta_\mathrm{init\_sqrt\_cd4}$',
                            'haart_period': '$\\beta_\mathrm{art\_period}$'})
data = data.round(decimals=3)
if not os.path.isfile(f'out/{table_name}.xlsx'):
    data.to_excel(f'out/{table_name}.xlsx', index=False)
colWidths = 1 * [1.5] + 8 *[1]

render_mpl_table(data, header_columns=0, col_width=2.6, header_color='#347DBE', colWidths=colWidths)
plt.savefig(f'out/{table_name}.png', bbox_inches='tight', dpi=300)
