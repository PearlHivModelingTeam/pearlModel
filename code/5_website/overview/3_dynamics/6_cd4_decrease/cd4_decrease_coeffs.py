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
    data = store['cd4_decrease']

data = data.reset_index()

data = data.rename(columns={'group': 'Population',
                            'intercept': '$\\beta_0$',
                            'time_out_of_naaccord': '$\\beta_\mathrm{years\_out}$',
                            'sqrtcd4n_exit': '$\\beta_\mathrm{sqrt\_cd4\_exit}$'})

data['Population'] = 'All'
print(data)

table_name = 'cd4_decrease_table'
data = data.round(decimals=3)
if not os.path.isfile(f'out/{table_name}.xlsx'):
    data.to_excel(f'out/{table_name}.xlsx', index=False)
colWidths = 4 *[1]

render_mpl_table(data, header_columns=0, col_width=2.6, header_color='#347DBE', colWidths=colWidths)
plt.savefig(f'out/{table_name}.png', bbox_inches='tight', dpi=300)

