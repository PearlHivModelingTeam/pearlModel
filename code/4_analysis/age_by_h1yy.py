import os
import feather
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
pd.set_option("display.max_rows", 1001)

def mixture(coeffs, x):
    if coeffs.loc['lambda1', 'estimate'] > 0.0:
        rv1 = stats.norm(loc=coeffs.loc['mu1', 'estimate'], scale=coeffs.loc['sigma1', 'estimate'])
    else:
        rv1 = None
    rv2 = stats.norm(loc=coeffs.loc['mu2', 'estimate'], scale=coeffs.loc['sigma2', 'estimate'])
    lambda1 = coeffs.loc['lambda1', 'estimate']
    if rv1 is not None:
        y = lambda1 * rv1.pdf(x) + (1.0 - lambda1) * rv2.pdf(x)
    else:
        y = rv2.pdf(x)
    return (y)

def msm_mix(data, n, h1yy, ages):
    n0 = n[0].loc[h1yy].values
    n1 = n[1].loc[h1yy].values
    n2 = n[2].loc[h1yy].values
    N = n0 + n1 + n2
    return (n0 / N) * mixture(data[0].loc[h1yy], ages) + (n1 / N) * mixture(data[1].loc[h1yy], ages) + (n2 / N) * mixture(data[2].loc[h1yy], ages)

# Prepare aggregated MSM data
group_names = ['msm_white_male', 'msm_black_male', 'msm_hisp_male']
replications = 100
ages = np.arange(18, 85, 1.0)

data = []
n = []
for group_name in group_names:
    total = pd.DataFrame()
    n_total = pd.DataFrame()
    for replication in range(replications):
        n_df = feather.read_dataframe(f'{os.getcwd()}/../../out/art_init_sim/{group_name}_{replication}.feather')
        n_total = n_total.append(n_df)
        rep_df = feather.read_dataframe(f'{os.getcwd()}/../../out/age_by_h1yy/{group_name}_{replication}.feather')
        total = total.append(rep_df)
    means = total.groupby(['h1yy', 'term']).mean()
    n_means = n_total.groupby('year').mean()
    data.append(means)
    n.append(n_means)

# Set seaborn style
sns.set(style='darkgrid')
sns.set_context('paper', font_scale = 2.1, rc={'lines.linewidth':3.0})
dpi = 300

# Make single aggregated plot
fig, ax = plt.subplots(figsize=(8.0, 6.0))
colors = sns.color_palette('Blues_d', 5)
df = pd.DataFrame()
for h1yy in [2010, 2015, 2020, 2025, 2030]:
    df = df.append(pd.DataFrame({'Age': ages, 'Proportion': msm_mix(data, n, h1yy, ages), 'Year': len(ages)*[h1yy]}))

g = sns.lineplot(ax=ax, x='Age', y='Proportion', hue='Year', data=df, legend='full',palette=colors)
plt.title("MSM Age at ART Initiation")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[1:], labels=labels[1:], frameon=False)
g.set(ylim=(-0.002, 0.058))
g.set(xlim=(16, 88))
plt.savefig( f'{os.getcwd()}/../../out/fig/age_by_h1yy/msm/msm_all.png', dpi=dpi, bbox_inches='tight')

# Make aggregated plot for each year
i = 0
for h1yy in [2010, 2015, 2020, 2025, 2030]:
    color = colors[i]
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    g = sns.lineplot(ax=ax, x='Age', y='Proportion', data=df.loc[df['Year'] == h1yy], hue='Year', palette=[color])
    i += 1
    g.set(ylim=(-0.002, 0.058))
    g.set(xlim=(16, 88))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:], frameon=False)
    plt.title(f'MSM Age at ART Initiation')
    plt.savefig( f'{os.getcwd()}/../../out/fig/age_by_h1yy/msm/msm_{h1yy}.png', dpi=dpi, bbox_inches='tight')

group_names = ['msm_white_male', 'msm_black_male', 'msm_hisp_male', 'idu_white_male', 'idu_black_male',
               'idu_hisp_male', 'idu_white_female', 'idu_black_female', 'het_white_male',
               'het_black_male', 'het_hisp_male', 'het_white_female', 'het_black_female', 'het_hisp_female']
for group_name in group_names:
    total = pd.DataFrame()
    for replication in range(replications):
        rep_df = feather.read_dataframe(f'{os.getcwd()}/../../out/age_by_h1yy/{group_name}_{replication}.feather')
        total = total.append(rep_df)
    means = total.groupby(['h1yy', 'term']).mean()

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    colors = sns.color_palette('Blues_d', 5)
    df = pd.DataFrame()
    for h1yy in [2010, 2015, 2020, 2025, 2030]:
        df = df.append(pd.DataFrame({'Age': ages, 'Proportion': mixture(means.loc[h1yy], ages), 'Year': len(ages)*[h1yy]}))

    #print(df)
    g = sns.lineplot(ax=ax, x='Age', y='Proportion', hue='Year', data=df.reset_index(), legend='full', palette=colors)
    g.set(ylim=(-0.002, 0.122))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:], frameon=False)
    plt.title(group_name)
    plt.savefig( f'{os.getcwd()}/../../out/fig/age_by_h1yy/individual/{group_name}.png', dpi=dpi, bbox_inches='tight')

# idu_hisp_female
group_name = 'idu_hisp_female'
colors = sns.color_palette('Blues_d', 1)
total = pd.DataFrame()
for replication in range(replications):
    rep_df = feather.read_dataframe(f'{os.getcwd()}/../../out/age_by_h1yy/{group_name}_{replication}.feather')
    total = total.append(rep_df)
means = total.groupby(['h1yy', 'term']).mean()

fig, ax = plt.subplots(figsize=(8.0, 6.0))
colors = sns.color_palette('Blues_d', 5)
df = pd.DataFrame({'Age': ages, 'Proportion': mixture(means.loc[2010], ages), 'Year': len(ages)*['2010 - 2030']})

print(df)
g = sns.lineplot(ax=ax, x='Age', y='Proportion', hue='Year', data=df.reset_index(), legend='full')
g.set(ylim=(-0.002, 0.122))
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[1:], labels=labels[1:], frameon=False)
plt.title(group_name)
plt.savefig( f'{os.getcwd()}/../../out/fig/age_by_h1yy/individual/{group_name}.png', dpi=dpi, bbox_inches='tight')
