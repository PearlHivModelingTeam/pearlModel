'''
Module for aggregating results from pearl simulation
'''

def bmi_info(population_dataframe, out_dir):
    bmi_info_df = population_dataframe.groupby(['group', 'replication', 't_dm']).size().reset_index().rename(columns={0: 'n'})
    bmi_info_df.to_parquet(out_dir/"bmi_info.parquet")
