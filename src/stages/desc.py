import os
import math

import pandas as pd
import numpy as np
import pydmr

root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
resultspath = os.path.join(root, 'build', 'Tables')



def summarise_visits():

    # Get all data as dataframe
    file = os.path.join(root, 'build', 'Data', 'all_data.dmr')
    vals = pydmr.read(file, 'pandas')['pars']
  
    # Pivot all data
    kwargs = {
        'columns': 'subject',
        'index': ['study', 'parameter'],
        'values': 'value',
    }  
    data_all_visits = vals.pivot(**kwargs)
    
    # Pivot data per visit
    kwargs['index'] = ['parameter']
    data_screening = vals[vals.study=='screening'].pivot(**kwargs)
    data_control = vals[vals.study=='control'].pivot(**kwargs)
    data_drug = vals[vals.study=='drug'].pivot(**kwargs)

    # Export to csv
    file_all_visits = os.path.join(resultspath, 'vals.csv')
    file_screening = os.path.join(resultspath, 'vals_screening.csv')
    file_control = os.path.join(resultspath, 'vals_control.csv')
    file_drug = os.path.join(resultspath, 'vals_drug.csv')

    data_all_visits.to_csv(file_all_visits, na_rep="NaN")
    data_screening.to_csv(file_screening, na_rep="NaN")
    data_control.to_csv(file_control, na_rep="NaN")
    data_drug.to_csv(file_drug, na_rep="NaN")


    
    # # Pivot all sdevs
    # kwargs = {
    #     'columns': 'subject', 
    #     'index': ['visit', 'parameter'],
    #     'values': 'value',
    # }  
    # data_pivot = sdev.pivot(**kwargs)
    # file = os.path.join(resultspath, 'stdev.csv')
    # data_pivot.to_csv(file, na_rep="NaN")

    # # Pivot stdev per visit
    # kwargs = {
    #     'columns': 'subject', 
    #     'index': ['parameter'],
    #     'values': 'value',
    # }
    # data_control = sdev[sdev.visit=='control'].pivot(**kwargs)
    # data_drug = sdev[sdev.visit=='drug'].pivot(**kwargs)

    # file_control = os.path.join(resultspath, 'stdev_control.csv')
    # file_drug = os.path.join(resultspath, 'stdev_drug.csv')

    # data_control.to_csv(file_control, na_rep="NaN")
    # data_drug.to_csv(file_drug, na_rep="NaN")


def effect_size():

    # Get all data as dataframe  
    file = os.path.join(root, 'build', 'Data', 'all_data_effect.dmr')

    dmr = pydmr.read(file, 'pandas', study='control')
    df0 = dmr['pars'].pivot(columns='parameter', index='subject', values='value')
    dmr = pydmr.read(file, 'pandas', study='drug')
    df1 = dmr['pars'].pivot(columns='parameter', index='subject', values='value')

    for abs in [False, True]:
        if abs:
            effect = df1-df0
        else:
            effect = 100*(df1-df0)/df0

        # drop subjects with only 1 visit
        effect = effect.dropna(axis=0, how='all')
        # drop biomarkers with only 1 value for some subjects
        #effect = effect.dropna(axis=1, how='any')

        # Export effect sizes
        suffix = 'absolute' if abs else 'relative'
        file = os.path.join(resultspath, f'effect_size_{suffix}.csv')
        effect.to_csv(file, na_rep="NaN")


def demographics():

    file = os.path.join(root, 'build', 'Data', 'all_data.dmr')
    demographics = [
        'Age', 'BMI', 'Height', 'Weight', 
        'Crea', 'Urea', 
        'AvrALP', 'AvrALT', 'AvrAlb', 'AvrBili', 'AvrConBili', 'AvrConTotBili', 
    ]
    dmr = pydmr.read(file, 'pandas', parameter=demographics, study=['control', 'screening'])
    df = dmr['pars'].pivot(columns='subject', index='parameter', values='value')
    df = df.T.describe().T.round(1)

    name = lambda x: f"{pydmr.metadata(file, x, 'description').replace('Average ', '')} ({pydmr.metadata(file, x, 'unit')})"
    df = df.rename(name, axis=0)

    file = os.path.join(resultspath, 'table_demographics.csv')
    df.to_csv(file)


def averages():
    
    # Concatenate all data
    file = os.path.join(root, 'build', 'Data', 'all_data_effect.dmr')
    vals = pydmr.read(file, 'pandas', study=['control', 'drug'])['pars']

    # Get mean, sdev and count of all variables
    avr = pd.pivot_table(vals, values='value', columns='study', 
                         index='parameter', aggfunc='mean')
    std = pd.pivot_table(vals, values='value', columns='study', 
                         index='parameter', aggfunc='std')
    cnt = pd.pivot_table(vals, values='value', columns='study', 
                         index='parameter', aggfunc='count')
    
    effect_file = os.path.join(resultspath, 'effect_size_relative.csv')
    effect = pd.read_csv(effect_file, index_col=0)
    
    # Get mean, sdev and count of all effect sizes
    avr_effect = effect.mean()
    std_effect = effect.std()
    cnt_effect = effect.count()
    
    # Calculate 95% CI intervals
    visits = ['control', 'drug']
    b_avr = around_sig(avr[visits[0]].values, 3)
    r_avr = around_sig(avr[visits[1]].values, 3)
    c_avr = around_sig(avr_effect.values, 3)
    b_err = around_sig(
        1.96*std[visits[0]].values / np.sqrt(cnt[visits[0]].values), 2)
    r_err = around_sig(
        1.96*std[visits[1]].values / np.sqrt(cnt[visits[1]].values), 2)
    c_err = around_sig(
        1.96*std_effect.values / np.sqrt(cnt_effect.values), 2)
    
    # Create output array
    output = pd.DataFrame(index=avr.index)
    output[visits[0] + ' Mean'] = b_avr
    output[visits[0] + ' 95%CI'] = b_err
    output[visits[1] + ' Mean'] = r_avr
    output[visits[1] + ' 95%CI'] = r_err
    output['Effect' + ' Mean'] = c_avr
    output['Effect' + ' 95%CI'] = c_err

    # Save output array
    file = os.path.join(resultspath, 'averages.csv')
    output.to_csv(file, na_rep="NaN")


def t_statistic():

    # Concatenate all data
    file = os.path.join(root, 'build', 'Data', 'all_data_effect.dmr')
    vals = pydmr.read(file, 'pandas')['pars']

    # Get data from subjects that completed voth visits
    data_drug = vals[vals.study=='drug']
    data_control = vals[(vals.study=='control') & (~vals.subject.isin(['LDS-001','LDS-005']))]

    # Separate visits
    kwargs = {'columns': 'subject', 'index': 'parameter', 'values': 'value'}
    data_control = data_control.pivot(**kwargs)
    data_drug = data_drug.pivot(**kwargs)

    # Compute t_statistic
    diff = data_drug - data_control
    std = diff.std(axis=1)
    cnt = diff.notna().sum(axis=1)
    norm = std/np.sqrt(cnt)
    diff = diff.div(norm, axis=0) 

    # Save
    file = os.path.join(resultspath, 't_statistic.csv')
    diff.to_csv(file, na_rep="NaN")



# Helper functions


def _first_digit(x):
    if np.isnan(x):
        return x
    return -int(math.floor(math.log10(abs(x))))

def _round_sig(x, n):
    # Round to n significant digits
    if x==0:
        return x
    if np.isnan(x):
        return x
    return round(x, _first_digit(x) + (n-1))
    
def around_sig(x, n):
    return np.array([_round_sig(v,n) for v in x])


def main():

    summarise_visits()
    effect_size()
    t_statistic()
    demographics()
    averages()
    
    



if __name__=='__main__':
    main()
