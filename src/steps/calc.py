import os
import math

import numpy as np
import pandas as pd
import pingouin as pg
import pydmr

from steps import data

root = os.getcwd()
# Create directories for tables
resultspath = os.path.join(root, 'build', 'Tables')


# Exclude variables that are analytically related to others from correlation
# analysis
EXCLUDE_CORREL = [
#    'RE_R1l',
    'Th_i', 'Th_f', 'Th', 
    'khe_i', 'khe_f', 'kbh_i', 'kbh_f',
    'Kbh', 'Khe', 
    'khe_slope', 'kbh_slope',
    #'AvrALP', 'AvrALT', 'AvrAlb', 'AvrBili', 'AvrConBili', 'AvrConTotBili',
    'PreALP', 'PreALT', 'PreAlb', 'PreBili', 'PreConBili', 'PreConTotBili',
    'PostALP', 'PostALT', 'PostAlb', 'PostBili', 'PostConBili', 'PostConTotBili',
]


def create_pivot():

    # Get all data as dataframe
    vals, sdev = data.read()
  
    # Pivot all data
    kwargs = {
        'columns': 'subject',
        'index': ['visit', 'parameter'],
        'values': 'value',
    }  
    data_pivot = vals.pivot(**kwargs)
    file = os.path.join(resultspath, 'vals.csv')
    data_pivot.to_csv(file, na_rep="NaN")

    # Pivot data per visit
    kwargs = {
        'columns': 'subject', 
        'index': ['parameter'],
        'values': 'value',
    }
    data_screening = vals[vals.visit=='screening'].pivot(**kwargs)
    data_control = vals[vals.visit=='control'].pivot(**kwargs)
    data_drug = vals[vals.visit=='drug'].pivot(**kwargs)

    file_screening = os.path.join(resultspath, 'vals_screening.csv')
    file_control = os.path.join(resultspath, 'vals_control.csv')
    file_drug = os.path.join(resultspath, 'vals_drug.csv')

    data_screening.to_csv(file_screening, na_rep="NaN")
    data_control.to_csv(file_control, na_rep="NaN")
    data_drug.to_csv(file_drug, na_rep="NaN")

    
    # Pivot all sdevs
    kwargs = {
        'columns': 'subject', 
        'index': ['visit', 'parameter'],
        'values': 'value',
    }  
    data_pivot = sdev.pivot(**kwargs)
    file = os.path.join(resultspath, 'stdev.csv')
    data_pivot.to_csv(file, na_rep="NaN")

    # Pivot stdev per visit
    kwargs = {
        'columns': 'subject', 
        'index': ['parameter'],
        'values': 'value',
    }
    data_control = sdev[sdev.visit=='control'].pivot(**kwargs)
    data_drug = sdev[sdev.visit=='drug'].pivot(**kwargs)

    file_control = os.path.join(resultspath, 'stdev_control.csv')
    file_drug = os.path.join(resultspath, 'stdev_drug.csv')

    data_control.to_csv(file_control, na_rep="NaN")
    data_drug.to_csv(file_drug, na_rep="NaN")


def derive_effect_size(abs=False):

    # Get all data as dataframe  
    file = os.path.join(root, 'build', 'Data', 'all_data_effect.dmr')

    dmr = pydmr.read(file, 'pandas', study='control')
    df0 = dmr['pars'].pivot(columns='parameter', index='subject', values='value')
    dmr = pydmr.read(file, 'pandas', study='drug')
    df1 = dmr['pars'].pivot(columns='parameter', index='subject', values='value')

    if abs:
        effect = df1-df0
    else:
        effect = 100*(df1-df0)/df0

    # drop subjects with only 1 visit
    effect = effect.dropna(axis=0, how='all')
    # drop biomarkers with only 1 value for some subjects
    #effect = effect.dropna(axis=1, how='any')

    # Export effect sizes
    prefix = 'abs_' if abs else 'rel_'
    file = os.path.join(resultspath, prefix + 'effect_size_wide.csv')
    effect.to_csv(file, na_rep="NaN")


def derive_t_statistic():

    # Concatenate all data
    vals, _ = data.read(effect=True)

    data_drug = vals[vals.visit=='drug']
    data_control = vals[(vals.visit=='control') & (~vals.subject.isin(['1','5']))]

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


def descriptives():
    
    # Concatenate all data
    vals, _ = data.read(effect=True)

    # Exclude parameters with uninteresting effect sizes
    vals = vals[vals.visit != 'screening']
    vals = vals[~vals.parameter.isin(data.NO_EFFECT)]

    # Get mean, sdev and count of all variables
    avr = pd.pivot_table(vals, values='value', columns='visit', 
                         index='parameter', aggfunc='mean')
    std = pd.pivot_table(vals, values='value', columns='visit', 
                         index='parameter', aggfunc='std')
    cnt = pd.pivot_table(vals, values='value', columns='visit', 
                         index='parameter', aggfunc='count')
    
    effect_file = os.path.join(resultspath, 'rel_effect_size_wide.csv')
    effect = pd.read_csv(effect_file, dtype=data.TYPES).set_index('subject')
    
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
    output.to_csv(os.path.join(resultspath, 'descriptives.csv'), na_rep="NaN")


def ttest():
    
    # Concatenate all data
    vals, _ = data.read(effect=True)

    # Perform t-tests and save if df output
    output = None
    for par in np.sort(vals.parameter.unique()):
        data_par = vals[vals.parameter==par]
        stats = pg.pairwise_tests(
                data=data_par, 
                dv='value', within='visit', subject='subject', 
                return_desc=False, effsize='odds-ratio',
        )
        stats['parameter'] = par
        if output is None:
            output = stats
        else:
            output = pd.concat([output, stats])

    # Save results
    output.to_csv(os.path.join(resultspath, 'ttest.csv'), index=False, na_rep="NaN")


def univariate():
    
    # Concatenate all data
    desc = pd.read_csv(os.path.join(resultspath, 'descriptives.csv')) 
    stats = pd.read_csv(os.path.join(resultspath, 'ttest.csv')) 

    desc = desc.sort_values(by='parameter')
    stats = stats.sort_values(by='parameter')
    cols = ['Contrast', 'A', 'B', 'Paired', 'Parametric', 'dof', 'alternative']
    stats = stats.drop(columns = cols + ['parameter'])

    # Update output array
    output = pd.concat([desc, stats], axis=1)
    output.to_csv(os.path.join(resultspath, 'univariate.csv'), index=False, na_rep="NaN")


def tables_univariate():

    file = os.path.join(resultspath, 'univariate.csv')
    output = pd.read_csv(file)

    # Exclude parameters with irrelevant/no effect sizes
    output = output[~output.parameter.isin(data.EXCLUDE_EFFECT+EXCLUDE_CORREL)]

    output.rename(columns={"parameter": "Biomarker", 
                           "p-unc": "p-value", 'BF10': 'Bayes Factor', 
                           'odds-ratio': 'Odds Ratio'}, inplace=True)
    
    b_avr = output['control Mean'].values
    b_err = output['control 95%CI'].values
    r_avr = output['drug Mean'].values
    r_err = output['drug 95%CI'].values
    c_avr = output['Effect Mean'].values
    c_err = output['Effect 95%CI'].values

    output['control'] = [
        f'{b_avr[i]} ({b_err[i]}) ' for i in range(b_avr.size)]
    output['drug'] = [
        f'{r_avr[i]} ({r_err[i]}) ' for i in range(r_avr.size)]
    output['Effect size (%)'] = [
        f'{c_avr[i]} ({c_err[i]}) ' for i in range(c_avr.size)]
    
    # lookup group, units and add as column
    
    unit = data.lookup_vals(output.Biomarker.values, 'unit')
    name = data.lookup_vals(output.Biomarker.values, 'name')
    output['Name'] = [f"{name[i].replace('Average ', '')} ({unit[i]})" for i in range(len(unit))]
    output['Group'] = data.lookup_vals(output.Biomarker.values, 'group')

    # Sort and format values
    output = output.sort_values(by=['Group', 'p-value'])
    output.loc[:,'p-value'] = np.around(output['p-value'].values, 5)
    output = output.astype({'Bayes Factor': 'float32'})
    output.loc[:,'Bayes Factor'] = np.around(output['Bayes Factor'].values, 2)
    output.loc[:,'Odds Ratio'] = np.around(output['Odds Ratio'].values, 2)
    output.loc[:,'T'] = np.around(output['T'].values, 1)
    output.loc[:,'p-value'] = np.around(output['p-value'].values, 3)
    
    # Save tables
    output = output[
        ['Name', 'Group', 'control', 'drug', 
         'Effect size (%)', 'T', "p-value"]
    ]
    output_liver = output[output.Group != 'MRI - aorta']
    output_aorta = output[output.Group == 'MRI - aorta']
    
    output_liver.to_csv(os.path.join(resultspath, 'table_liver_univariate.csv'), index=False)
    output_aorta.to_csv(os.path.join(resultspath, 'table_aorta_univariate.csv'), index=False)


def correlations():

    file = os.path.join(resultspath, 'abs_effect_size_wide.csv')
    effect = pd.read_csv(file).set_index('subject')
    #corr = pg.pairwise_corr(effect)
    corr = []
    vars = set(effect.columns) - set(EXCLUDE_CORREL)
    for X in vars:
        for Y in vars:
            x = effect[X].values
            y = effect[Y].values
            c = pg.corr(x,y)
            c['X'] = X
            c['Y'] = Y
            corr.append(c)
    corr = pd.concat(corr)
    corr.rename(columns={'p-val':'p-unc'}, inplace=True)
    file = os.path.join(resultspath, 'corr_abs_effect_size.csv')
    corr.to_csv(file, na_rep="NaN", index=False)


def correlations_control():
    
    file = os.path.join(resultspath, 'vals_control.csv')
    vals_control = pd.read_csv(file).set_index('parameter')
    file = os.path.join(resultspath, 'vals_screening.csv')
    vals_screening = pd.read_csv(file).set_index('parameter')
    vals = pd.concat([vals_control, vals_screening]).T

    corr = []
    vars = set(vals.columns) - set(EXCLUDE_CORREL) - set(data.EXCLUDE_EFFECT) - set(data.NO_EFFECT)
    for X in vars:
        for Y in vars:
            x = vals[X].values
            y = vals[Y].values
            c = pg.corr(x,y)
            c['X'] = X
            c['Y'] = Y
            corr.append(c)
    corr = pd.concat(corr)
    corr.rename(columns={'p-val':'p-unc'}, inplace=True)
    file = os.path.join(resultspath, 'corr_control.csv')
    corr.to_csv(file, na_rep="NaN", index=False)


def correlations_submatrix():

    Y = 'MRI - liver'
    correlations_single_submatrix('MRI - liver', Y, 'corr_liver')
    correlations_single_submatrix('MRI - aorta', Y, 'corr_liver_aorta')
    correlations_single_submatrix('Blood - liver function test', Y, 'corr_liver_blood')


def correlations_single_submatrix(X, Y, filename):

    file = os.path.join(resultspath, 'corr_abs_effect_size.csv')
    corr = pd.read_csv(file)

    # select parameters
    xpars = data.lookup_params('group', X)
    ypars = data.lookup_params('group', Y)
    corr = corr[(corr.X.isin(xpars)) & (corr.Y.isin(ypars))]

    # Pivot to wide format
    vals = corr.pivot(values='r', index='X', columns='Y')
    file = os.path.join(resultspath, filename + '_vals.csv')
    vals.to_csv(file, na_rep="NaN")

    pval = corr.pivot(values='p-unc', index='X', columns='Y')
    file = os.path.join(resultspath, filename + '_pval.csv')
    pval.to_csv(file, na_rep="NaN")


def correlations_control_submatrix():

    Y = 'MRI - liver'
    correlations_control_single_submatrix('MRI - liver', Y, 'corr_liver_control')
    correlations_control_single_submatrix('MRI - aorta', Y, 'corr_liver_aorta_control')
    correlations_control_single_submatrix('Blood - liver function test', Y, 'corr_liver_blood_control')
    correlations_control_single_submatrix('Screening', Y, 'corr_liver_screening')


def correlations_control_single_submatrix(X, Y, filename):

    file = os.path.join(resultspath, 'corr_control.csv')
    corr = pd.read_csv(file)

    # select parameters
    xpars = data.lookup_params('group', X)
    ypars = data.lookup_params('group', Y)
    corr = corr[(corr.X.isin(xpars)) & (corr.Y.isin(ypars))]

    # Pivot to wide format
    vals = corr.pivot(values='r', index='X', columns='Y')
    file = os.path.join(resultspath, filename + '_vals.csv')
    vals.to_csv(file, na_rep="NaN")

    pval = corr.pivot(values='p-unc', index='X', columns='Y')
    file = os.path.join(resultspath, filename + '_pval.csv')
    pval.to_csv(file, na_rep="NaN")


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

    if not os.path.exists(resultspath):
        os.makedirs(resultspath)
    
    create_pivot()
    derive_effect_size(abs=False)
    derive_effect_size(abs=True)
    derive_t_statistic()
    descriptives()
    ttest()
    univariate()
    correlations()
    correlations_submatrix()
    correlations_control()
    correlations_control_submatrix()

    tables_univariate()


if __name__=='__main__':
    main()



