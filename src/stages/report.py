import os
import miblab

root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def main():

    path = os.path.join(root, 'build')
    figpath = os.path.join(path, 'Figs')
    tablepath = os.path.join(path, 'Tables')

    print('Creating report..')

    # Cover and title pages
    doc = miblab.Report(
        path,
        'Report',
        title = 'Predicting liver-mediated drug-drug interactions with MRI: A first-in-human study',
        subtitle = 'Results',
        subject = 'Internal report',
    )

    doc.chapter('Figures')

    doc.figure( 
        os.path.join(figpath, 'primary_outcomes.png'),
        clearpage=True,
        caption = (
            "Visualisation of the primary endpoints khe and kbh across "
            "the population, showing a significant response to "
            "drug: (a) the relative and absolute effect size "
            "across the population as box plots; and (b) the individual "
            "values at control (left of plots) and after single dose of "
            "drug (right of plot). Colored lines in (b) represent "
            "individual volunteers. Note: absolute effect sizes for kbh "
            "have been scaled with a factor 10 to improve visualisation." 
        ),
    )
    doc.figure(
        os.path.join(figpath, 'secondary_outcomes.png'),
        clearpage=True,
        caption = (
            "drug effect for all parameters that show a "
            "significant reduction in the mean value (p<0.01). The top "
            "row shows the difference relative to the standard error of "
            "the difference, and the bottom row shows the individual "
            "effects for the corresponding biomarkers."  
        ),
    )
    doc.figure(
        os.path.join(figpath, 'correlations_control.png'),
        clearpage=True,
        caption = (
            "Correlations between parameters at control."  
        ),
    )
    doc.figure(
        os.path.join(figpath, 'correlations_effect.png'),
        clearpage=True,
        caption = (
            "Correlations between parameter changes." 
        ),
    )

    doc.chapter('Tables')

    doc.table(
        os.path.join(tablepath, 'table_demographics.csv'),
        caption = 'Demographics of the study population.'
    )
    doc.table(
        os.path.join(tablepath, 'table_lft_between_visits.csv'),
        caption = 'LFT changes between both visits.',
        clearpage=True,
    )
    doc.table(
        os.path.join(tablepath, 'table_liver_univariate.csv'),
        caption = 'Univariate data analysis (liver).',
        cwidth=1.5, clearpage=True,
    )
    doc.table(
        os.path.join(tablepath, 'table_aorta_univariate.csv'),
        caption = 'Univariate data analysis (aorta).',
        cwidth=1.5, clearpage=True,
    )

    doc.chapter('Supplements')

    doc.figure(
        os.path.join(figpath, 'clustering.png'),
        caption = (
            "Clustering of parameters and subjects." 
        ),
        clearpage=True,
    )

    doc.build()


if __name__ == '__main__':
    main()
