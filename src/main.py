"""
Main script to reproduce the results in the paper.
"""

from steps import data, tables, calc, plot, report

data.main()     # Format data
tables.main()   # Create tables
calc.main()     # Statistical analysis
plot.main()     # Create plots
report.main()   # Create report