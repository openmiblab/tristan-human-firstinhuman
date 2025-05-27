"""
Main script to reproduce the results in the paper.
"""

from stages import setup, data, desc, stats, plot, report

setup.main()    # Build folders
data.main()     # Format data
desc.main()     # Create tables
stats.main()    # Statistical analysis
plot.main()     # Create plots
report.main()   # Create report