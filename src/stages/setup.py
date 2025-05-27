import os


def build():

    # Build folder for all generated results
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    build = os.path.join(root, 'build')

    # Subfolders for data, tables and figures
    data_dir = os.path.join(build, 'Data')
    tables_dir = os.path.join(build, 'Tables')
    figures_dir = os.path.join(build, 'Figs')

    # Creat clean directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)  
    os.makedirs(figures_dir, exist_ok=True)

    return data_dir, tables_dir, figures_dir


def main():
    """
    Main script to set up the project structure.
    """
    build()


if __name__ == "__main__":
    main()