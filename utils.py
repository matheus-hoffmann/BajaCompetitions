import os
import pandas as pd
import matplotlib.pyplot as plt


def plot(x, y,
         xlabel: str = "",
         ylable: str = "",
         title: str = "",
         marker: bool = False,
         grid: bool = True) -> None:
    """
    Generic plot function.
    :param x: X data.
    :param y: Y data.
    :param xlabel: Y label text.
    :param ylable: Y label text.
    :param title: Chart title.
    :param marker: If True, data points will be marked with '*'.
    :param grid: If True, gridded chart.
    """
    plt.Figure()
    if marker:
        plt.plot(x, y, '*k')
    else:
        plt.plot(x, y, 'k')
    plt.grid(grid)
    plt.xlim([min(x), max(x)])
    plt.xlabel(xlabel)
    plt.ylabel(ylable)
    plt.title(title)
    plt.show()


def data_reader(path: str = "",
                filename: str = "",
                rmv_missing_rows: bool = False) -> pd.DataFrame:
    """
    Read specified dataset and return pandas dataframe.
    :param path: Path to dataset.
    :param filename: Dataset name.
    :param rmv_missing_rows: Remove rows with missing data.
    :return: Pandas dataframe with data.
    """
    fullpath = path
    if path[-1] != "/":
        fullpath += "/"
    fullpath += filename

    if ".csv" in fullpath:
        df = pd.read_csv(fullpath)
    elif ".xlsx" in fullpath:
        df = pd.read_excel(fullpath, engine='openpyxl')
    else:
        os.error("ExtensionError: File must be .csv or .xlsx")

    if rmv_missing_rows:
        df = df.dropna()
        df = df.reset_index(drop=True)

    return df

