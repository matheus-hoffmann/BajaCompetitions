"""
python main.py --datapath "data/"
               --filename "dataset.csv"
               --safety 0
               --project 180
               --dynamics 100
               --endurance 200

Author: Matheus Hoffmann
"""

import sys
import argparse
from utils import plot, data_reader
from predictive_model import set_input, set_model


def main(argv):
    parser = argparse.ArgumentParser(description="Get some insights from available data.")
    parser.add_argument("--datapath",
                        dest="datapath",
                        default="data/",
                        help="Path to dataset.",
                        metavar="String",
                        type=str)
    parser.add_argument("--filename",
                        dest="filename",
                        default="dataset.csv",
                        help="Dataset file with extension.",
                        metavar="String",
                        type=str)
    parser.add_argument("--safety",
                        dest="safety",
                        default=None,
                        help="Safety points",
                        type=float)
    parser.add_argument("--project",
                        dest="project",
                        default=None,
                        help="Project points (presentation + technical report).",
                        type=float)
    parser.add_argument("--dynamics",
                        dest="dynamics",
                        default=None,
                        help="Dynamics points (S&T + Slalom + ...).",
                        type=float)
    parser.add_argument("--endurance",
                        dest="endurance",
                        default=None,
                        help="Endurance points.",
                        type=float)
    args = parser.parse_args()

    df = data_reader(path=args.datapath,
                     filename=args.filename,
                     rmv_missing_rows=True)
    df = df.drop(df[df['year'] < 2014].index).reset_index(drop=True)  # Safety points are wrong

    xtest = set_input(df=df,
                      safety=args.safety,
                      project=args.project,
                      dynamics=args.dynamics,
                      endurance=args.endurance)

    x = df[['safety', 'project', 'dynamics', 'endurance']].values
    y = df['position'].values
    model = set_model(x, y,
                      train_percentage=0.8,
                      n_splits=10,
                      n_folds=5,
                      n_params=10,
                      n_repeats=1)

    print("Predicted position is {}".format(int(model.predict(xtest))))

    xname = 'endurance'
    yname = 'position'
    plot(x=df[xname].values,
         y=df[yname].values,
         xlabel=xname,
         ylable=yname,
         title=yname + " x " + xname,
         marker=True,
         grid=True)


if __name__ == "__main__":
    main(sys.argv[1:])
