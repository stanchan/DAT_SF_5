#!/usr/bin/env python
# ------------------------------------------------------------------------------
# About                                                                      {{{
# Filename:         hw2.py
# Created:          30 Mar 2014
# Last Modified:    30 Mar 2014
# Maintainer:       Stan Chan
# Note:             Homework Assignment #2
# Requires:
# Dependencies:     numpy 1.8.0
#                   pandas 0.13.1
#                   matplotlib 1.3.1
#                   sklearn
#
# $Id$                               }}}
# ------------------------------------------------------------------------------
# Import modules                                                             {{{

import sys
import os
import re
import argparse
import csv

print("Loading Numpy library..."),
try:
    import numpy as np
except Exception as e:
    raise ImportError, "Numpy library import failed!"
print("done!")

print("Loading Pandas library..."),
try:
    import pandas as pd
except Exception as e:
    raise ImportError, "Pandas library import failed!"
print("done!")

print("Loading Matplotlib library..."),
try:
    import matplotlib.pyplot as plt
except Exception as e:
    raise ImportError, "Matplotlib library import failed!"
print("done!")

print("Loading scikit-learn library..."),
try:
    from sklearn.cross_validation import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.cross_validation import KFold
except Exception as e:
    raise ImportError, "scikit-learn library import failed!"
print("done!")

#                                                                            }}}
# ------------------------------------------------------------------------------
# Configuration                                                              {{{

__version__ = "1.0.0"

# Enable automatic display of errors in pdb
use_pdb = "no"

dataset_filename = os.path.join(os.getcwd(),"iris_data.csv")
work_filename = os.path.join(os.getcwd(),"work_data.csv")
logconfig_filename = os.path.join(os.getcwd(),"logging.conf")
plot_filename = os.path.join(os.getcwd(),"k_vs_accuracy.png")

# Data type for data or columns.
dtype_values = { 0 : np.float, 1 : np.float, 2 : np.float, 3 : np.float, 4 : np.str }

# List of column names to use. If file contains no header row, then you should explicitly pass header=None
field_names = [ "sepal_length", "sepal_width", "petal_length", "petal_width", "class" ]

# Field descriptions
field_descriptions = [ "sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)" ]

# Subset of fields used as dataset
data_fields = [ "sepal_length", "sepal_width", "petal_length", "petal_width" ]

# Field to generate an index from
index_field_name = "class"

#                                                                            }}}
# ------------------------------------------------------------------------------
# Catch exceptions and send them to pdb                                      {{{

if use_pdb == "yes" :
    import IPython.core.ultratb as utb
    sys.excepthook = utb.FormattedTB(mode='Verbose',
                                     color_scheme='Linux', call_pdb=1)

#                                                                            }}}
# ------------------------------------------------------------------------------
# Blacklist older versions of python                                         {{{

ver = sys.version.split(' ')[0].split(".")
major = ver[:1]
minor = ver[1:2]
version = "{0}.{1}".format(major[0],minor[0])
if version in ('2.6','2.5','2.4','2.3','2.2','2.1','2.0'):
    print("Incompatible python version detected: {}".format(version))
    sys.exit(1)
else:
    pyver = version

class HomeworkException(Exception):
    def __init__(self, description):
        self.description = description
    def __str__(self):
        return repr(self.description)

class hw2(object):
    def __init__(self, options):
        """Constructor"""
        self.options = options
        try:
            self.dataset = pd.read_csv(self.options.work_file, header=None, dtype=self.options.dtype_values, names=self.options.field_names)
        except Exception as e:
            raise HomeworkException("Error loading data file: {}".format(self.options.csv_file))

        self.append_targets()

    def head(self):
        """Prints head of pandas object"""
        self._result = ("{}".format(self.dataset.head()))
        return self._result

    def tail(self):
        """Prints tail of pandas object"""
        self._result = ("{}".format(self.dataset.tail()))
        return self._result

    def append_targets(self):
        """Create index of the classes dataset"""
        self._class_names = np.unique(self.dataset[self.options.index_field_name])
        self._map_kv = dict()
        self._map_kv = { value : idx for idx, value in enumerate(self._class_names) }
        self.dataset['target'] = self.dataset[self.options.index_field_name].map(lambda idx: self._map_kv[idx])

    def get_data(self):
        """Get the relevant data fields"""
        self._temp_array = np.array([])
        self._temp_array = self.dataset[self.options.data_fields]
        return self._temp_array

    def knn(self, test_size=0.2, random_state=0):
        """Implement KNN classification"""
        self.X = self.get_data()
        self.names = self.options.field_descriptions
        self.y = self.dataset['target']
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split( self.X, self.y, test_size=test_size, random_state=random_state)
        self.knn_classifier = KNeighborsClassifier(3).fit(self._X_train, self._y_train.ravel())
        self.knn_score = self.knn_classifier.score(self._X_test, self._y_test)
        return self.knn_score

    # generic cross validation function
    def cross_validate(self, X=pd.DataFrame, y=pd.DataFrame, classifier=None, k_fold=5, indices=True, shuffle=True, random_state=0):
        """Compute score using KFold"""
        if X.empty:
            X = self.X
        if y.empty:
            y = self.y
        if classifier == None:
            classifier = KNeighborsClassifier(3).fit

        X = X.values
        y = y.values

        # derive a set of (random) training and testing indices
        self._k_fold_indices = KFold(len(X), n_folds=k_fold,
                                indices=indices, shuffle=shuffle,
                                random_state=random_state)
        self._k_score_total = 0
        # for each training and testing slices run the classifier, and score the results
        for train_slice, test_slice in self._k_fold_indices:
            self._model = classifier(X[[ train_slice  ]], y[[ train_slice  ]])
            self._k_score = self._model.score(X[[ test_slice ]], y[[ test_slice ]])
            self._k_score_total += self._k_score

        # return the average accuracy
        return self._k_score_total/k_fold

    def get_optimal(self, X=pd.DataFrame, y=pd.DataFrame, classifier=None, k_fold=5, indices=True, shuffle=True, random_state=0):
        """Get the optimal score"""
        if X.empty:
            X = self.X
        if y.empty:
            y = self.y
        if classifier == None:
            classifier = KNeighborsClassifier

        self.optimal_scores = { idx+1: self.cross_validate(X, y, classifier(idx+1).fit, k_fold, indices=indices, shuffle=shuffle, random_state=random_state) for idx in xrange(len(self.dataset))}

        max_scores = max(self.optimal_scores, key=self.optimal_scores.get)
        max_value = max(self.optimal_scores.values())

        return { max_scores: max_value }

    def plot_optimal_graph(self):
        plt.title("K vs Accuracy", fontsize=16)
        plt.plot(self.optimal_scores.keys(), self.optimal_scores.values(), marker="o", c="b")
        plt.savefig(str(self.options.plot_filename), format="png")
        plt.show()

def clean_csv(file, workfile, fields):
    with open(file, "rb") as infile, open(workfile, "wb") as outfile:
        reader = csv.reader(infile, delimiter=',')
        writer = csv.writer(outfile)
        for line in reader:
            if line:
                newline = list()
                for idx in xrange(fields):
                    newline.append(line[idx])
                writer.writerow(newline)

def main(argv):
    """Main function for this program"""
    progpathname = re.search(r'.*?[\./]?([A-Za-z0-9_]+\.py)$', argv[0])

    parser = argparse.ArgumentParser(description="Homework Assignment #2")
    parser.add_argument("--logconfig", action="append", dest="log_config",
                                       metavar="<FILE>", default=logconfig_filename,
                                       help="Specify a logging config file (Defaults to \"{}\")".format(logconfig_filename))
    parser.add_argument("-f", "--file", action="store", dest="csv_file",
                                        metavar="<FILE>", default=dataset_filename,
                                        help="Specify the csv file to process (Defaults to \"{}\")".format(dataset_filename))
    parser.add_argument("--workfile", action="store", dest="work_file",
                                      metavar="<FILE>", default=work_filename,
                                      help="Specify the working csv file to for post processed data (Defaults to \"{}\")".format(work_filename))
    parser.add_argument("--datatypes", action="store", dest="dtype_values",
                                       type=dict, default=dtype_values,
                                       help="Specify the data types for each field of the csv file (Defaults to \"{}\")".format(dtype_values))
    parser.add_argument("--fields", action="store", dest="field_names",
                                    type=list, default=field_names,
                                    help="Specify the field names for each field of the csv file (Defaults to \"{}\")".format(field_names))
    parser.add_argument("--fielddesc", action="store", dest="field_descriptions",
                                       type=list, default=field_descriptions,
                                       help="Specify the field descriptions for each field of the csv file (Defaults to \"{}\")".format(field_descriptions))
    parser.add_argument("--datafields", action="store", dest="data_fields",
                                        type=list, default=data_fields,
                                        help="Specify the field names for each field of the csv file (Defaults to \"{}\")".format(data_fields))
    parser.add_argument("--indexfield", action="store", dest="index_field_name", default=index_field_name,
                                        help="Specify the class field sub-names for the csv file (Defaults to \"{}\")".format(index_field_name))
    parser.add_argument("-p", "--plotfile", action="store", dest="plot_filename",
                                            metavar="<FILE>", default=plot_filename,
                                            help="Specify the plot output file (Defaults to \"{}\")".format(plot_filename))
    args = parser.parse_args()

    try:
        clean_csv(args.csv_file, args.work_file, len(field_names))
    except Exception as e:
        print("Error processing file: {}".format(args.csv_file))
        print("{}".format(e))
        sys.exit(255)

    homework = hw2(args)

    try:
        print("Head of dataset:")
        print(homework.head())
        print("Tail of dataset:")
        print(homework.tail())
        print("KNN Classification Result: {}".format(homework.knn()))
        print("Samples:")
        print(homework.X.take([1,2,4,8,16,32]))
        print("Cross Validation Result: {}".format(homework.cross_validate()))
        print("Optimal value of K (aka Hyperparameter): {}".format(homework.get_optimal()))
        print(homework.plot_optimal_graph())

    except HomeworkException as e:
        print("Error in program:")
        print(e)

if __name__ == "__main__":
    main(sys.argv[0:])