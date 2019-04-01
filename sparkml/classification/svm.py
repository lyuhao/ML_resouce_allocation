
from __future__ import print_function

from pyspark import SparkContext
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.util import MLUtils


if __name__ == "__main__":
    sc = SparkContext(appName="PythonrandomForestClaexample")
    data = MLUtils.loadLibSVMFile(sc,"file:///home/yl408/yuhao_datasets/kdda_part")
    model = SVMWithSGD.train(data, iterations=100)
    #model.save(sc, "file://home/yl408/spark-ml/pythonSVMWithSGDModel")

