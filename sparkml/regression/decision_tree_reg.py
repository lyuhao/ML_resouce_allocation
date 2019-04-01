from __future__ import print_function

# $example on$
from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils



if __name__ == "__main__":
    sc = SparkContext(appName="PythonDecisionTreeRegxample")
    data = MLUtils.loadLibSVMFile(sc,"file:///home/yl408/yuhao_datasets/phishing_new")

    model = DecisionTree.trainRegressor(data, categoricalFeaturesInfo={},
                                     impurity='variance', maxDepth=5, maxBins=32)
    #model.save(sc, "file:///home/yl408/spark-ml/myDecisionTreeRegModel")
    