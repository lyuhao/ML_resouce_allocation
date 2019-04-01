from __future__ import print_function

# $example on$
from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils


if __name__ == "__main__":
    sc = SparkContext(appName="PythonrandomForestClaexample")
    data = MLUtils.loadLibSVMFile(sc,"file:///home/yl408/yuhao_datasets/phishing")

    model = RandomForest.trainClassifier(data, numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=10, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=4, maxBins=32)
  #d  model.save(sc, "file://home/yl408/spark-ml/myrandomForestclaModel")