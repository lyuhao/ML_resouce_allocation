#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function

# $example on$
from pyspark import SparkContext
from pyspark.ml.clustering import BisectingKMeans
# $example off$
from pyspark.sql import SparkSession
from numpy import array
from math import sqrt

from pyspark.mllib.clustering import KMeans, KMeansModel

"""
An example demonstrating bisecting k-means clustering.
Run with:
  bin/spark-submit examples/src/main/python/ml/bisecting_k_means_example.py
"""

if __name__ == "__main__":
    sc = SparkContext(appName="Pythonkmeansexample")

    # $example on$
    # Loads data.
    data = sc.textFile("file:///home/yl408/yuhao_datasets/kmean_data")
    parsedData = data.map(lambda line: array([float(x) for x in line.split(' ')]))

    # Trains a bisecting k-means model.
    clusters = KMeans.train(parsedData, 2, maxIterations=10, initializationMode="random")

    #spark.stop()
