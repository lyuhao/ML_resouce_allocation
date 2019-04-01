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

"""
Linear Regression With SGD Example.
"""
from __future__ import print_function

from pyspark import SparkContext
from pyspark.ml.regression import AFTSurvivalRegression
# $example on$
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
# $example off$
from pyspark.mllib.util import MLUtils
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
if __name__ == "__main__":
    #spark= SparkSession.builder.getOrCreate()
    sc = SparkContext(appName="PythonLinearRegressionWithSGDExample")
    spark = SparkSession(sc)
    # $example on$
    # Load and parse the data
    def parsePoint(line):
        values = [float(x) for x in line.replace(',', ' ').split(' ')]
        return LabeledPoint(values[0], values[1:])

    #data = sc.textFile("/home/yl408/yuhao_datasets/kdda_part")
    #parsedData = data.map(parsePoint)
    dataset = spark.read.format("libsvm").load("file:///home/yl408/yuhao_datasets/kdda_part")
    # Build the model
    
    dataset = dataset.withColumn("censor",lit(1))
    dataset = dataset.withColumn("label",lit(1))

#    values = dataset.select('label').collect()
 #   print(values)
    #result = parsedData.collect()
    #print(result[0])
    #print(parsedData.collect())
    #training = spark.createDataFrame(dataset)
    #training = parsedData.toDF()
    quantileProbabilities = [0.3, 0.6]
    aft = AFTSurvivalRegression(quantileProbabilities=quantileProbabilities,
                            quantilesCol="quantiles")
    model = aft.fit(dataset)


    #print("Coefficients: " + str(model.coefficients))
    #print("Intercept: " + str(model.intercept))
    #print("Scale: " + str(model.scale))

    # Evaluate the model on training data
    
    # valuesAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
    # MSE = valuesAndPreds \
    #     .map(lambda (v, p): (v - p)**2) \
    #     .reduce(lambda x, y: x + y) / valuesAndPreds.count()
    # print("Mean Squared Error = " + str(MSE))

    # Save and load model
   # model.save(sc, "target/tmp/pythonLinearRegressionWithSGDModel")
   # sameModel = LinearRegressionModel.load(sc, "target/tmp/pythonLinearRegressionWithSGDModel")
    # $example off$
