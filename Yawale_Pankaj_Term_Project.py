import os
import re
import sys
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.feature import ChiSqSelector
import time
import pandas as pd
from pyspark.sql import functions as F

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MultilabelMetrics
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.classification import *


# Get absolute file path in an OS independent way
def get_file_path(file_name):
    return os.path.join(os.path.dirname(__file__), file_name)

def get_classifier(labelCol="label", featuresCol="features"):
    if len(sys.argv) < 2:
        # Use RandomForest as default classifier with 10 trees
        print("\nNo classifier provided. Using default classifier: RandomForest with 10 trees")
        return RandomForestClassifier(labelCol=labelCol, featuresCol=featuresCol, numTrees=10)
    else:
        args = sys.argv[1]
        # first token from colon separated string
        classifier_arg = args.split(":")
        classifier = classifier_arg[0]
        if classifier == "rf":
            num_trees = 10 if len(classifier_arg) == 1 or classifier_arg == "" else int(classifier_arg[1])
            print("\n Training with RandomForest classifier with ", num_trees, " trees")
            return RandomForestClassifier(labelCol=labelCol, featuresCol=featuresCol, numTrees=num_trees)
        elif classifier == "dt":
            maxDepth = 5 if len(classifier_arg) == 1 or classifier_arg == "" else int(classifier_arg[1])
            print(f"\n Training with DecisionTree classifier with maxDepth {maxDepth}")
            return DecisionTreeClassifier(labelCol=labelCol, featuresCol=featuresCol, maxDepth=maxDepth)
        elif classifier == "nb":
            print("\n Training with NaiveBayes classifier with default parameter multinomial")
            return NaiveBayes(labelCol=labelCol, featuresCol=featuresCol)
        else:
            print(f"Invalid classifier argument provided {classifier_arg}. Using default classifier: RandomForest")
            return RandomForestClassifier(labelCol=labelCol, featuresCol=featuresCol, numTrees=10)
    
def evaluate_print_metrics(predictions):
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Accuracy: ", accuracy)
    
    # Print Confusion Matrix
    predictionAndLabels = predictions.select("label","prediction").rdd.map(lambda x: (float(x[0]), float(x[1]))).cache()
    metrics = MulticlassMetrics(predictionAndLabels)
    
    # Overall statistics
    precision = metrics.precision(1.0)
    print("\nPrecision: ", precision)
    recall = metrics.recall(1.0)
    print("\nRecall: ", recall)
    f1Score = metrics.fMeasure(1.0)
    print("\nF1 Score: ", f1Score)

    confusion_matrix = metrics.confusionMatrix().toArray()
    print("Confusion Matrix:\n", confusion_matrix.astype(int))


# Export the tree structure to a DOT format
def export_tree_to_dot(model, feature_names, out_file):
    with open(out_file, 'w') as f:
        f.write('digraph Tree {\n')
        f.write('node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;\n')
        f.write('edge [fontname=helvetica] ;\n')
        f.write(model.toDebugString.replace('ClassificationModel', ''))
        f.write('}\n')

def main():
    dataset_file = get_file_path("agaricus-lepiota.data")
    print("Loading Dataset from file: " + dataset_file)

    # Initialize Spark Session
    sc = SparkContext.getOrCreate()
    spark = SparkSession.builder.getOrCreate()

    # Read the data from the file "agaricus-lepiota.data"
    # The data is in CSV format
    # columns: cap-shape,cap-surface,cap-color,bruises,odor,gill-attachment,gill-spacing,gill-size,gill-color,stalk-shape,stalk-root,stalk-surface-above-ring,stalk-surface-below-ring,stalk-color-above-ring,stalk-color-below-ring,veil-type,veil-color,ring-number,ring-type,spore-print-color,population,habitat
    # First column is the label (e/p)

    cols = ["label","cap-shape","cap-surface","cap-color","bruises","odor","gill-attachment","gill-spacing",
            "gill-size","gill-color","stalk-shape","stalk-root","stalk-surface-above-ring",
            "stalk-surface-below-ring","stalk-color-above-ring","stalk-color-below-ring","veil-type",
            "veil-color","ring-number","ring-type","spore-print-color","population","habitat"]
    # create dataframe
    df = spark.read.csv(dataset_file, header=False, inferSchema="false")
    # rename columns
    df = df.toDF(*cols)

    ### CLEANING
    # drop col "veil-type" as it has only one value "p"
    df = df.drop("veil-type")
    print("\nSample Full Data")
    df.show(5, False)

    ### PREPROCESSING
    # Change label column value to [1=poisonous, 0=edible]
    df = df.withColumn("label", F.when(F.col("label") == "e", 0).otherwise(1))
    # Split the data into training and test sets with 80% training and 20% test
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)


    df_cols = [col for col in df.columns if col != "label"]
    index_cols = [column + "_index" for column in df_cols]
    vec_cols = [column + "_vec" for column in df_cols]

    ### NORMALIZATION
    # Index
    indexers = [StringIndexer(inputCol=column, outputCol=column+"_index") for column in df_cols]
    # One hot encoding
    encoders = [OneHotEncoder(inputCol=index_cols[i], outputCol=vec_cols[i]) for i in range(len(index_cols))]
    # Vector assembler
    assembler = VectorAssembler(inputCols=vec_cols, outputCol="features")

    ### MODEL TRAINING
    #classifier = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10)
    classifier = get_classifier()

    stages = indexers + encoders + [assembler, classifier]
    pipeline = Pipeline(stages=stages)
    model = pipeline.fit(train_df)

    # Print debug string of model
    #print(model.stages[-1].toDebugString)


    ### PREDICTION
    predictions = model.transform(test_df)
    predictions.select("label", "prediction").show(5) 
    
    ### EVALUATION
    evaluate_print_metrics(predictions)

    ### WITH FEATURE SELECTION IF ARGUMENTS ARE PROVIDED
    if len(sys.argv) > 2:
        top_features = sys.argv[2]
        # check if top_features is a number
        if top_features.isdigit():
            top_features = int(top_features)
            print("\nUsing Feature Selection with ChiSqSelector and top features: ", top_features)
            selector = ChiSqSelector(numTopFeatures=top_features, featuresCol="features", outputCol="selectedFeatures", labelCol="label")
            classifier = get_classifier(featuresCol="selectedFeatures")
            stages = indexers + encoders + [assembler, selector, classifier]
            pipeline = Pipeline(stages=stages)
            model = pipeline.fit(train_df)
            predictions = model.transform(test_df)
            evaluate_print_metrics(predictions)
        else:
            print("\nInvalid argument provided for feature selection. Please provide a number for top features")

    #export_tree_to_dot(model.stages[-1], df_cols, get_file_path("tree.dot"))

if __name__ == "__main__":
    main()