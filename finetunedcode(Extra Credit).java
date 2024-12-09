package com.wqp.spark;

import org.apache.spark.ml.*;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.classification.*;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.tuning.*;
import org.apache.spark.sql.*;
import org.apache.spark.sql.functions.*;

import static org.apache.spark.sql.functions.col;

public class WineQualityPrediction {

    public static void main(String[] args) {
        try {
            // Set up the Spark session
            SparkSession spark = SparkSession.builder()
                    .appName("Improved Wine Quality Prediction")
                    .master("local[*]") // Update this for your cluster
                    .getOrCreate();
            spark.sparkContext().setLogLevel("ERROR");

            // Paths for training and validation datasets
            String trainDataPath = "s3://s3bucket9542/TrainingDataset.csv";
            String validationDataPath = "s3://s3bucket9542/ValidationDataset.csv";

            // Load and prepare datasets
            Dataset<Row> trainData = loadDataAndPrepare(spark, trainDataPath);
            Dataset<Row> validationData = loadDataAndPrepare(spark, validationDataPath);

            // Define feature columns
            String[] featureCols = {"fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
                    "density", "pH", "sulphates", "alcohol"};

            // Train the model with hyperparameter tuning
            PipelineModel bestModel = trainModelWithTuning(trainData, featureCols);

            // Save the trained model
            bestModel.write().overwrite().save("s3://s3bucket9542/WineQualityPredictor/optimizedModel");

            // Validate the model
            Dataset<Row> predictions = bestModel.transform(validationData);

            // Evaluate the model
            evaluateModel(predictions);

            // Stop Spark session
            spark.stop();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static Dataset<Row> loadDataAndPrepare(SparkSession spark, String dataPath) {
        Dataset<Row> data = spark.read()
                .option("header", "true")
                .option("delimiter", ";")
                .csv(dataPath);

        // Clean and prepare data
        data = cleanAndPrepareData(data);
        return data;
    }

    private static Dataset<Row> cleanAndPrepareData(Dataset<Row> data) {
        String[] cleanedColumns = data.columns();
        for (int i = 0; i < cleanedColumns.length; i++) {
            cleanedColumns[i] = cleanedColumns[i].replaceAll("\"", "").trim();
        }
        data = data.toDF(cleanedColumns);
        data = data.withColumn("quality", col("quality").cast("int"));
        data = data.withColumn("label", col("quality").geq(7).cast("int"));

        for (String column : new String[]{"fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                "chlorides", "free sulfur dioxide", "total sulfur dioxide",
                "density", "pH", "sulphates", "alcohol"}) {
            data = data.withColumn(column, col(column).cast("double"));
        }
        return data;
    }

    private static PipelineModel trainModelWithTuning(Dataset<Row> trainData, String[] featureCols) {
        // Define pipeline stages
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(featureCols)
                .setOutputCol("features");

        StandardScaler scaler = new StandardScaler()
                .setInputCol("features")
                .setOutputCol("scaledFeatures")
                .setWithMean(true)
                .setWithStd(true);

        LogisticRegression lr = new LogisticRegression()
                .setLabelCol("label")
                .setFeaturesCol("scaledFeatures");

        // Create pipeline
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{assembler, scaler, lr});

        // Hyperparameter tuning
        ParamGridBuilder paramGrid = new ParamGridBuilder()
                .addGrid(lr.regParam(), new double[]{0.01, 0.1, 0.3, 0.5})
                .addGrid(lr.elasticNetParam(), new double[]{0.0, 0.5, 0.8, 1.0})
                .addGrid(lr.maxIter(), new int[]{100, 500, 1000});

        TrainValidationSplit trainValidationSplit = new TrainValidationSplit()
                .setEstimator(pipeline)
                .setEvaluator(new MulticlassClassificationEvaluator().setMetricName("accuracy"))
                .setEstimatorParamMaps(paramGrid.build())
                .setTrainRatio(0.8);

        // Train and return the best model
        TrainValidationSplitModel model = trainValidationSplit.fit(trainData);
        return (PipelineModel) model.bestModel();
    }

    private static void evaluateModel(Dataset<Row> predictions) {
        // Evaluate accuracy
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);

        // Additional metrics
        evaluator.setMetricName("f1");
        double f1 = evaluator.evaluate(predictions);

        evaluator.setMetricName("weightedPrecision");
        double precision = evaluator.evaluate(predictions);

        evaluator.setMetricName("weightedRecall");
        double recall = evaluator.evaluate(predictions);

        System.out.println("Model Evaluation Metrics:");
        System.out.println("Accuracy = " + accuracy);
        System.out.println("F1 Score = " + f1);
        System.out.println("Precision = " + precision);
        System.out.println("Recall = " + recall);
    }
}
