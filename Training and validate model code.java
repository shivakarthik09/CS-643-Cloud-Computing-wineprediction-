package com.wqp.spark;

import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import static org.apache.spark.sql.functions.col;

public class App {

    public static void main(String[] args) {
        try {
            // Set up the Spark session
            SparkSession spark = SparkSession.builder()
                    .appName("Wine Quality Prediction Model Training")
                    .master("local[*]") // Update this if you are running on a cluster
                    .getOrCreate();
            spark.sparkContext().setLogLevel("ERROR");

            // Define the paths for training and validation data on S3
            String trainDataPath = "s3://s3bucket9542/TrainingDataset.csv";
            String validationDataPath = "s3://s3bucket9542/ValidationDataset.csv";

            // Load and prepare the training dataset
            Dataset<Row> trainData = loadDataAndPrepare(spark, trainDataPath);

            // Define feature columns and prepare the pipeline
            String[] featureCols = {"fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                    "chlorides", "free sulfur dioxide", "total sulfur dioxide",
                    "density", "pH", "sulphates", "alcohol"};
            PipelineModel model = trainModel(trainData, featureCols);

            // Save the trained model to S3
            model.write().overwrite().save("s3://s3bucket9542/WineQualityPredictor/model");

            // Load the model from S3
            PipelineModel loadedModel = PipelineModel.load("s3://s3bucket9542/WineQualityPredictor/model");

            // Load and prepare the validation dataset
            Dataset<Row> validationData = loadDataAndPrepare(spark, validationDataPath);

            // Validate the model using the validation dataset
            Dataset<Row> validationPredictions = loadedModel.transform(validationData);

            // Evaluate the model's performance
            evaluateModel(validationPredictions);

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
        data = cleanAndPrepareData(data);
        return data;
    }

    private static Dataset<Row> cleanAndPrepareData(Dataset<Row> data) {
        // Clean the column names by removing extra quotes and spaces
        String[] cleanedColumns = data.columns();
        for (int i = 0; i < cleanedColumns.length; i++) {
            cleanedColumns[i] = cleanedColumns[i].replaceAll("\"", "").trim();
        }
        data = data.toDF(cleanedColumns);

        // Cast "quality" to integer and create binary labels for training data
        data = data.withColumn("quality", col("quality").cast("int"));
        data = data.withColumn("label", functions.when(col("quality").geq(7), 1).otherwise(0));

        // Cast all feature columns to DoubleType
        for (String column : new String[]{"fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                "chlorides", "free sulfur dioxide", "total sulfur dioxide",
                "density", "pH", "sulphates", "alcohol"}) {
            data = data.withColumn(column, col(column).cast("double"));
        }
        return data;
    }

    private static PipelineModel trainModel(Dataset<Row> trainData, String[] featureCols) {
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
                .setFeaturesCol("scaledFeatures")
                .setMaxIter(1000)
                .setRegParam(0.3)
                .setElasticNetParam(0.8);
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{assembler, scaler, lr});
        return pipeline.fit(trainData);
    }

    private static void evaluateModel(Dataset<Row> predictions) {
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);
        evaluator.setMetricName("f1");
        double f1 = evaluator.evaluate(predictions);
        evaluator.setMetricName("weightedPrecision");
        double precision = evaluator.evaluate(predictions);
        evaluator.setMetricName("weightedRecall");
        double recall = evaluator.evaluate(predictions);

        System.out.println("Test Error = " + (1 - accuracy));
        System.out.println("Accuracy = " + accuracy);
        System.out.println("F1 = " + f1);
        System.out.println("Precision = " + precision);
        System.out.println("Recall = " + recall);
    }
}