import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Random;

public class cross_validation {
    public static void minMaxScaling(Instances data) {
        for (int i = 0; i < data.numAttributes(); i++) {
            if (data.attribute(i).isNumeric()) {
                double min = data.attributeStats(i).numericStats.min;
                double max = data.attributeStats(i).numericStats.max;
                for (int j = 0; j < data.numInstances(); j++) {
                    double value = data.instance(j).value(i);
                    data.instance(j).setValue(i, (value - min) / (max - min));
                }
            }
        }
    }
    public static void evaluateModel(Object model, String modelName, Instances testData) {
        try {
            // Evaluate model
            Evaluation eval = new Evaluation(testData);
            eval.evaluateModel((weka.classifiers.Classifier) model, testData);

            // Print evaluation results
            System.out.println("=== " + modelName + " Evaluation ===");
            System.out.println(eval.toSummaryString());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    public static void main(String[] args) {
        try
        {
            DataSource source = new DataSource("Data/wind_dataset_cleaned.arff");
            Instances data = source.getDataSet();
//        minMaxScaling(data);
            data.setClassIndex(0); // Set index of the class attribute

            // Build and evaluate the model
            LinearRegression linearRegression = new LinearRegression();
            linearRegression.buildClassifier(data);
            System.out.println(linearRegression);
            Evaluation evaluation = new Evaluation(data);
            evaluation.crossValidateModel(linearRegression, data, 10, new Random(1));

            // Print evaluation metrics
            System.out.println("Cross - Validtaion 10 time.");
            System.out.println("Mean Absolute Error: " + evaluation.meanAbsoluteError());
            System.out.println("Root Mean Squared Error: " + evaluation.rootMeanSquaredError());
            System.out.println("Relative Absolute Error: " + evaluation.relativeAbsoluteError());
            System.out.println("Root Relative Squared Error: " + evaluation.rootRelativeSquaredError());
            System.out.println("Correlation Coefficient: " + evaluation.correlationCoefficient());
        } catch (Exception e)
        {
            e.printStackTrace();
        }
    }
}
