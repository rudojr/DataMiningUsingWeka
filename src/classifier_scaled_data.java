import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.functions.SimpleLinearRegression;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;

public class classifier_scaled_data {
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
        try {
            // Load ARFF file
            ArffLoader loader = new ArffLoader();
            loader.setSource(new File("Data/output.arff"));
            Instances data = loader.getDataSet();
            minMaxScaling(data);
            data.setClassIndex(0);

            int trainSize = (int) Math.round(data.size() * 0.8);
            int testSize = data.size() - trainSize;
            Instances trainData = new Instances(data, 0, trainSize);
            Instances testData = new Instances(data, trainSize, testSize);

            //Evaluation 10 time
            for (int i = 0; i < 4; i++) {
                System.out.println("=== Evaluation Run " + (i + 1) + " ===");

                // Train and evaluate LinearRegression model
                LinearRegression linearRegression = new LinearRegression();
                linearRegression.buildClassifier(trainData);
                evaluateModel(linearRegression, "Linear Regression", testData);

                // Train and evaluate SimpleLinearRegression model
                SimpleLinearRegression simpleLinearRegression = new SimpleLinearRegression();
                simpleLinearRegression.buildClassifier(trainData);
                evaluateModel(simpleLinearRegression, "Simple Linear Regression", testData);

                // Train and evaluate SMOreg (Support Vector Regression) model
                SMOreg smoReg = new SMOreg();
                smoReg.buildClassifier(trainData);
                evaluateModel(smoReg, "SMOreg (Support Vector Regression)", testData);

                System.out.println();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
