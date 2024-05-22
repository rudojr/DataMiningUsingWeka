import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.SimpleLinearRegression;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.functions.SGD;
import weka.classifiers.functions.SimpleLogistic;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;
public class Classifier {
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
            data.setClassIndex(0);

            int trainSize = (int) Math.round(data.size() * 0.8);
            int testSize = data.size() - trainSize;
            Instances trainData = new Instances(data, 0, trainSize);
            Instances testData = new Instances(data, trainSize, testSize);

            //Evaluation 10 time
            for (int i = 0; i < 10; i++) {
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
