import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;

public class mohing {
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


            LinearRegression linearRegression = new LinearRegression();
            linearRegression.buildClassifier(trainData);
            System.out.println(linearRegression);
            evaluateModel(linearRegression, "Linear Regression", testData);
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }
    }
}
