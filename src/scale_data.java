import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.classifiers.functions.MultilayerPerceptron;
import java.io.File;

public class scale_data {
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
            ArffLoader loader = new ArffLoader();
            loader.setSource(new File("Data/output.arff"));
            Instances data = loader.getDataSet();
            minMaxScaling(data);
            data.setClassIndex(0);

            int trainSize = (int) Math.round(data.size() * 0.8);
            int testSize = data.size() - trainSize;
            Instances trainData = new Instances(data, 0, trainSize);
            Instances testData = new Instances(data, trainSize, testSize);

//            System.out.println(trainData);
//            System.out.println(testData);


            MultilayerPerceptron mlp = new MultilayerPerceptron();
            mlp.buildClassifier(trainData);
            System.out.println(mlp);
            evaluateModel(mlp, "Multilayer Perceptron", testData);

        } catch (Exception e)
        {
            e.printStackTrace();
        }
    }
}
