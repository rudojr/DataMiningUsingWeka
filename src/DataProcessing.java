import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instance;
import weka.core.converters.ArffSaver;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToNominal;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class DataProcessing {

    private static Instances loadData(String filePath) throws Exception
    {
        DataSource dataSource = new DataSource(filePath);
        Instances data = dataSource.getDataSet();
        return data;
    }

    private static int countNAValues(Instances data) {
        int count = 0;
        for (int i = 0; i < data.numAttributes(); i++) {
            for (int j = 0; j < data.numInstances(); j++) {
                Instance instance = data.instance(j);
                if (instance.toString(i).equals("NA")) {
                    count++;
                }
            }
        }
        return count;
    }

    private static void removeRowsWithNA(Instances data) {
        for (int i = data.numInstances() - 1; i >= 0; i--) {
            Instance instance = data.instance(i);
            for (int j = 0; j < instance.numAttributes(); j++) {
                if (instance.toString(j).equals("NA")) {
                    data.delete(i);
                    break;
                }
            }
        }
    }

    private static int[] findStringColumns(Instances data) {
        int numAttributes = data.numAttributes();
        int[] stringColumns = new int[numAttributes];
        int count = 0;

        for (int i = 0; i < numAttributes; i++) {
            if (data.attribute(i).isString()) {
                stringColumns[count++] = i;
            }
        }

        return Arrays.copyOf(stringColumns, count);
    }

    public static void saveInstancesToCSV(Instances data, String outputFilePath) {
        try {
            // Initialize CSVSaver
            CSVSaver csvSaver = new CSVSaver();
            csvSaver.setInstances(data);

            // Set the destination file
            csvSaver.setFile(new File(outputFilePath));

            // Save as CSV
            csvSaver.writeBatch();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void convertCSVtoARFF(String inputFilePath, String outputFilePath) {
        try {
            // Load data from CSV using CSVLoader
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(inputFilePath));
            Instances data = loader.getDataSet();

            // Save data as ARFF using ArffSaver
            ArffSaver saver = new ArffSaver();
            saver.setInstances(data);
            saver.setFile(new File(outputFilePath));
            saver.writeBatch();

            System.out.println("Data converted from CSV to ARFF successfully.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    public static void main(String[] args)
    {
        try {

            //Load data from csv and summary
            Instances data = loadData("Dataa/wind_dataset.csv");
            System.out.println("Data loaded successfully.\n");
            System.out.println(data.toSummaryString());
            System.out.println("Number of instances before clean: " + data.numInstances());
            System.out.println("Number of attributes before clean: " + data.numAttributes());
            System.out.println("=====================================\n");
            System.out.println("Data Summary\n");

            //Clean missing value
            int numNAValues = countNAValues(data);
            System.out.println("Number of NA values: " + numNAValues);
            removeRowsWithNA(data);
            System.out.println("Number of instances after removing rows with NA values: " + data.numInstances());
            System.out.println("=====================================\n");

            //Summary of data cleaned
            System.out.println("Data Cleaned Summary:\n");
            System.out.println(data.toSummaryString());
            System.out.println("Number of instances after clean: " + data.numInstances());
            System.out.println("Number of attributes after clean: " + data.numAttributes());
            System.out.println("=====================================\n");

            //Remove first column
            Remove removeFirstColumn = new Remove();
            removeFirstColumn.setAttributeIndices("1"); // Indexing starts from 1
            removeFirstColumn.setInputFormat(data);
            Instances newData = Filter.useFilter(data, removeFirstColumn);

            //Save with CSV format
            String outputFilePath = "Data/output.csv";
            saveInstancesToCSV(newData, outputFilePath);

            //Save data with ARFF
            String inputFilePath = "Data/output.csv";
            // Specify output ARFF file path
            String outputFilePathARFF = "Data/wind_dataset_cleaned.arff";

            // Load data from CSV and save it as ARFF
            convertCSVtoARFF(inputFilePath, outputFilePathARFF);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
