import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;

public class TradingNeuralNetwork {
    public static void main(String[] args) throws Exception {
        // Konfiguracja Deeplearning4j
        System.setProperty("DL4J_USE_CUDA", "true"); // Włącz obsługę CUDA (GPU)

        List<Double> openPrices = new ArrayList<>();
        List<Double> closePrices = new ArrayList();
        List<Double> highPrices = new ArrayList();
        List<Double> lowPrices = new ArrayList();

        try {
            Reader reader = new FileReader("D:\\Projekty Programowanie\\Java\\TradingNN\\src\\main\\resources\\daneD.csv");
            CSVParser csvParser = new CSVParser(reader, CSVFormat.DEFAULT.withDelimiter(';').withFirstRecordAsHeader());
            List<CSVRecord> records = csvParser.getRecords();
            for (CSVRecord record : records) {
                String time = record.get("Time (UTC)");
                double open = Double.parseDouble(record.get("Open").replace(",", "."));
                double high = Double.parseDouble(record.get("High").replace(",", "."));
                double low = Double.parseDouble(record.get("Low").replace(",", "."));
                double close = Double.parseDouble(record.get("Close").replace(",", "."));
                openPrices.add(open);
                closePrices.add(close);
                highPrices.add(high);
                lowPrices.add(low);
            }

            csvParser.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        double[] openArray = openPrices.stream().mapToDouble(Double::doubleValue).toArray();
        double[] closeArray = closePrices.stream().mapToDouble(Double::doubleValue).toArray();
        double[] highArray = highPrices.stream().mapToDouble(Double::doubleValue).toArray();
        double[] lowArray = lowPrices.stream().mapToDouble(Double::doubleValue).toArray();

        // Przygotowanie danych treningowych
        List<DataSet> trainingData = new ArrayList<>();
        int sequenceLength = 20;
        for (int i = 0; i < openArray.length - sequenceLength; i++) {
            INDArray features = Nd4j.create(new double[sequenceLength * 4]); // 4 cechy na świecę
            INDArray labels = Nd4j.create(new double[]{closeArray[i + sequenceLength]});

            for (int j = 0; j < sequenceLength; j++) {
                int index = i + j;
                features.putScalar(j * 4, openArray[index]);
                features.putScalar(j * 4 + 1, highArray[index]);
                features.putScalar(j * 4 + 2, lowArray[index]);
                features.putScalar(j * 4 + 3, closeArray[index]);
            }

            features = features.reshape(1, sequenceLength * 4);
            labels = labels.reshape(1, 1);
            DataSet dataSet = new DataSet(features, labels);
            trainingData.add(dataSet);
        }
        /*for (int i = 0; i < openArray.length; i++) {
            INDArray labels = null;
            INDArray features = Nd4j.create(new double[]{openArray[i], highArray[i], lowArray[i], closeArray[i]});
            if (i <= closeArray.length){
                labels = Nd4j.create(new double[]{closeArray[i] + 1});
            }

            features = features.reshape(1, 4); // 3 cechy
            labels = labels.reshape(1, 1);
            DataSet dataSet = new DataSet(features, labels);
            trainingData.add(dataSet);
        }*/

        DataSetIterator dataSetIterator = new ListDataSetIterator<>(trainingData, trainingData.size());

        // Konfiguracja sieci neuronowej
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(0.01))
                .weightInit(WeightInit.XAVIER)
                .list()
                .setInputType(InputType.recurrent(sequenceLength * 4)) // 3 wejścia
                .layer(0, new DenseLayer.Builder()
                        .nIn(4)
                        .nOut(1000)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(1000)
                        .nOut(1000)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nIn(1000)
                        .nOut(1000)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new DenseLayer.Builder()
                        .nIn(1000)
                        .nOut(1000)
                        .activation(Activation.RELU)
                        .build())
                .layer(4, new OutputLayer.Builder(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
                        .nIn(1000)
                        .nOut(1)
                        .activation(Activation.IDENTITY)
                        .build())
                .backpropType(BackpropType.Standard)
                .tBPTTForwardLength(sequenceLength)
                .tBPTTBackwardLength(sequenceLength)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        model.setListeners(new ScoreIterationListener(1));

        // Trenowanie modelu
        int numEpochs = 100;
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            model.fit(dataSetIterator);
        }

        // Ewaluacja modelu
        dataSetIterator.reset();
        RegressionEvaluation evaluation = model.evaluateRegression(dataSetIterator);
        System.out.println("Mean Squared Error: " + evaluation.meanSquaredError(0));

        // Przewidywanie na nowych danych
        double[] newInput = {1.06605, 1.07131, 1.06385, 1.06608};  // Przykładowe wartości Open, High, Low
        INDArray input = Nd4j.create(newInput).reshape(1, 4);
        INDArray predicted = model.output(input);
        System.out.println("Przewidywana wartość zamknięcia: " + predicted.getDouble(0));
    }
}