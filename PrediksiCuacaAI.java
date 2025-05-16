/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.mavenproject_dl4j;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class PrediksiCuacaAI {

    public static void main(String[] args) {

        // === 1. Siapkan data latih sederhana (suhu, kelembapan) ===
        // Format input: suhu (°C), kelembapan (%)
        // Format label: kemungkinan hujan (1 = hujan, 0 = tidak)
        INDArray input = Nd4j.create(new double[][]{
                {30, 80},
                {25, 60},
                {35, 90},
                {20, 50},
                {28, 70}
        });

        INDArray labels = Nd4j.create(new double[][]{
                {1},
                {0},
                {1},
                {0},
                {1}
        });

        DataSet dataSet = new DataSet(input, labels);

        // === 2. Konfigurasi model jaringan saraf sederhana ===
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(123)  // Untuk hasil konsisten
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(2) // 2 input: suhu dan kelembapan
                        .nOut(4)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder()
                        .nIn(4)
                        .nOut(1)
                        .activation(Activation.SIGMOID)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        // === 3. Latih model dengan data sederhana ===
        for (int i = 0; i < 1000; i++) {
            model.fit(dataSet);
        }

        // === 4. Input baru untuk diprediksi ===
        INDArray newInput = Nd4j.create(new double[][]{
                {29, 85},  // Kasus 1
                {22, 55},  // Kasus 2
                {33, 92}   // Kasus 3
        });

        INDArray output = model.output(newInput);

        // === 5. Tampilkan hasil prediksi ===
        System.out.println("Prediksi kemungkinan hujan:");
        for (int i = 0; i < newInput.rows(); i++) {
            System.out.println("Input suhu: " + newInput.getDouble(i, 0) +
                    ", kelembapan: " + newInput.getDouble(i, 1) +
                    " → Prediksi: " + String.format("%.2f", output.getDouble(i)));
        }
    }
}
