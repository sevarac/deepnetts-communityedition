/**
 *  DeepNetts is pure Java Deep Learning Library with support for Backpropagation
 *  based learning and image recognition.
 *
 *  Copyright (C) 2017  Zoran Sevarac <sevarac@gmail.com>
 *
 * This file is part of DeepNetts.
 *
 * DeepNetts is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program. If not, see <https://www.gnu.org/licenses/>.package
 * deepnetts.core;
 */
package deepnetts.net.layers;

import deepnetts.net.layers.activation.ActivationFunctions;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.core.DeepNetts;
import deepnetts.net.layers.activation.Relu;
import deepnetts.net.layers.activation.Sigmoid;
import deepnetts.net.train.Optimizers;
import deepnetts.net.train.optimizer.Optimizer;
import deepnetts.net.train.optimizer.SGDOptimizer;
import deepnetts.util.WeightsInit;
import deepnetts.util.Tensor;
import java.util.Arrays;
import java.util.logging.Logger;
import deepnetts.net.layers.activation.ActivationFunction;

/**
 * Fully connected layer has a single row of neurons connected to all neurons in
 * previous and next layer.
 *
 * Next layer can be fully connected or output Previous layer can be fully
 * connected, input, convolutional or max pooling
 *
 * @author Zoran Sevarac
 */
public final class DenseLayer extends AbstractLayer {

    private static Logger LOG = Logger.getLogger(DeepNetts.class.getName());

    /**
     * Creates an instance of fully connected layer with specified width (number
     * of neurons) and sigmoid activation function.
     *
     * @param width layer width / number of neurons in this layer
     */
    public DenseLayer(int width) {
        this.width = width;
        this.height = 1;
        this.depth = 1;
        this.activationType = ActivationType.SIGMOID;
        this.activation = ActivationFunction.create(activationType);
    }

    /**
     * Creates an instance of fully connected layer with specified width (number
     * of neurons) and activation function.
     *
     * @param width layer width / number of neurons in this layer
     * @param activationFunction activation function to use with this layer
     * @see ActivationFunctions
     */
    public DenseLayer(int width, ActivationType actType) {
        this(width);
        this.activationType = actType;
        this.activation = ActivationFunction.create(actType);
    }

    /**
     * Creates all data strucutres: inputs, weights, biases, outputs, deltas,
     * deltaWeights, deltaBiases prevDeltaWeights, prevDeltaBiases. Init weights
     * and biases. This method is called from network builder during
     * initialisation
     */
    @Override
    public void init() {
        // check prev and next layers and throw exception if its illegall architecture
        // if next layer is conv or max throw exception UnsupportedArchitectureException - nema smisla

        inputs = prevLayer.outputs;
        outputs = new Tensor(width);
        deltas = new Tensor(width);

        // sta ako je input layer a nema vise dimenzija nego samo jednu?
        if (prevLayer instanceof DenseLayer) { // ovo ako je prethodni 1d layer, odnosno ako je prethodni fully connected
            weights = new Tensor(prevLayer.width, width);
            deltaWeights = new Tensor(prevLayer.width, width); // dont store delta weighst but gradients? thas seem s betetr soltion
            gradients = new Tensor(prevLayer.width, width);
            prevDeltaWeights = new Tensor(prevLayer.width, width);

            prevGradSqrSum = new Tensor(prevLayer.width, width);  // for AdaGrad
            prevDeltaWeightSqrSum = new Tensor(prevLayer.width, width);
            prevBiasSqrSum = new Tensor(width);
            prevDeltaBiasSqrSum = new Tensor(width);

            WeightsInit.xavier(weights.getValues(), prevLayer.width, width);
            // WeightsInit.randomize(weights.getValues());

        } else if ((prevLayer instanceof MaxPoolingLayer) || (prevLayer instanceof ConvolutionalLayer) || (prevLayer instanceof InputLayer)) { // ako je pooling ili konvolucioni 2d ili 3d
            weights = new Tensor(prevLayer.width, prevLayer.height, prevLayer.depth, width); // ovde bi trebalo: prevLayer.width, prevLayer.height, width, prevLayer.depth ili prevLayer.width, prevLayer.height, prevLayer.depth, width,
            deltaWeights = new Tensor(prevLayer.width, prevLayer.height, prevLayer.depth, width);
            gradients = new Tensor(prevLayer.width, prevLayer.height, prevLayer.depth, width);
            prevDeltaWeights = new Tensor(prevLayer.width, prevLayer.height, prevLayer.depth, width);

            prevGradSqrSum = new Tensor(prevLayer.width, prevLayer.height, prevLayer.depth, width);
            prevBiasSqrSum = new Tensor(width);

            prevDeltaWeightSqrSum = new Tensor(prevLayer.width, prevLayer.height, prevLayer.depth, width); // ada delta
            prevDeltaBiasSqrSum = new Tensor(width);

            int totalInputs = prevLayer.getWidth() * prevLayer.getHeight() * prevLayer.getDepth();
            WeightsInit.xavier(weights.getValues(), totalInputs, width);
        }

        biases = new float[width];
        deltaBiases = new float[width];
        prevDeltaBiases = new float[width];
        WeightsInit.randomize(biases);
    }

    // TODO: idalno bi bilo da ova metoda ima samo jednu granu bez obzira na dimenzije prethodnog lejera,
    // to se verovatno postize broadcastingom po prethodnom lejeru sa dimenzijama 1
    // problem j eindeksiranje tezina u zavisnosti od broja dimenzija
    // resenje je da outCol u weights bude prva dimenzija u get metodi!
    // prouci formule u https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html
    @Override
    public void forward() {
        // pokusaj generalizacije da ima samo jednu granu bez obzira na dimenzije prethodnog lejera
//        int inRow = 0, inDepth = 0;
//        outputs.copyFrom(biases);                                             // first use (add) biases to all outputs
//        for (int outCol = 0; outCol < outputs.getCols(); outCol++) {          // for all neurons/outputs in this layer
////            for (int inDepth = 0; inDepth < inputs.getDepth(); inDepth++) {   // iterate depth from prev/input layer
////                for (int inRow = 0; inRow < inputs.getRows(); inRow++) {      // iterate current channel by height (rows)
//            for (int inCol = 0; inCol < inputs.getCols(); inCol++) {   // iterate current feature map by width (cols)
//                // puca zbog redosleda dimenzija u weights tensoru, outCol tumaci kao forth dimenziju i index odmah iskace cim outCol predje na 1
//                final float wi = inputs.get(inRow, inCol, inDepth) * weights.get(inCol, inRow, inDepth, outCol);
//                outputs.add(outCol, wi); // puca na 50 iteraciju
////                outputs.add(outCol, inputs.get(inCol) * weights.get(inCol, outCol));
//            }
////                }
//            //  }
//            // apply activation function to all weigthed sums stored in outputs
//            outputs.set(outCol, ActivationFunctions.calc(activationType, outputs.get(outCol)));
//        }
//        if previous layer is DenseLayer
        if (prevLayer instanceof DenseLayer) {
            // weighted sum of input and weight tensors with added biases
            // folowed by activation function applied to each output element

            // pomonizi ulazni vektor sa matricom tezina, na to dodaj bias i sve to propusti kroz aktivacionu funkciju
            // mozda najbolje ovo benchmarkovati nezavisno od ovog koda koji radi pa onda odluciti kako implementirati
            outputs.copyFrom(biases);                                                       // first use (add) biases to all outputs
            // put gtters cols in fibal locals
            for (int outCol = 0; outCol < outputs.getCols(); outCol++) {                    // for all neurons/outputs in this layer
                for (int inCol = 0; inCol < inputs.getCols(); inCol++) {                    // iterate all inputs from prev layer
                    outputs.add(outCol, inputs.get(inCol) * weights.get(inCol, outCol));    // and add weighted sum to outputs
                }

                // apply activation function to all weigthed sums stored in outputs
                // ovo moze i nezavisno drugi thread da radi samo ne sme da pretekne mnozenje matrica
                //outputs.set(outCol, ActivationFunctions.calc(activationType, outputs.get(outCol))); // this could be lambda or Function - apply it to entire Tensor
                // outputs.applyFunction(activationFunction);
                // outputs.applyFunction(activationFunction); // apply activation function to all elements of the tensor/vector
                //outputs.set(outCol, activation.getValue(outputs.get(outCol)));                                 
            }
            outputs.apply(activation::getValue);
        } // if previous layer is MaxPooling, Convolutional or input layer (2D or 3D) - TODO: posto je povezanost svi sa svima ovo mozda moze i kao 1d na 1d niz, verovatno je efikasnije
        else if ((prevLayer instanceof MaxPoolingLayer) || (prevLayer instanceof ConvolutionalLayer) || (prevLayer instanceof InputLayer)) { // povezi sve na sve
            // prethodni loop se svodi na ovaj pri cemu su inRow i inDepth 1
            outputs.copyFrom(biases);                                             // first use (add) biases to all outputs
            for (int outCol = 0; outCol < outputs.getCols(); outCol++) {          // for all neurons/outputs in this layer
                for (int inDepth = 0; inDepth < inputs.getDepth(); inDepth++) {   // iterate depth from prev/input layer
                    for (int inRow = 0; inRow < inputs.getRows(); inRow++) {      // iterate current channel by height (rows)
                        for (int inCol = 0; inCol < inputs.getCols(); inCol++) {   // iterate current feature map by width (cols)
                            outputs.add(outCol, inputs.get(inRow, inCol, inDepth) * weights.get(inCol, inRow, inDepth, outCol)); // add to weighted sum of all inputs (TODO: ako je svaki sa svima to bi mozda moglo da bude i jednostavno i da se prodje u jednom loopu a ugnjezdeni loopovi bi se lakse paralelizovali)
                            // posto se inCol najbrze vrti  trebalo bi da bude poslednja dimenzija u tensoru  weights.get(outCol, inDepth, inRow, inCol)
                            // a spoljna dimenzija se moze paralelizovati  (pola niza radi jedan thread drugu polovinu drugi)- optimizuj celinu!
                            // glavno pitanje je kako to uraditi pa da se omoguci laka implementacija broadcastinga i paralelizacija sa threadovima i GPU-om
                            // napravi implementaciju koja ce da zakuca 4d tensor
                            //  cilj je da imam samo jednu granu dfa nema ovog if, nego da radi broadcasting zapravo, ali zakucan na 4 dimenzije
                        }
                    }
                }
                // apply activation function to all weigthed sums stored in outputs
                //outputs.set(outCol, ActivationFunctions.calc(activationType, outputs.get(outCol)));
                outputs.set(outCol, activation.getValue(outputs.get(outCol)));
            }
        }
    }

    @Override
    public void backward() {
        if (!batchMode) { // if online mode reset deltaWeights and deltaBiases to zeros
            deltaWeights.fill(0);
            Arrays.fill(deltaBiases, 0);
        }

        deltas.fill(0); // reset current delta

        // STEP 1. propagate weighted deltas from next layer (which can be output or fully connected) and calculate deltas for this layer
        for (int deltaCol = 0; deltaCol < deltas.getCols(); deltaCol++) {   // for every neuron/delta in this layer
            for (int ndCol = 0; ndCol < nextLayer.deltas.getCols(); ndCol++) { // iterate all deltas from next layer
                deltas.add(deltaCol, nextLayer.deltas.get(ndCol) * nextLayer.weights.get(deltaCol, ndCol)); // calculate weighted sum of deltas from the next layer
            }

//            final float delta = deltas.get(deltaCol) * ActivationFunctions.prime(activationType, outputs.get(deltaCol));
            final float delta = deltas.get(deltaCol) * activation.getPrime(outputs.get(deltaCol));
            deltas.set(deltaCol, delta);
        } // end sum weighted deltas from next layer

        // STEP 2. calculate delta weights if previous layer is Dense (2D weights matrix) - optimize
        if ((prevLayer instanceof DenseLayer)) {
//            Optimizer opt = new SGDOptimizer(); // create instance in init method
//            opt.optimize(this);
            for (int deltaCol = 0; deltaCol < deltas.getCols(); deltaCol++) { // this iterates neurons (weights depth)
                for (int inCol = 0; inCol < inputs.getCols(); inCol++) {
                    final float grad = deltas.get(deltaCol) * inputs.get(inCol); // gradient dE/dw
                    gradients.set(inCol, deltaCol, grad);

                    float deltaWeight = 0;
                    // OPTIMIZER TREBA DA ima referencu na layer i da sam uzima sta mu treba. Ili da sam optimizer kod sebe skladisti stukrure podataka potrebne za konkretan algoritam
                    // napravi interfejs optimizer, konkretna implementacija d aima referencu na lejer, i da sadrzi sve strukture podataka koje su potrebne za konkretan optimizer
                    // idealno da radi nad matricama a ne nad pojedinacnim vrednostima
                    // treba da radi nad weights i biasom
                    // svi optimizeri mogu da nasledjuju jedan osnovni koji vrti petlju i da imaju templejt metod koji predstavlja konkretnu implementaciju
                    switch (optimizer) {
                        case SGD:
                            deltaWeight = Optimizers.sgd(learningRate, grad);
                            break;
                        case MOMENTUM:
                            deltaWeight = Optimizers.momentum(learningRate, grad, momentum, prevDeltaWeights.get(inCol, deltaCol));
                            break;
                        case ADAGRAD:
                            prevGradSqrSum.add(inCol, deltaCol, grad * grad); // da li ovo treba resetovati na nulu nekad? da lije samo za mini batch mode?
                            deltaWeight = Optimizers.adaGrad(learningRate, grad, prevGradSqrSum.get(inCol, deltaCol));
                            break;
                        case RMSPROP:
                            final float gama = 0.9f;
                            prevGradSqrSum.add(inCol, deltaCol,
                                    gama * prevGradSqrSum.get(inCol, deltaCol) + (1 - gama) * grad * grad);
                            deltaWeight = Optimizers.rmsProp(learningRate, grad, prevGradSqrSum.get(inCol, deltaCol));
                            break;
                        case ADADELTA:
                            //gama = 0.9f;
                            prevGradSqrSum.add(inCol, deltaCol,
                                    0.9f * prevGradSqrSum.get(inCol, deltaCol) + 0.1f * grad * grad);
                            deltaWeight = Optimizers.adaDelta(grad, prevGradSqrSum.get(inCol, deltaCol), prevDeltaWeightSqrSum.get(inCol, deltaCol));
                            prevDeltaWeightSqrSum.add(inCol, deltaCol,
                                    0.9f * prevDeltaWeightSqrSum.get(inCol, deltaCol) + 0.1f * deltaWeight * deltaWeight);
                            break;
//                        case ADAM:
//                            throw new NotImplementedException("Adam optimizer is not implemented yet");
                    }

                    deltaWeights.add(inCol, deltaCol, deltaWeight);

                }

                float deltaBias = 0;
                switch (optimizer) {
                    case SGD:
                        deltaBias = Optimizers.sgd(learningRate, deltas.get(deltaCol));
                        break;
                    case MOMENTUM:
                        deltaBias = Optimizers.momentum(learningRate, deltas.get(deltaCol), momentum, prevDeltaBiases[deltaCol]);
                        break;
                    case ADAGRAD:
                        prevBiasSqrSum.add(deltaCol, deltas.get(deltaCol) * deltas.get(deltaCol));
                        deltaBias = Optimizers.adaGrad(learningRate, deltas.get(deltaCol), prevBiasSqrSum.get(deltaCol));
                        break;
                    case RMSPROP:
                        final float gama = 0.9f;
                        prevBiasSqrSum.add(deltaCol,
                                gama * prevBiasSqrSum.get(deltaCol) + (1 - gama) * deltas.get(deltaCol) * deltas.get(deltaCol));
                        deltaBias = Optimizers.rmsProp(learningRate, deltas.get(deltaCol), prevBiasSqrSum.get(deltaCol));
                        break;
                    case ADADELTA:
                        //gama = 0.9f;
                        prevBiasSqrSum.add(deltaCol,
                                0.9f * prevBiasSqrSum.get(deltaCol) + 0.1f * deltas.get(deltaCol) * deltas.get(deltaCol));
                        deltaBias = Optimizers.adaDelta(deltas.get(deltaCol), prevBiasSqrSum.get(deltaCol), prevDeltaBiasSqrSum.get(deltaCol));
                        prevDeltaBiasSqrSum.add(deltaCol,
                                0.9f * prevDeltaBiasSqrSum.get(deltaCol) + 0.1f * deltas.get(deltaCol));
                        break;
                }

                deltaBiases[deltaCol] += deltaBias;
            }
        } else if ((prevLayer instanceof InputLayer) // CHECK: sta ako j einput layer 1d da li ovo radi akko treba?
                || (prevLayer instanceof ConvolutionalLayer)
                || (prevLayer instanceof MaxPoolingLayer)) {

            for (int deltaCol = 0; deltaCol < deltas.getCols(); deltaCol++) { // for all neurons/deltas in this layer
                for (int inDepth = 0; inDepth < inputs.getDepth(); inDepth++) { // iterate all inputs from previous layer
                    for (int inRow = 0; inRow < inputs.getRows(); inRow++) {
                        for (int inCol = 0; inCol < inputs.getCols(); inCol++) {
                            final float grad = deltas.get(deltaCol) * inputs.get(inRow, inCol, inDepth);  // da li je ovde greska treba ih sumitrati sve tri po dubini  // da li ove ulaze treba sabirati??? jer jedna celija ima ulaze iz tri prethodna kanala?
                            gradients.set(inCol, inRow, inDepth, grad);   // da li ovo radi kada je fc sa obicnim input lyerom -  proveri??

                            float deltaWeight = 0;
                            switch (optimizer) {
                                case SGD:
                                    deltaWeight = Optimizers.sgd(learningRate, grad);
                                    break;
                                case MOMENTUM:
                                    deltaWeight = Optimizers.momentum(learningRate, grad, momentum, prevDeltaWeights.get(inCol, inRow, inDepth, deltaCol));//
                                    break;
                                case ADAGRAD:
                                    prevGradSqrSum.add(inCol, inRow, inDepth, deltaCol, grad * grad);
                                    deltaWeight = Optimizers.adaGrad(learningRate, grad, prevGradSqrSum.get(inCol, inRow, inDepth, deltaCol));
                                    break;
                                case RMSPROP:
                                    final float gama = 0.9f;
                                    prevGradSqrSum.add(inCol, inRow, inDepth, deltaCol,
                                            gama * prevGradSqrSum.get(inCol, inRow, inDepth, deltaCol) + (1 - gama) * grad * grad);
                                    deltaWeight = Optimizers.rmsProp(learningRate, grad, prevGradSqrSum.get(inCol, inRow, inDepth, deltaCol));
                                    break;
                                case ADADELTA:
                                    //gama = 0.9f;
                                    prevGradSqrSum.add(inCol, inRow, inDepth, deltaCol,
                                            0.9f * prevGradSqrSum.get(inCol, inRow, inDepth, deltaCol) + 0.1f * grad * grad);
                                    deltaWeight = Optimizers.adaDelta(grad, prevGradSqrSum.get(inCol, inRow, inDepth, deltaCol), prevDeltaWeightSqrSum.get(inCol, inRow, inDepth, deltaCol));
                                    prevDeltaWeightSqrSum.add(inCol, inRow, inDepth, deltaCol,
                                            0.9f * prevDeltaWeightSqrSum.get(inCol, inRow, inDepth, deltaCol) + 0.1f * deltaWeight * deltaWeight);
                                    break;
                            }

                            deltaWeights.add(inCol, inRow, inDepth, deltaCol, deltaWeight);
                        }
                    }
                }

                float deltaBias = 0;
                switch (optimizer) {
                    case SGD:
                        deltaBias = Optimizers.sgd(learningRate, deltas.get(deltaCol));
                        break;
                    case MOMENTUM:
                        deltaBias = Optimizers.momentum(learningRate, deltas.get(deltaCol), momentum, prevDeltaBiases[deltaCol]);
                        break;
                    case ADAGRAD:
                        prevBiasSqrSum.add(deltaCol, deltas.get(deltaCol) * deltas.get(deltaCol));
                        deltaBias = Optimizers.adaGrad(learningRate, deltas.get(deltaCol), prevBiasSqrSum.get(deltaCol));
                        break;
                    case RMSPROP:
                        final float gama = 0.9f;
                        prevBiasSqrSum.add(deltaCol,
                                gama * prevBiasSqrSum.get(deltaCol) + (1 - gama) * deltas.get(deltaCol) * deltas.get(deltaCol));
                        deltaBias = Optimizers.rmsProp(learningRate, deltas.get(deltaCol), prevBiasSqrSum.get(deltaCol));
                        break;
                    case ADADELTA:
                        //gama = 0.9f;
//                        prevBiasSqrSum.add(deltaCol,
//                                0.9f * prevBiasSqrSum.get(deltaCol) + 0.1f * deltas.get(deltaCol));
//                        deltaBias = Optimizers.adaDelta(deltas.get(deltaCol), prevBiasSqrSum.get(deltaCol), prevDeltaBiasSqrSum.get(deltaCol));
//                        prevDeltaBiasSqrSum.add(deltaCol,
//                                0.9f * prevDeltaBiasSqrSum.get(deltaCol) + 0.1f * deltas.get(deltaCol) * deltas.get(deltaCol));
//                        break;

                        prevBiasSqrSum.add(deltaCol,
                                0.9f * prevBiasSqrSum.get(deltaCol) + 0.1f * deltas.get(deltaCol) * deltas.get(deltaCol));
                        deltaBias = Optimizers.rmsProp(learningRate, deltas.get(deltaCol), prevBiasSqrSum.get(deltaCol));
                        break;
                }

                deltaBiases[deltaCol] += deltaBias;
            }
        }
    }

    @Override
    public void applyWeightChanges() {
        if (batchMode) { // podeli Delta weights sa brojem uzoraka odnosno backward passova
            deltaWeights.div(batchSize);
            Tensor.div(deltaBiases, batchSize);
        }

        Tensor.copy(deltaWeights, prevDeltaWeights); // save as prev delta weight
        Tensor.copy(deltaBiases, prevDeltaBiases);

        weights.add(deltaWeights);
        Tensor.add(biases, deltaBiases);

        if (batchMode) {
            deltaWeights.fill(0);
            Tensor.fill(deltaBiases, 0);
        }

    }

}
