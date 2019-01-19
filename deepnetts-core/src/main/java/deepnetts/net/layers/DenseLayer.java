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

import deepnetts.net.layers.activation.ActivationType;
import deepnetts.core.DeepNetts;
import deepnetts.util.WeightsInit;
import deepnetts.util.Tensor;
import java.util.Arrays;
import java.util.logging.Logger;
import deepnetts.net.train.opt.Optimizer;
import deepnetts.net.train.opt.OptimizerType;

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
    
    private Optimizer optim;

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
        
        setActivationType(ActivationType.SIGMOID);
    }

    /**
     * Creates an instance of fully connected layer with specified width (number
     * of neurons) and activation function.
     *
     * @param width layer width / number of neurons in this layer
     * @param actType activation function to use with this layer
     * @see ActivationFunctions
     */
    public DenseLayer(int width, ActivationType actType) {
        this(width);
        setActivationType(actType);
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
        if (prevLayer instanceof DenseLayer || (prevLayer instanceof InputLayer && prevLayer.height == 1 && prevLayer.depth == 1)) { // ovo ako je prethodni 1d layer, odnosno ako je prethodni fully connected
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
        
        if (activationType == ActivationType.RELU)
               Tensor.fill(biases, 0.5f);
            else
               WeightsInit.randomize(biases);

        setOptimizer(OptimizerType.SGD);        
    }

    @Override
    public void setOptimizer(OptimizerType optimizer) {
        super.setOptimizer(optimizer); 
        optim = Optimizer.create(optimizer, this);
    }
    

    

    // TODO: idalno bi bilo da ova metoda ima samo jednu granu bez obzira na dimenzije prethodnog lejera,
    // to se verovatno postize broadcastingom po prethodnom lejeru sa dimenzijama 1
    // problem j eindeksiranje tezina u zavisnosti od broja dimenzija
    // resenje je da outCol u weights bude prva dimenzija u get metodi!
    // prouci formule u https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html
    @Override
    public void forward() {
        // pokusaj generalizacije da ima samo jednu granu bez obzira na dimenzije prethodnog lejera
        if (prevLayer instanceof DenseLayer || (prevLayer instanceof InputLayer && prevLayer.height == 1 && prevLayer.depth == 1)) { // ovde bi rebalo or InputLayer u 2D
            // vectorized implementation:
            // weighted sum of input and weight tensors with added biases
            // folowed by activation function applied to each output element

            // pomonizi ulazni vektor sa matricom tezina, na to dodaj bias i sve to propusti kroz aktivacionu funkciju
            // mozda najbolje ovo benchmarkovati nezavisno od ovog koda koji radi pa onda odluciti kako implementirati
            outputs.copyFrom(biases);                                                       // first use (add) biases to all outputs
            // put getters cols in final locals
            for (int outCol = 0; outCol < outputs.getCols(); outCol++) {                    // for all neurons/outputs in this layer
                for (int inCol = 0; inCol < inputs.getCols(); inCol++) {                    // iterate all inputs from prev layer
                    outputs.add(outCol, inputs.get(inCol) * weights.get(inCol, outCol));    // and add weighted sum to outputs
                }
                outputs.set(outCol, activation.getValue(outputs.get(outCol)));
            }
            // outputs.apply(activation::getValue);
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
               outputs.set(outCol, activation.getValue(outputs.get(outCol)));
            }
             //outputs.apply(activation::getValue);
        }
    }

    
    //SgdOptimizer2 opt2 = new SgdOptimizer2(this); //ali kad ce mu biti setovan i learning rate, kad krene priprema za trening
    @Override
    public void backward() {
        if (!batchMode) { // if online mode reset deltaWeights and deltaBiases to zeros
            deltaWeights.fill(0);
            Arrays.fill(deltaBiases, 0);
        }

        deltas.fill(0); // reset current delta
        
        // STEP 1. propagate and sum weighted deltas from the next layer (which can be output or fully connected) to calculate deltas for this layer
        for (int deltaCol = 0; deltaCol < deltas.getCols(); deltaCol++) {   // for every neuron/delta in this layer
            for (int ndCol = 0; ndCol < nextLayer.deltas.getCols(); ndCol++) { // iterate all deltas from next layer
                deltas.add(deltaCol, nextLayer.deltas.get(ndCol) * nextLayer.weights.get(deltaCol, ndCol)); // calculate weighted sum of deltas from the next layer
            }

            final float delta = deltas.get(deltaCol) * activation.getPrime(outputs.get(deltaCol));
            deltas.set(deltaCol, delta);
        } // end sum weighted deltas from next layer

        // STEP 2. calculate delta weights if previous layer is Dense (2D weights matrix) - optimize
        if ((prevLayer instanceof DenseLayer) ||
            ((prevLayer instanceof InputLayer) && (prevLayer.height==1 && prevLayer.depth==1))   ) { // ili 1d Input Layer, dodati uslov
            
            for (int deltaCol = 0; deltaCol < deltas.getCols(); deltaCol++) { // this iterates neurons (weights depth)
                for (int inCol = 0; inCol < inputs.getCols(); inCol++) {
                   final float grad = deltas.get(deltaCol) * inputs.get(inCol); // gradient dE/dw
//                    final float grad = deltas.get(deltaCol) * inputs.get(inCol) + 2 * regularization * weights.get(inCol, deltaCol); // gradient dE/dw + L2 regularization , what if L1 regularization?
//                    final float grad = deltas.get(deltaCol) * inputs.get(inCol) + 0.01f * ( weights.get(inCol, deltaCol)>=0? 1 : -1 );

                    gradients.set(inCol, deltaCol, grad);

                    final float deltaWeight = optim.calculateWeightDelta(grad, inCol, deltaCol); 
                    // OPTIMIZER TREBA DA ima referencu na layer i da sam uzima sta mu treba. Ili da sam optimizer kod sebe skladisti stukrure podataka potrebne za konkretan algoritam
                    // napravi interfejs optimizer, konkretna implementacija d aima referencu na lejer, i da sadrzi sve strukture podataka koje su potrebne za konkretan optimizer
                    // idealno da radi nad matricama a ne nad pojedinacnim vrednostima. Da li se onda moze generalizovati?
                    // treba da radi nad weights i biasom
                    // svi optimizeri mogu da nasledjuju jedan osnovni koji vrti petlju i da imaju templejt metod koji predstavlja konkretnu implementaciju
                    
                // mozda je najbolje da optimizer ima refrencu na layer i da sam uzme sta mu sve treba od polja. 
                // layar onda mora da expousuje sve sto treba sto je sasvim u redu i vec radi         
                
                //    deltaWeight = opt.optimize(grad); // array of floats?
                                                  // ako su razliciti parametri mora optimizer da ih povuce iz layera
                 // kako da za sve varijante ispod imam samo jedan poziv      
                 // prosledi kao parametar optimizer type? uslovnu logiku prebaci u optimizer? razne vrste optimizera mogu interno da skladiste specificne strukture
                 // idealno bi bilo ak obi imao instance razlicitih optimizera
                 // za prosledjivanja parametara opcija 1 je da se prosledi vrednost a opcija 2 indeks
                 // ako se prosledjuje indeksi onda bi morali razliciti optimizeri za razlcite layere?
                 // preostalo pitanje je kako da azuriram strukture tipa  prevGradSqrSum prevDeltaWeightSqrSum koje nisu deo apdejta
                 //                 - pozovi metodu save structures na kraju koja ce da prekopira ceo tensor koji treba
                 // kako istoj apstrakciji prosledjivati razlicite liste parametara, u zavisnosti od implementacije?
                 // jedno resenje da bude varijabilna lista parametara? float ... izgleda kao jedino moguce....
                 // jedino moguce je da sam optimizer uzima sta mu treba i to da radi za ceo tensor - u tom slucaju bi mu trebalo prosledjivati index pociciju kao niz?
// !!!           // trenutno jedino resenje je da mu se prosledjuje  vrta optimizacije i da swith logika bude u optimizeru tako se nece duplirti kod u lejerima
//              cilj: umesto switcha jedan poziv metode
                // optim.calculateParamDelta(grad, inCol, deltaCol);
                 
                /*
                switch (optimizer) {
                        case SGD:
                            //deltaWeight = Optimizers.sgd(learningRate, grad);
                            deltaWeight = optim.calculateParamDelta(grad);
                            break;
                        case MOMENTUM:
                             deltaWeight = optim.calculateParamDelta(grad, inCol, deltaCol); // int[] param hold the Tensor indexes . Tensor index je tuple. E sad da li ce da radi inlajning?
                           // deltaWeight = Optimizers.momentum(learningRate, grad, momentum, prevDeltaWeights.get(inCol, deltaCol));
                            break;
                        case ADAGRAD:
                            prevGradSqrSum.add(inCol, deltaCol, grad * grad); // da li ovo treba resetovati na nulu nekad? da lije samo za mini batch mode? treba po meni uvek
                            deltaWeight = Optimizers.adaGrad(learningRate, grad, prevGradSqrSum.get(inCol, deltaCol));  // prevGradSqrSum takodje budziti negde i sabirati odjenom
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
*/
                    deltaWeights.add(inCol, deltaCol, deltaWeight); // add zbog batch moda!
                }
                
                // todo: kako i bias uklopiti u isti sablon? posebna metoda ili bias prebaciti u tensor
                //float deltaBias = 0;
                final float deltaBias = optim.calculateBiasDelta(deltas.get(deltaCol), deltaCol); 
            /*    switch (optimizer) {
                    case SGD:
                        //deltaBias = Optimizers.sgd(learningRate, deltas.get(deltaCol));
                        deltaBias = optim.calculateWeightDelta(deltas.get(deltaCol));
                        break;
                    case MOMENTUM:
                        //deltaBias = opt2.calculateParamDelta(deltas.get(deltaCol));
                        deltaBias = Optimizers.momentum(learningRate, deltas.get(deltaCol), momentum, prevDeltaBiases[deltaCol]); //  FIX: ovo ovde treba srediti!
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
                }*/

                deltaBiases[deltaCol] += deltaBias;
            }
        } else if ((prevLayer instanceof InputLayer) // CHECK: sta ako je input layer 1d da li ovo radi akko treba?
                || (prevLayer instanceof ConvolutionalLayer)
                || (prevLayer instanceof MaxPoolingLayer)) {

            for (int deltaCol = 0; deltaCol < deltas.getCols(); deltaCol++) { // for all neurons/deltas in this layer
                for (int inDepth = 0; inDepth < inputs.getDepth(); inDepth++) { // iterate all inputs from previous layer
                    for (int inRow = 0; inRow < inputs.getRows(); inRow++) {
                        for (int inCol = 0; inCol < inputs.getCols(); inCol++) {
                            final float grad = deltas.get(deltaCol) * inputs.get(inRow, inCol, inDepth);  // da li je ovde greska treba ih sumitrati sve tri po dubini  // da li ove ulaze treba sabirati??? jer jedna celija ima ulaze iz tri prethodna kanala?
                            gradients.set(inRow, inCol, inDepth, deltaCol, grad);   // da li ovo radi kada je fc sa obicnim input lyerom -  proveri??

                            final float deltaWeight = optim.calculateWeightDelta(grad, inCol, inRow, inDepth, deltaCol); 
                            
                        /*    float deltaWeight = 0;
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
                            }*/

                            deltaWeights.add(inCol, inRow, inDepth, deltaCol, deltaWeight);
                        }
                    }
                }

                /*
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
*/
                float deltaBias = optim.calculateBiasDelta(deltas.get(deltaCol), deltaCol);                
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
