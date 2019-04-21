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
import deepnetts.net.train.opt.Optimizers;
import deepnetts.util.DeepNettsException;
import deepnetts.util.WeightsInit;
import deepnetts.util.Tensor;
import java.util.logging.Logger;
import deepnetts.net.layers.activation.ActivationFunction;
import deepnetts.util.DeepNettsThreadPool;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CyclicBarrier;

/**
 * Convolutional layer performs image convolution operation on outputs of a
 * previous layer using filters. This filtering operation is similar like
 * applying image filters in photoshop, but this filters can also be trained to
 * learn image features of interest.
 *
 * Layer include parameters: filter's width, heigh Number of filters / depth
 * Step when applying filters : stride Padding, which is an image border to keep
 * the size of image and avoid information loss padding Stride defaults to 1
 *
 * @author Zoran Sevarac
 */
public final class ConvolutionalLayer extends AbstractLayer {

    Tensor[] filters;           // each filter corresponds to a single channel. Each filter can be 3D, where 3rd dimension coreesponds to depth in previous layer. TODO: the depth pf th efilter should be tunable
    Tensor[] deltaWeights;      //ovo za sada ovako dok proradi. Posle mozda ubaciti jos jednu dimenziju u matricu - niz za kanale. i treba da overriduje polje jer su weights u filterima za sve prethdne kanale
    Tensor[] prevDeltaWeights;  // delta weights from previous iteration (used for momentum)
    Tensor[] prevGradSums;  // delta weights from previous iteration (used for momentum)

    /**
     * Convolutional filter width
     */
    int filterWidth,
            /**
             * Filter dimensions, filter depth is equal to number of depth /
             * depth of
             */
            filterHeight,
            /**
             * Filter dimensions, filter channels is equal to number of channels
             * / channels of
             */
            /**
             * Filter dimensions, filter depth is equal to number of depth /
             * depth of
             */
            filterDepth; // da li je filter istih dimenzija za sve feature mape?

    /**
     * Convolution step, 1 by default. Number of steps convolutional filter is
     * moved during convolution. Commonly used values 1, 2, rarely 3
     */
    int stride = 1;

    /**
     * Border padding filled with zeros (0, 1 or 2) Usually half of the filter
     * size
     */
    int padding = 0;

    int fCenterX; //  padding = (kernel-1)/2
    int fCenterY;

    int[][][][] maxIdx;

    private transient List<Callable<Void>> forwardTasks;
    private transient List<Callable<Void>> backwardFromPoolingTasks;
    private transient List<Callable<Void>> backwardFromConvolutionalTasks;
    private transient List<Callable<Void>> backwardFromFullyConnectedTasks;
    private boolean multithreaded = false;

    private static final Logger LOG = Logger.getLogger(DeepNetts.class.getName());

    /**
     * Create a new instance of convolutional layer with specified filter.
     * dimensions, default padding (filter-1)/2, default stride stride value 1,
     * and specified number of channels.
     *
     * @param filterWidth
     * @param filterHeight
     * @param channels
     */
    public ConvolutionalLayer(int filterWidth, int filterHeight, int channels) {
        // sve mora da bude pozitivno. filteri motaju da budu  neparni - validacija

        this.filterWidth = filterWidth;
        this.filterHeight = filterHeight;
        this.depth = channels; // ovo je isto kao i depth, broj feature mapa
        this.stride = 1;
        this.activationType = ActivationType.TANH; // use relu as default?
        this.activation = ActivationFunction.create(activationType);
    }

    public ConvolutionalLayer(int filterWidth, int filterHeight, int channels, ActivationType activationType) {
        this.filterWidth = filterWidth;
        this.filterHeight = filterHeight;
        this.depth = channels;
        this.stride = 1;
        this.activationType = activationType;
        this.activation = ActivationFunction.create(activationType);
    }

    // todo: move stride after channels parameter to be consistent
    public ConvolutionalLayer(int filterWidth, int filterHeight, int channels, int stride, ActivationType activationType) {
        this.filterWidth = filterWidth;
        this.filterHeight = filterHeight;
        this.depth = channels;
        this.stride = stride;
        this.activationType = activationType;
        this.activation = ActivationFunction.create(activationType);
    }

    /**
     * Init dimensions, create matrices, filters, weights, biases and all
     * internal structures etc.
     *
     * Assumes that prevLayer is set in network builder
     */
    @Override
    public void init() {
        // prev layer can only be input, max pooling or convolutional
        if (!(prevLayer instanceof InputLayer || prevLayer instanceof ConvolutionalLayer || prevLayer instanceof MaxPoolingLayer)) {
            throw new DeepNettsException("Illegal architecture: convolutional layer can be used only after input, convolutional or maxpooling layer");
        }

        inputs = prevLayer.outputs;

        width = (prevLayer.getWidth()) / stride;
        height = (prevLayer.getHeight()) / stride;
        // depth is set in constructor

        fCenterX = (filterWidth - 1) / 2; //  padding = filter /2
        fCenterY = (filterHeight - 1) / 2;

        // init output cells, deltas and derivative buffer
        outputs = new Tensor(height, width, depth);
        deltas = new Tensor(height, width, depth);
//        derivatives = new Tensor(height, width, depth);

        // init filters(weights) - broj filtera je isti kao i broj kanala/dubina prethodnog lejera
        filterDepth = prevLayer.getDepth();
        filters = new Tensor[depth]; // depth of the filters should be configurable - its a hyper param!
        deltaWeights = new Tensor[depth];
        prevDeltaWeights = new Tensor[depth];
        prevGradSums = new Tensor[depth];

        int inputCount = (filterWidth * filterHeight + 1) * filterDepth;

        // kreiraj pojedinacne filtere ovde
        for (int ch = 0; ch < filters.length; ch++) {
            filters[ch] = new Tensor(filterHeight, filterWidth, filterDepth);
            WeightsInit.uniform(filters[ch].getValues(), inputCount); // vidi koji algoritam da koristim ovde: uzmi u obzir broj kanala i dimenzije filtera pa da im suma bude 1 ili sl. neka gausova distribucija...

            deltaWeights[ch] = new Tensor(filterHeight, filterWidth, filterDepth);
            prevDeltaWeights[ch] = new Tensor(filterHeight, filterWidth, filterDepth);
            prevGradSums[ch] = new Tensor(filterHeight, filterWidth, filterDepth);
        }

        // and biases               // svaki kanal ima svoj filter i svoj bias - sta ako prethodni sloj ima vise biasa? mislim da bi tada svaki filter trebalo da ima svoj bias ovo bi znaci trebalo da bude 2D biases[depth][prevLayerDepth]
        biases = new float[depth]; // biasa ima koliko ima kanala - svaki FM ima jedan bias - mozda i vise... ili ce svaki filter imati svoj bias? - tako bi trebalo...
        // svaki kanal u ovom sloju ima filtera onoliko kolik ima kanala u prethodnom sloji. i Svi ti filteri imaju jedan bias
        deltaBiases = new float[depth];
        prevDeltaBiases = new float[depth];
        prevBiasSqrSum = new Tensor(depth);
        WeightsInit.randomize(biases);        // sometimes the init to 0 fir relu 0.1
        
        if (DeepNettsThreadPool.getInstance().getThreadCount() > 1) {
            multithreaded = true;

            forwardTasks = new ArrayList<>();
            backwardFromPoolingTasks = new ArrayList<>();
            backwardFromFullyConnectedTasks = new ArrayList<>();
            backwardFromConvolutionalTasks = new ArrayList<>();
            float channelsPerThread = depth / (float) DeepNettsThreadPool.getInstance().getThreadCount();
            CyclicBarrier fcb = new CyclicBarrier(DeepNettsThreadPool.getInstance().getThreadCount());    // all threads share the same cyclic barrier
            CyclicBarrier bcb = new CyclicBarrier(DeepNettsThreadPool.getInstance().getThreadCount());    // all threads share the same cyclic barrier
            for (int i = 0; i < DeepNettsThreadPool.getInstance().getThreadCount(); i++) {
                ForwardCallable ftask = new ForwardCallable((int) channelsPerThread * i, (int) channelsPerThread * (i + 1), fcb);
                forwardTasks.add(ftask);

                if (nextLayer instanceof MaxPoolingLayer) {
                    BackwardFromPoolingCallable btask = new BackwardFromPoolingCallable((int) channelsPerThread * i, (int) channelsPerThread * (i + 1), bcb);
                    backwardFromPoolingTasks.add(btask);
                } else if (nextLayer instanceof FullyConnectedLayer) {
                    BackwardFromFullyConnectedCallable bfctask = new BackwardFromFullyConnectedCallable((int) channelsPerThread * i, (int) channelsPerThread * (i + 1), bcb);
                    backwardFromFullyConnectedTasks.add(bfctask);
                } else if (nextLayer instanceof ConvolutionalLayer) {
                    BackwardFromConvolutionalCallable bctask = new BackwardFromConvolutionalCallable((int) channelsPerThread * i, (int) channelsPerThread * (i + 1), bcb);
                    backwardFromConvolutionalTasks.add(bctask);
                }
                
                
            }
        }

    }

    /**
     * Forward pass for convolutional layer. Performs convolution operation on
     * output from previous layer using filters in this layer, on all channels.
     * Each channel from prev layer has its own filter (3D filter), and every
     * channel in this layer has its 3D filter used to scan all channels in prev
     * layer.
     *
     * Previous layers can be: Input, MaxPooling or Convolutional.
     *
     * For more about convolution see
     * http://www.songho.ca/dsp/convolution/convolution.html
     */
    @Override
    public void forward() {
        if (!multithreaded) {
            for (int ch = 0; ch < this.depth; ch++) {
                forwardForChannel(ch);
            }
        } else {
            try {
                DeepNettsThreadPool.getInstance().run(forwardTasks);
            } catch (InterruptedException ex) {
                LOG.warning(ex.getMessage()); // throw excepti0on here!!!
                //Logger.getLogger(ConvolutionalLayer.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }

    /**
     * Performs forward pass calculation for specified channel.
     * 
     * @param ch channel to calculate
     */
    private void forwardForChannel(int ch) {
        int outRow = 0, outCol = 0; // reset indexes for current output's row and col

        for (int inRow = 0; inRow < inputs.getRows(); inRow += stride) { // iterate all input rows
            outCol = 0; // every time when input goes in next row, output does too, so reset column idx

            for (int inCol = 0; inCol < inputs.getCols(); inCol += stride) { // iterate all input cols
                outputs.set(outRow, outCol, ch, biases[ch]); // sum will be added to bias - I can set entire matrix to bias initial values  above

                // apply filter to all channnels in previous layer
                for (int fz = 0; fz < filterDepth; fz++) { // iterate filter by depth - number of channels in previous layer
                    for (int fr = 0; fr < filterHeight; fr++) { // iterate filter by height/rows
                        for (int fc = 0; fc < filterWidth; fc++) { // iterate filter by width / columns
                            final int cr = inRow + (fr - fCenterY); // convolved row idx
                            final int cc = inCol + (fc - fCenterX); // convolved col idx

                            // skip input indexes which are out of bounds
                            if (cr < 0 || cr >= inputs.getRows() || cc < 0 || cc >= inputs.getCols()) {
                                continue;
                            }

                            final float out = inputs.get(cr, cc, fz) * filters[ch].get(fr, fc, fz); // output of a single conv filter cell
                            outputs.add(outRow, outCol, ch, out); // accumulate filters from all channels
                        }
                    }
                }

                // apply activation function
                final float out = activation.getValue(outputs.get(outRow, outCol, ch));
                outputs.set(outRow, outCol, ch, out);
                outCol++; // move to next col in out layer after each filter position
            }
            outRow++; // every time input goes to next row (inR), output does too
        }
    }

    /**
     * Backward pass for convolutional layer tweaks the weights in filters.
     *
     * Next layer can be: FC, MaxPooling, Conv, (output same as FC), 1D or 3D
     * Prev layer can: Input, pool, conv, all 2D or 3D - all can be as
     * generalized 3D
     *
     * U 2 koraka:
     *
     * 1. povuci delte iz sledeceg lejera, i izracunaj tezinsku sumu delta za
     * sve neurone/outpute u ovom sloju 2. izracunaj promene tezina za sve veze
     * iz prethodnog lejera za svaki neuron/output u ovom sloju
     */
    @Override
    public void backward() {
        if (nextLayer instanceof FullyConnectedLayer) {
            backwardFromFullyConnected();
        }

        if (nextLayer instanceof MaxPoolingLayer) {
            backwardFromMaxPooling();
        }

        if (nextLayer instanceof ConvolutionalLayer) {
            // NOTE: average weights for the filter and biases? - koliko ima pozicija i kanala??? negde sam procitao da treba da se sabiraju...
            backwardFromConvolutional();
        }
    }

    /**
     * Backward pass when next layer is fully connected.
     *
     * Calculates deltas for this layer
     *
     */
    private void backwardFromFullyConnected() {
        deltas.fill(0); // reset deltas for all units

        // CHECK: posle ovog backwarda ima dosta nula u deltas sto je cudno
        if (!multithreaded) {
            for (int ch = 0; ch < this.depth; ch++) { // iteriraj sve kanale/feature mape u ovom lejeru - razbij kanale sa fork join frejmvorkom
                backwardFromFullyConnectedForChannel(ch);
            }
        } else {
            try {
                DeepNettsThreadPool.getInstance().run(backwardFromFullyConnectedTasks);
            } catch (InterruptedException ex) {
                LOG.warning(ex.getMessage());
            }
        }           
    }
    

    private void backwardFromFullyConnectedForChannel(int ch) {
        // 1. Propagate deltas from the next FC layer
        for (int row = 0; row < this.height; row++) {
            for (int col = 0; col < this.width; col++) {
                final float actDerivative = activation.getPrime(outputs.get(row, col, ch)); // dy/ds
                for (int ndC = 0; ndC < nextLayer.deltas.getCols(); ndC++) {
                    final float delta = nextLayer.deltas.get(ndC) * nextLayer.weights.get(col, row, ch, ndC) * actDerivative;
                    deltas.add(row, col, ch, delta);
                }
            }
        } // end back propagate deltas

        // 2. calculate weight changes for this layer - ako je batch mode ona ne racunati nego samo akumulirati delte
        calculateDeltaWeightsForChannel(ch); 
    }  

    private void backwardFromMaxPooling() {
        final MaxPoolingLayer nextPoolLayer = (MaxPoolingLayer) nextLayer;
        maxIdx = nextPoolLayer.maxIdx; // uzmi index neurona koji je poslao max output na tekucu poziciju filtera
        // proveri da li ovo da radim za svaki poseban kanal
        deltas.fill(0); // reset all deltas

        
        if (!multithreaded) {
            for (int ch = 0; ch < this.depth; ch++) {
                backwardFromMaxPoolingForChannel(ch);
            }
        } else {
            try {
                DeepNettsThreadPool.getInstance().run(backwardFromPoolingTasks);
            } catch (InterruptedException ex) {
                LOG.warning(ex.getMessage());
            }
        }        
    }


    /**
     * Performs backward pass from maxpooling layer for specified channel in this layer.
     * 
     * @param ch 
     */
    private void backwardFromMaxPoolingForChannel(int ch) {
        // 1. Propagate deltas from next layer for max outputs from this layer
        for (int dr = 0; dr < nextLayer.deltas.getRows(); dr++) { // sledeci lejer delte po visini
            for (int dc = 0; dc < nextLayer.deltas.getCols(); dc++) { // sledeci lejer delte po sirini

                final float nextLayerDelta = nextLayer.deltas.get(dr, dc, ch); // uzmi deltu iz sledeceg sloja za tekuci neuron sledeceg sloja
                final int maxR = maxIdx[ch][dr][dc][0];
                final int maxC = maxIdx[ch][dr][dc][1];

                final float derivative = activation.getPrime(outputs.get(maxR, maxC, ch));
                deltas.set(maxR, maxC, ch, nextLayerDelta * derivative);
            }
        } // end propagate deltas

        calculateDeltaWeightsForChannel(ch);
    }

    // TODO: ovajnije paralelizovan!!!
    // peretpostavlja se da je sledeci sloj konvolucioni - ovo nije paralelizovano!!!
    private void backwardFromConvolutional() {
        deltas.fill(0); // reset all deltas in this layer (deltas are 3D)
        
//        for (int ch = 0; ch < this.depth; ch++) {
//            backwardFromConvolutionalForChannel(ch);
//        }
        
        if (!multithreaded) {
            for (int ch = 0; ch < this.depth; ch++) {
                backwardFromConvolutionalForChannel(ch);
            }
        } else {
            try {
                DeepNettsThreadPool.getInstance().run(backwardFromConvolutionalTasks);
            } catch (InterruptedException ex) {
                LOG.warning(ex.getMessage());
            }
        }              
        
//        for (int ch = 0; ch < this.depth; ch++) {
//            // 2. calculate delta weights for this layer (this is same for all types of next layers)
//            calculateDeltaWeightsForChannel(ch); // ovo je sad prebaceno u backwardFromConvolutionalForChannel radi identicno proverio sam
//        }
    }
    
    private void backwardFromConvolutionalForChannel(int fz) {
        ConvolutionalLayer nextConvLayer = (ConvolutionalLayer) nextLayer;  // ovo u atribut i init metodu!!!
        int filterCenterX = (nextConvLayer.filterWidth - 1) / 2;
        int filterCenterY = (nextConvLayer.filterHeight - 1) / 2;
        
        // 1. Propagate deltas from next conv layer for max outputs from this layer
        for (int ndZ = 0; ndZ < nextLayer.deltas.getDepth(); ndZ++) {
            for (int ndRow = 0; ndRow < nextLayer.deltas.getRows(); ndRow++) { // iteriraj delte sledeceg lejera po visini
                for (int ndCol = 0; ndCol < nextLayer.deltas.getCols(); ndCol++) { // iteriraj delte sledeceg lejera po sirini
                    final float nextLayerDelta = nextLayer.deltas.get(ndRow, ndCol, ndZ); // uzmi deltu iz sledeceg sloja za tekuci neuron (dx, dy, dz) sledeceg sloja, da li treba d ase sabiraju?

                   // for (int fz = 0; fz < nextConvLayer.filterDepth; fz++) {    //!! pa da li je ovo isto kao i prvi loop - da li je dupliranje???!!! kao ovaj prvi ch!!! mislim da tu ima nepotrebnog preklapanja /dupliranja. kada filter ne bude iseo preko svih u prethodnom lejeru onda nece biti preklapanja
                        for (int fr = 0; fr < nextConvLayer.filterHeight; fr++) {
                            for (int fc = 0; fc < nextConvLayer.filterWidth; fc++) {
                                final int row = ndRow * nextConvLayer.stride + (fr - filterCenterY); // proveri da li ovo dobro racuna
                                final int col = ndCol * nextConvLayer.stride + (fc - filterCenterX);

                                if (row < 0 || row >= outputs.getRows() || col < 0 || col >= outputs.getCols()) {
                                    continue;
                                }

                                // Mnoziti van petlje nakon zavrsetka sabiranja. Izracunati izvode u jednom prolazu, pa onda mnoziti  ane za svaku celiju.
                                final float derivative = activation.getPrime(outputs.get(row, col, fz)); // ne pozivati ovu funkciju ovde u petlji  vec optimizovati nekako. Mnoziti van petlje nakon zavrsetka sabiranja. Izracunati izvode u jednom prolazu, pa onda mnoziti  ane za svaku celiju.
                                //   ... ovde treba razjasniti kako se mnozi sa weightsomm? da li ih treba sabirati
                                deltas.add(row, col, fz, nextLayerDelta * nextConvLayer.filters[ndZ].get(fr, fc, fz) * derivative);
                            }
                        }
                   // }
                }
            }
        }    
        
        calculateDeltaWeightsForChannel(fz);
    }    

    /**
     * Calculates delta weights for the specified channel ch in this
     * convolutional layer.
     *
     * @param ch channel/depth index
     */
    private void calculateDeltaWeightsForChannel(int ch) {
        if (!batchMode) {
            deltaWeights[ch].fill(0); // reset all delta weights for the current channel - these are 4d matrices
            deltaBiases[ch] = 0;
        }

        final float divisor = width * height; //  tezine u filteru racunari kao prosek sa svih pozicija u feature mapi

        // assumes that deltas from the next layer are allready propagated
        // 2. calculate weight changes in filters
        for (int deltaRow = 0; deltaRow < deltas.getRows(); deltaRow++) {
            for (int deltaCol = 0; deltaCol < deltas.getCols(); deltaCol++) {
                // iterate all weights in filter for filter depth
                for (int fz = 0; fz < filterDepth; fz++) { // filter depth, input channel
                    for (int fr = 0; fr < filterHeight; fr++) {
                        for (int fc = 0; fc < filterWidth; fc++) {

                            final int inRow = deltaRow * stride + fr - fCenterY;
                            final int inCol = deltaCol * stride + fc - fCenterX;

                            if (inRow < 0 || inRow >= inputs.getRows() || inCol < 0 || inCol >= inputs.getCols()) {
                                continue;
                            }
                            // da li ovde ispod nedostaje kanal cs? ne kanal je fz - to je dubina filtera i to ide kroz svekanale
                            final float input = inputs.get(inRow, inCol, fz); // get input for this output and weight; padding?  da li ovde imam kanal?
                            final float grad = deltas.get(deltaRow, deltaCol, ch) * input;

                            float deltaWeight = 0;  // DODO: put optimizaer instances here!!!
                            switch (optimizerType) {
                                case SGD:
                                    deltaWeight = Optimizers.sgd(learningRate, grad);
                                    break;
                                case MOMENTUM:
                                    deltaWeight = Optimizers.momentum(learningRate, grad, momentum, prevDeltaWeights[ch].get(fr, fc, fz)); // ovaj sa momentumom odmah izleti u NaN
                                    break;
                                case ADAGRAD:
                                    prevGradSums[ch].add(fr, fc, fz, grad * grad);
                                    deltaWeight = Optimizers.adaGrad(learningRate, grad, prevGradSums[ch].get(fr, fc, fz));
                                    break;
                            }
                            deltaWeight /= divisor;  // da li je ovo matematicki tacno? momentum baca nana ako ovog nema
                            deltaWeights[ch].add(fr, fc, fz, deltaWeight);
                        }
                    }
                }
                float deltaBias = 0;
                switch (optimizerType) {
                    case SGD:
                        deltaBias = Optimizers.sgd(learningRate, deltas.get(deltaRow, deltaCol, ch));
                        break;
                    case MOMENTUM:
//                       deltaBias = Optimizers.sgd(learningRate, deltas.get(deltaRow, deltaCol, ch));
                        deltaBias = Optimizers.momentum(learningRate, deltas.get(deltaRow, deltaCol, ch), momentum, prevDeltaBiases[ch]);
                        break;
                    case ADAGRAD:
                        deltaBias = Optimizers.adaGrad(learningRate, deltas.get(deltaRow, deltaCol, ch), prevBiasSqrSum.get(ch));
                        prevBiasSqrSum.add(ch, deltas.get(deltaRow, deltaCol, ch) * deltas.get(deltaRow, deltaCol, ch));
                        break;
                }
                deltaBiases[ch] /= divisor;
                deltaBiases[ch] += deltaBias;
            }
        } // end calculate weight changes in filter   }
    }

    /**
     * Apply weight changes calculated in backward pass
     */
    @Override
    public void applyWeightChanges() {

        if (batchMode) {    // divide biases with batch samples if it is in batch mode
            Tensor.div(deltaBiases, batchSize);
        }

        Tensor.copy(deltaBiases, prevDeltaBiases);  // save this for momentum

        for (int ch = 0; ch < depth; ch++) {
            if (batchMode) { // podeli Delta weights sa brojem uzoraka odnosno backward passova
                deltaWeights[ch].div(batchSize);
            }

            Tensor.copy(deltaWeights[ch], prevDeltaWeights[ch]); // da li ovo treba pre ilo posle prethodnog kad aje u batch mode-u?, ok je d abude posle jer se prienjuje pojedinacno

            filters[ch].add(deltaWeights[ch]);
            biases[ch] += deltaBiases[ch];

            if (batchMode) {    // reset delta weights for next batch
                deltaWeights[ch].fill(0);
            }
        }

        if (batchMode) { // reset delta biases for next batch
            Tensor.fill(deltaBiases, 0);
        }

    }

    public Tensor[] getFilters() {
        return filters;
    }

    public void setFilters(Tensor[] filters) {
        this.filters = filters;
    }

    public void setFilters(String filtersStr) {

        String[] strVals = filtersStr.split(";"); // ; is hardcoded filter separator see FileIO // also can be splited at "
        int filterSize = filterWidth * filterHeight * filterDepth;

        for (int i = 0; i < filters.length; i++) {
            float[] filterValues = new float[filterSize];
            String[] vals = strVals[i].split(",");
            for (int k = 0; k < filterSize; k++) {
                filterValues[k] = Float.parseFloat(vals[k]);
            }

            filters[i].setValues(filterValues); // ovde je tensor 5x5x3 a imamomo samo 25 vrednosti
        }
    }

    public int getFilterWidth() {
        return filterWidth;
    }

    public int getFilterHeight() {
        return filterHeight;
    }

    public int getFilterDepth() {
        return filterDepth;
    }

    public int getStride() {
        return stride;
    }

    public Tensor[] getFilterDeltaWeights() {
        return deltaWeights;
    }

    /**
     * Task to parallelize channel calculation using executors thread pool.
     * Also provide synchronization for all threads to wait before next layer.
     * using  CyclicBarrier.
     */
    private class ForwardCallable implements Callable<Void> {

        private final int fromCh, toCh;
        private final CyclicBarrier cb;

        public ForwardCallable(int fromCh, int toCh, CyclicBarrier cb) {
            this.fromCh = fromCh;
            this.toCh = toCh;
            this.cb = cb;
        }
        
        @Override
        public Void call() throws Exception {

            for (int ch = fromCh; ch < toCh; ch++) {
                forwardForChannel(ch);
            }

            cb.await();
            return null;
        }
    }

    private class BackwardFromConvolutionalCallable implements Callable<Void> {

        private final int fromCh, toCh;
        private final CyclicBarrier cb;

        public BackwardFromConvolutionalCallable(int fromCh, int toCh, CyclicBarrier cb) {
            this.fromCh = fromCh;
            this.toCh = toCh;
            this.cb = cb;
        }

        @Override
        public Void call() throws Exception {

            for (int ch = fromCh; ch < toCh; ch++) {
                backwardFromConvolutionalForChannel(ch);   // ovde sad zavisi koji je sledeci pa njegovu metodu poziva. Da li da imam 3 threada ili da mi proledjujem referencu nametodu?
            }

            cb.await();
            return null;
        }
    }    
    
    private class BackwardFromPoolingCallable implements Callable<Void> {

        private final int fromCh, toCh;
        private final CyclicBarrier cb;

        public BackwardFromPoolingCallable(int fromCh, int toCh, CyclicBarrier cb) {
            this.fromCh = fromCh;
            this.toCh = toCh;
            this.cb = cb;
        }

        @Override
        public Void call() throws Exception {

            for (int ch = fromCh; ch < toCh; ch++) {
                backwardFromMaxPoolingForChannel(ch);   // ovde sad zavisi koji je sledeci pa njegovu metodu poziva. Da li da imam 3 threada ili da mi proledjujem referencu nametodu?
            }

            cb.await();
            return null;
        }
    }
    
    private class BackwardFromFullyConnectedCallable implements Callable<Void> {

        private final int fromCh, toCh;
        private final CyclicBarrier cb;

        public BackwardFromFullyConnectedCallable(int fromCh, int toCh, CyclicBarrier cb) {
            this.fromCh = fromCh;
            this.toCh = toCh;
            this.cb = cb;
        }

        @Override
        public Void call() throws Exception {

            for (int ch = fromCh; ch < toCh; ch++) {
                backwardFromFullyConnectedForChannel(ch);
            }

            cb.await();
            return null;
        }
    }    

    @Override
    public String toString() {
        return "Convolutional Layer { filter width:" + filterWidth + ", filter height: " + filterHeight + ", channels: " + depth + ", stride: " + stride + ", activation: " + activationType.name() + "}";
    }

}
