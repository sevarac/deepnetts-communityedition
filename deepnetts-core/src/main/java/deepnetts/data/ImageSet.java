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
 * this program. If not, see <https://www.gnu.org/licenses/>.
 */
package deepnetts.data;

import deepnetts.core.DeepNetts;
import deepnetts.util.DeepNettsException;
import deepnetts.util.Tensor;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import javax.imageio.ImageIO;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Represents data set with images
 *
 * @author zoran
 */
public class ImageSet extends BasicDataSet<ExampleImage> {

    private final int imageWidth;
    private final int imageHeight;
    private Tensor mean;

    private static final Logger LOGGER = LogManager.getLogger(DeepNetts.class.getName());

    // ovi ne mogu svi da budu u memoriji odjednom...
    // osmisliti i neki protocni / buffered data set, koji ucitava jedan batch
    public ImageSet(int imageWidth, int imageHeight) {
        super();
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;

        // labels = new ArrayList();
    }

//    public ImageSet(int capacity) {
//        super();
//        labels = new ArrayList();
//    }
    /**
     * Adds image to this image set.
     *
     * @param exImage
     * @throws DeepNettsException if image is empty or has wrong dimensions.
     */
    public void add(ExampleImage exImage) throws DeepNettsException {
        if (exImage == null) {
            throw new DeepNettsException("Example image cannot be null!");
        }
        items.add(exImage);

//        if ((exImage.getWidth() == imageWidth) && (exImage.getHeight() == imageHeight)) {
//            items.add(exImage);
//        } else {
//            throw new DeepNettsException("Wrong image dimensions for this data set. All images should be "+imageWidth + "x"  + imageHeight);
//        }
    }

    /**
     * Loads example images with labels from specified file.
     *
     * TODO: First load entire image index, then load and preprocess image in
     * multithreaded way TODO2: load images in batches
     *
     * @param imageIdxFile Plain text file that contains space delimited image
     * file paths and labels
     * @param absPaths True if file contains absolute paths for images, false
     * otherwise
     * @throws java.io.FileNotFoundException if imageIdxFile was not found
     */
    public void loadImages(File imageIdxFile, boolean absPaths) throws FileNotFoundException {
        String parentPath = "";
        if (absPaths == false) {
            parentPath = imageIdxFile.getPath().substring(0, imageIdxFile.getPath().lastIndexOf(File.separator));
        }

        String imgFileName = null;
        String label = null;
        final String[] fColumnNames = columnNames;

        ExecutorService es = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors() - 1);

        // TODO: napravi ovo asinhrono da ucitava i preprocesira u posebnim threadovima, u perspektivi u batchovima, ne sve odjendnom
        // ucitaj prvo indeks slika a onda ucitavanje i preprocrsiranje slika parelelizuj da jedan thread radi ucitavanje a drugi preprocsiranje onoga sto je ucitano
        try (BufferedReader br = new BufferedReader(new FileReader(imageIdxFile))) {
            String line = null;
            List<Future<?>> results = new LinkedList<>();
            // we can also catch and log FileNotFoundException, IOException in this loop
            while ((line = br.readLine()) != null) {
                if (line.isEmpty()) {
                    continue;
                }
                String[] str = line.split(" "); // parse file and class label from current line - sta ako naziv fajla sadrzi space? - to ne sme ili detektuj nekako sa lastIndex

                imgFileName = str[0];
                if (!absPaths) {
                    imgFileName = parentPath + File.separator + imgFileName;
                }
                if (str.length == 2) {
                    label = str[1];
                } else if (str.length == 1) {
                    // todo: extract label from parent folder - check this
                    final int labelEndIdx = imgFileName.lastIndexOf(File.separator);
                    final int labelStartIdx = imgFileName.lastIndexOf(File.separator, labelEndIdx) + 1;
                    label = imgFileName.substring(labelStartIdx, labelEndIdx);
                    //parentPath
                }

                // todo: ucitavaj slike u ovom a preprocesiraj u psebnom threadu, najbolje submituj preprocesiranje na neki thread pool
                final BufferedImage image = ImageIO.read(new File(imgFileName));
                final String flabel = label;
                final ExampleImage exImg = new ExampleImage(image, flabel);
                exImg.setTargetOutput(oneHotEncode(flabel, fColumnNames));
                add(exImg);

//                Future<?> result = es.submit(() -> {
//                    try {
//                        final ExampleImage exImg = new ExampleImage(image, flabel);
//                        exImg.setTargetOutput(oneHotEncode(flabel, fColumnNames));
//                        add(exImg);
//                        return true;
//                    } catch (IOException ex) {
//                        java.util.logging.Logger.getLogger(ImageSet.class.getName()).log(Level.SEVERE, null, ex);
//                    }
//                    return false;
//                    // make sure all images are the same size
////                if ((exImg.getWidth() != imageWidth) || (exImg.getHeight() != imageHeight)) throw new DeepNettsException("Bad image size for "+exImg.getFile().getName());
//                });
             //   results.add(result);
            }

//            results.forEach((f) -> {
//                try {
//                    f.get();
//                } catch (InterruptedException ex) {
//                    java.util.logging.Logger.getLogger(ImageSet.class.getName()).log(Level.SEVERE, null, ex);
//                } catch (ExecutionException ex) {
//                    java.util.logging.Logger.getLogger(ImageSet.class.getName()).log(Level.SEVERE, null, ex);
//                }
//            });
//            es.shutdown();

            if (isEmpty()) {
                throw new DeepNettsException("Zero images loaded!");
            }

            LOGGER.info("Loaded " + size() + " images");

        } catch (FileNotFoundException ex) {
            LOGGER.error(ex);
            throw new DeepNettsException("Could not find image file: " + imgFileName, ex);
        } catch (IOException ex) {
            LOGGER.error(ex);
            throw new DeepNettsException("Error loading image file: " + imgFileName, ex);
        }

    }

    public void loadImages(String imageIdxFile, boolean absPaths) throws FileNotFoundException {
        loadImages(new File(imageIdxFile), absPaths);
    }

    /**
     * Loads example images and corresponding labels from specified file.
     *
     * @param imageIdxFile Plain text file which contains space delimited image
     * file paths and labels
     * @param absPaths True if file contains absolute paths for images, false
     * otherwise
     * @param numOfImages number of images to load
     */
    public void loadImages(File imageIdxFile, boolean absPaths, int numOfImages) throws DeepNettsException {
        String parentPath = "";
        if (absPaths == false) {
            parentPath = imageIdxFile.getPath().substring(0, imageIdxFile.getPath().lastIndexOf(File.separator));
        }

        String imgFileName = null;
        String label = null;
        final String[] fColumnNames = columnNames;

        ExecutorService es = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors() - 1);
        List<Future<?>> results = new LinkedList<>();

        // ako je numOfImages manji od broja slika u fajlu logovati
        try (BufferedReader br = new BufferedReader(new FileReader(imageIdxFile))) {
            String line = null;

            for (int i = 0; i < numOfImages; i++) {
                line = br.readLine();
                if (line.isEmpty()) {
                    continue;
                }
                String[] str = line.split(" "); // parse file and class label from line
                imgFileName = str[0];
                if (!absPaths) {
                    imgFileName = parentPath + File.separator + imgFileName;
                }
                label = str[1]; // TODO: if there is no label, use the name of the parent folder as a label

//                try {
//                // ucitavaj slike u ovom a preprocesiraj u posebnom threadu, najbolje submituj preprocesiranje na neki thread pool
//                    final BufferedImage image = ImageIO.read(new File(imgFileName));
//                    final String flabel = label;
//                    final ExampleImage exImg;
//
//                    exImg = new ExampleImage(image, flabel);
//                    exImg.setTargetOutput(oneHotEncode(flabel, fColumnNames));
//                    add(exImg);
//                } catch (IOException ex) {
//                    java.util.logging.Logger.getLogger(ImageSet.class.getName()).log(Level.SEVERE, null, ex);
//                    throw new DeepNettsException("Image loading error!", ex);
//                }
                final BufferedImage image = ImageIO.read(new File(imgFileName));
                final String flabel = label;

                Future<?> result = es.submit(() -> {
                    try {
                        final ExampleImage exImg = new ExampleImage(image, flabel);
                        exImg.setTargetOutput(oneHotEncode(flabel, fColumnNames));
                        add(exImg); // ovaj add i kolekcija bi morali da budu sinhronizovani ...
                        return true;
                    } catch (IOException ex) {
                        java.util.logging.Logger.getLogger(ImageSet.class.getName()).log(Level.SEVERE, null, ex);
                    }
                    return false;
                    // make sure all images are the same size
//                if ((exImg.getWidth() != imageWidth) || (exImg.getHeight() != imageHeight)) throw new DeepNettsException("Bad image size for "+exImg.getFile().getName());
                });
                results.add(result);
            }

//            results.forEach((f) -> {
//                try {
//                    f.get();
//                } catch (InterruptedException ex) {
//                    java.util.logging.Logger.getLogger(ImageSet.class.getName()).log(Level.SEVERE, null, ex);
//                } catch (ExecutionException ex) {
//                    java.util.logging.Logger.getLogger(ImageSet.class.getName()).log(Level.SEVERE, null, ex);
//                }
//            });
            es.shutdown();
            try {
                es.awaitTermination(Long.MAX_VALUE, TimeUnit.SECONDS);
            } catch (InterruptedException ex) {
                java.util.logging.Logger.getLogger(ImageSet.class.getName()).log(Level.SEVERE, null, ex);
            }

            // sacekaj da pool zavrsi
            if (isEmpty()) {
                throw new DeepNettsException("Zero images loaded!");
            }
            LOGGER.info("Loaded " + size() + " images");

        } catch (FileNotFoundException ex) {
            LOGGER.error(ex);
            throw new DeepNettsException("Could not find image file: " + imgFileName, ex);
        } catch (IOException ex) {
            LOGGER.error(ex);
            throw new DeepNettsException("Error loading image file: " + imgFileName, ex);
        }
    }

    /**
     * Creates and returns binary target vector for specified label using
     * one-of-many scheme. Returns all zeros for label 'negative'.
     *
     * TODO: add params size and idx and move to some util class
     *
     * @param label
     * @return
     */
    private float[] oneHotEncode(final String label, final String[] labels) {
        final float[] vect = new float[labels.length];

        if (label.equals("negative")) {
            return vect; // ovaj izbaci u opstoj verziji
        }
        for (int i = 0; i < labels.length; i++) {
            if (labels[i].equals(label)) {
                vect[i] = 1;
            }
        }

        return vect;
    }

//    public String> getLabels() {
//        return Collections.unmodifiableList(labels);
//    }
    public int getLabelsCount() {
        return columnNames.length;
    }

    /**
     * Splits data set into several parts specified by the input parameter
     * partSizes. Values of partSizes parameter represent the sizes of data set
     * parts that will be returned. Part sizes are integer values that represent
     * percents, cannot be negative or zero, and their sum must be 100
     *
     * @param partSizes sizes of the parts in percents
     * @return parts of the data set of specified size
     */
    @Override
    public ImageSet[] split(double... partSizes) {
        if (partSizes.length < 2) {
            throw new IllegalArgumentException("Must specify at least two parts");
        }
        int partsSum = 0;
        for (int i = 0; i < partSizes.length; i++) {
            if (partSizes[i] <= 0) {
                throw new IllegalArgumentException("Value of the part cannot be zero or negative!");
            }
            partsSum += partSizes[i];
        }

        if (partsSum > 100) {
            throw new IllegalArgumentException("Sum of parts/percents cannot be larger than 100!");
        }

        ImageSet[] subSets = new ImageSet[partSizes.length];
        int itemIdx = 0;

        for (int p = 0; p < partSizes.length; p++) {
            ImageSet subSet = new ImageSet(imageWidth, imageHeight);
            int itemsCount = (int) (size() * partSizes[p]);

            for (int j = 0; j < itemsCount; j++) {
                subSet.add(items.get(itemIdx));
                itemIdx++;
            }

            subSets[p] = subSet;
            subSets[p].columnNames = columnNames;
            // anything else? image dimensions?
        }

        return subSets;
    }

    public String[] loadLabels(String filePath) throws DeepNettsException {
        return loadLabels(new File(filePath));
    }

    public String[] loadLabels(File file) throws DeepNettsException {
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line = null;
            List<String> labelsList = new ArrayList<>(); // temporary labels list
            // we can also catch and log FileNotFoundException, IOException in this loop
            while ((line = br.readLine()) != null) {
                labelsList.add(line);
            }
            br.close();
            this.columnNames = labelsList.toArray(new String[labelsList.size()]);
            return columnNames;
        } catch (FileNotFoundException ex) {
            LOGGER.error("Could not find labels file: " + file.getAbsolutePath(), ex);
            throw new DeepNettsException("Could not find labels file: " + file.getAbsolutePath(), ex);
        } catch (IOException ex) {
            LOGGER.error("Error reading labels file: " + file.getAbsolutePath(), ex);
            throw new DeepNettsException("Error reading labels file: " + file.getAbsolutePath(), ex);
        }
    }

    /**
     * Applies zero mean normalization to entire dataset, and returns mean
     * matrix. TODO: this mean tensor is not correct!
     *
     * @return Returns mean matrix for the entire dataset
     */
    public Tensor zeroMean() {
        ExampleImage img = items.get(0);
        mean = new Tensor(img.getHeight(), img.getWidth(), 3);

        // sum all matrices
        for (ExampleImage image : items) {
            mean.add(image.getInput());
        }

        // divide by number of images
        mean.div(items.size());

        // subtract mean from each image
        for (ExampleImage image : items) {
            image.getInput().sub(mean);
        }

        return mean;
    }

    public void invert() {
        for (ExampleImage image : items) {
            //    mean.add(image.getInput());
            float[] rgbVector = image.getRgbVector();
            for (int i = 0; i < rgbVector.length; i++) {
                rgbVector[i] = 1 - rgbVector[i];
            }
        }
    }

    // since inputs are pixels we dont care about input labels
    @Override
    public String[] getOutputLabels() {
        return columnNames;
    }

}
