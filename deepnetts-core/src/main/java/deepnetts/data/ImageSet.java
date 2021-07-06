/**
 * DeepNetts is pure Java Deep Learning Library with support for Backpropagation
 * based learning and image recognition.
 * <p>
 * Copyright (C) 2017  Zoran Sevarac <sevarac@gmail.com>
 * <p>
 * This file is part of DeepNetts.
 * <p>
 * DeepNetts is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 * <p>
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details.
 * <p>
 * You should have received a copy of the GNU General Public License along with
 * this program. If not, see <https://www.gnu.org/licenses/>.
 */
package deepnetts.data;

import deepnetts.core.DeepNetts;
import deepnetts.util.DeepNettsException;
import deepnetts.util.ImageSetUtils;
import deepnetts.util.ImageUtils;
import deepnetts.util.Tensor;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import javax.imageio.ImageIO;
import javax.visrec.ml.data.Column;
import javax.visrec.ml.data.DataSet;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Callable;

/**
 * Data set with images that will be used to train convolutional neural network.
 *
 * @author Zoran Sevarac
 */
public class ImageSet extends TabularDataSet<ExampleImage> {

    private final int imageWidth;
    private final int imageHeight;
    private boolean scaleImages = true;
    private boolean invertImages = false;
    private Tensor mean;
    private String delimiter = " ";

    // TODO: method load which takes path to folder with images. May constructor
    // create image and laels index if they are not present
    // mozda ImageDataSetBuilder
    // ukljuci broj slika , index fajlove i sve ostalo
    private static String NEGATIVE_LABEL = "negative";

    private static final Logger LOGGER = LogManager.getLogger(DeepNetts.class.getName());

    // ovi ne mogu svi da budu u memoriji odjednom...
    // osmisliti i neki protocni / buffered data set, koji ucitava jedan batch
    public ImageSet(int imageWidth, int imageHeight) {
        super();
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
    }

    public ImageSet(int imageWidth, int imageHeight, String imageDirPath) throws IOException {
        super();
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;

        ImageSetUtils.createImageIndex(imageDirPath);
        ImageSetUtils.createLabelsIndex(imageDirPath);
        setScaleImages(true);
    }

    final private Object LOCK = new Object();

    /**
     * Adds image to this image set.
     *
     * @param exImage
     * @throws DeepNettsException if image is empty or has wrong dimensions.
     */
    @Override
    public DataSet<ExampleImage> add(ExampleImage exImage) throws DeepNettsException {
        if (exImage == null) {
            throw new DeepNettsException("Example image cannot be null!");
        }
        synchronized (LOCK) {
            items.add(exImage);
        }

//        if ((exImage.getWidth() == imageWidth) && (exImage.getHeight() == imageHeight)) {
//            items.add(exImage);
//        } else {
//            throw new DeepNettsException("Wrong image dimensions for this data set. All images should be "+imageWidth + "x"  + imageHeight);
//        }
        return this;
    }

    public void loadImages(String imageIdxFile) throws FileNotFoundException {
        loadImages(new File(imageIdxFile));
    }

    /**
     * Loads example images with corresponding labels from the specified file.
     *
     *
     * @param imageIdxFile Plain text file that contains space delimited image
     * paths and labels
     * @throws java.io.FileNotFoundException if imageIdxFile was not found
     */
    public void loadImages(File imageIdxFile) throws FileNotFoundException {
        // TODO: First load entire image index, then load and preprocess image in
        // multithreaded way TODO2: load images in batches  verovtano neki iterator nextBatch()

        Objects.requireNonNull(imageIdxFile, "Index file cannot be null!");
        if (columnNames == null) {
            throw new DeepNettsException("Error: Labels are not loaded. In order to load images correctly you have to load labels first using ImageSet.loadLabels method.");
        }

        // use paths of the image index file as root path for image categories
        final String rootPath = imageIdxFile.getPath().substring(0, imageIdxFile.getPath().lastIndexOf(File.separator));

        String imgFileName = null;
        String label = null;

        List<BufferedImage> images = new LinkedList<>();
        List<String> labels = new LinkedList<>();

        // TODO: da radi u batch-u. Da ima interni brojac dokle je stigao. Ili da drzi otvoren stream da iam metodu loadNextBatch() mozda to najbolje u posebnoj metodi ako je mod za trening batch.
        // TODO: napravi ovo asinhrono da ucitava i preprocesira u posebnim threadovima, u perspektivi u batchovima, ne sve odjendnom
        // ucitaj prvo indeks slika a onda ucitavanje i preprocrsiranje slika parelelizuj da jedan thread radi ucitavanje a drugi preprocsiranje onoga sto je ucitano
        try (BufferedReader br = new BufferedReader(new FileReader(imageIdxFile))) {
            String line = null;
            int lineCount = 0;
            //  List<Future<?>> results = new LinkedList<>();
            // we can also catch and log FileNotFoundException, IOException in this loop
            while ((line = br.readLine()) != null) {
                lineCount++;
                if (line.isEmpty()) { // skip empty lines
                    continue;
                }
                String[] parts = line.split(delimiter); // parse file and class label from current line - sta ako naziv fajla sadrzi space? - to ne sme ili detektuj nekako sa lastIndex
                if (parts.length > 2) {
                    throw new DeepNettsException("Bad file format: image paths and labels should not contain spaces! At line " + lineCount);
                }

                imgFileName = parts[0];

                if (parts.length == 2) { // use specified label if it is available
                    label = parts[1];
                } else if (parts.length == 1) {  // otherwise use name of parent folder as label
                    final int labelEndIdx = imgFileName.lastIndexOf(File.separator); // assumes one top directory which corresponds to category label
                    label = imgFileName.substring(0, labelEndIdx);
                }
                //  String shortFileName = parts[0].substring(parts[0].indexOf(File.separator)+1);
                imgFileName = rootPath + File.separator + imgFileName;

                BufferedImage image = ImageIO.read(new File(imgFileName));
                images.add(image);
                labels.add(label);
            }

            processImages(images, labels);

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
        } catch (NullPointerException ex) {
            LOGGER.error(ex);
            throw new DeepNettsException("Error loading image file: " + imgFileName, ex);
        }
    }

    /**
     * Loads specified number of example images with corresponding labels from
     * the specified file.
     *
     * @param imageIdxFile Plain text file which contains space delimited image
     * file paths and label
     * @param numOfImages number of images to load
     */
    public void loadImages(File imageIdxFile, int numOfImages) throws DeepNettsException {
        Objects.requireNonNull(imageIdxFile, "Index file cannot be null!");

        if (columnNames == null) {
            throw new DeepNettsException("Error: Labels are not loaded. In order to load images correctly you have to load labels first using ImageSet.loadLabels method.");
        }

        final String rootPath = imageIdxFile.getPath().substring(0, imageIdxFile.getPath().lastIndexOf(File.separator));

        String imgFileName = null;
        String label = null;

        List<BufferedImage> images = new LinkedList<>();
        List<String> labels = new LinkedList<>();

        // ako je numOfImages manji od broja slika u fajlu logovati
        try (BufferedReader br = new BufferedReader(new FileReader(imageIdxFile))) {
            String line = null;

            for (int i = 0; i < numOfImages; i++) {
                line = br.readLine();
                if (line.isEmpty()) {
                    continue;
                }
                String[] parts = line.split(delimiter); // parse file and class label from line

                if (parts.length > 2) {
                    throw new DeepNettsException("Bad file format: image paths and labels should not contain spaces! At line " + i);
                }

                imgFileName = parts[0];

                if (parts.length == 2) { // use specified label if it is available
                    label = parts[1];
                } else if (parts.length == 1) {  // otherwise use name of parent folder as label
                    final int labelEndIdx = imgFileName.lastIndexOf(File.separator); // assumes one top directory which corresponds to category label
                    label = imgFileName.substring(0, labelEndIdx);
                }

                imgFileName = rootPath + File.separator + imgFileName;
                final BufferedImage image = ImageIO.read(new File(imgFileName));
                images.add(image);
                labels.add(label);
            }

            processImages(images, labels);
        } catch (FileNotFoundException ex) {
            LOGGER.error(ex);
            throw new DeepNettsException("Could not find image file: " + imgFileName, ex);
        } catch (IOException ex) {
            LOGGER.error(ex);
            throw new DeepNettsException("Error loading image file: " + imgFileName, ex);
        }


        // sacekaj da pool zavrsi
        if (isEmpty()) {
            throw new DeepNettsException("Zero images loaded!");
        }
        LOGGER.info("Loaded " + size() + " images");
    }

    private void processImages(List<BufferedImage> images, List<String> labels) throws IOException {
        for (int i = 0; i < images.size(); i++) {
            BufferedImage img = images.get(i);
            final String lbl = labels.get(i);
            if (scaleImages) img = ImageUtils.scaleImage(img, imageWidth, imageHeight);

            final ExampleImage exImg = new ExampleImage(img, lbl);
            exImg.setTargetOutput(new Tensor(oneHotEncode(lbl, columnNames)));
            if (invertImages) exImg.invert();
            add(exImg);
        }
    }

    public void invert() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    private class ImageProcessor implements Callable<Boolean> {

        private final List<BufferedImage> images;
        private final List<String> labels;
        private final int start;
        private final int end;

        public ImageProcessor(List<BufferedImage> images, List<String> labels, int start, int end) {
            this.images = images;
            this.labels = labels;
            this.start = start;
            this.end = end;
        }

        @Override
        public Boolean call() throws IOException {
            Iterator<String> li = labels.iterator();
            for (int i = start; i < end; i++) {
                BufferedImage img = images.get(i);
                String lbl = li.next();
                if (scaleImages) img = ImageUtils.scaleImage(img, imageWidth, imageHeight);

                final ExampleImage exImg = new ExampleImage(img, lbl);
                exImg.setTargetOutput(new Tensor(oneHotEncode(lbl, columnNames)));
                if (invertImages) exImg.invert();
                add(exImg); // vratiti kao batch rezultata
            }

            return Boolean.TRUE;
        }
    }

    /**
     * Creates and returns binary array for specified label using
     * one-hot-encoding scheme. Each position in array corresponds to one label,
     * position with label given as parameter is 1, while other positions are
     * zero. Returns all zeros for label 'negative'.
     *
     * TODO: maybe to greate map and just get corresponding vector for each
     *
     * @param label specific tabel to encode with 1 in return vector
     * @param labels all available labels
     * @return
     */
    private float[] oneHotEncode(final String label, final String[] labels) {
        final float[] returnArr = new float[labels.length];

        if (label.equalsIgnoreCase(NEGATIVE_LABEL)) {
            return returnArr;
        }
        for (int i = 0; i < labels.length; i++) {
            if (labels[i].equals(label)) {
                returnArr[i] = 1;
            }
        }

        return returnArr;
    }

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

        if (partsSum > 1) {
            throw new IllegalArgumentException("Sum of parts/percents cannot be larger than 1!");
        }

        LOGGER.info("Splitting data set: " + Arrays.toString(partSizes));

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
            //subSets[p].columnNames = columnNames;
            subSet.setColumnNames(this.columnNames);
            subSet.setColumns(this.getColumns());
            // anything else? image dimensions?
            subSet.setColumns(this.getColumns());
        }

        return subSets;
    }

    /**
     * Loads and returns image labels to train neural network from the specified
     * file. These labels will be used to label network's outputs.
     *
     * @param filePath
     * @return
     * @throws DeepNettsException
     */
    public String[] loadLabels(String filePath) throws DeepNettsException {
        return loadLabels(new File(filePath));
    }

    /**
     * Loads and returns image labels to train neural network from the specified
     * file.These labels will be used to label network's outputs.
     *
     * @param file file to load labels from
     * @return
     * @throws DeepNettsException
     */
    public String[] loadLabels(File file) throws DeepNettsException {
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line = null;
            List<String> labelsList = new ArrayList<>(); // temporary labels list
            while ((line = br.readLine()) != null) {
                if (line.isEmpty()) continue; // skip empty lines

                line = line.trim();
                if (line.contains(" ")) {
                    throw new DeepNettsException("Bad label format: Labels should not contain space characters! For label:" + line);
                }
                labelsList.add(line);
                this.getColumns().add(new Column(line, Column.Type.BINARY, true));
            }
            this.columnNames = labelsList.toArray(new String[labelsList.size()]);
            setAsTargetColumns(columnNames);

            LOGGER.info("Loaded " + labelsList.size() + " labels");
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
     * tensor.
     *
     * @return mean Tensor for the entire dataset
     */
    public Tensor zeroMean() {
        mean = new Tensor(imageHeight, imageWidth, 3);

        // sum all matrices
        items.forEach((img) -> mean.add(img.getInput()));

        // divide by number of images
        mean.div(items.size());


        //List<Callable<Boolean>> workers = new ArrayList<>();

        // subtract mean from each image
        for (ExampleImage image : items) {
            image.getInput().sub(mean);
        }

//        items.parallelStream().forEach((item)->image.getInput().sub(mean));

        return mean;
    }

    /**
     * Inverts pixels of all images. Usefull when white bacground should be
     * ignored.
     */
//    public void invert() {
//        for (ExampleImage image : items) {
//            float[] rgbVector = image.getRgbVector();
//            for (int i = 0; i < rgbVector.length; i++) {
//                rgbVector[i] = 1 - rgbVector[i];
//            }
//        }
//    }

    /**
     * Returns flag that indicates wheather images should be scaled to specified
     * dimensions while creating image set.
     *
     * @return
     */
    public boolean getScaleImages() {
        return scaleImages;
    }

    public final void setScaleImages(boolean scaleImages) {
        this.scaleImages = scaleImages;
    }

    public boolean getInvertImages() {
        return invertImages;
    }

    public void setInvertImages(boolean invertImages) {
        this.invertImages = invertImages;
    }

    public Map<String, Integer> countByClasses() {
        HashMap<String, Integer> map = new HashMap<>();

        for (ExampleImage item : items) {
            if (map.containsKey(item.getLabel())) {
                final String key = item.getLabel();
                map.put(key, map.get(key) + 1);
            } else {
                map.put(item.getLabel(), 0);
            }
        }

        LOGGER.info("Number of images by label/class");
        for (String key : map.keySet()) {
            LOGGER.info(key + " : " + map.get(key));
        }

        return map;
    }

    public String getDelimiter() {
        return delimiter;
    }

    public void setDelimiter(String delimiter) {
        this.delimiter = delimiter;
    }


}
