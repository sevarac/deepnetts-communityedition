/**
 *  DeepNetts is pure Java Deep Learning Library with support for Backpropagation
 *  based learning and image recognition.
 *
 *  Copyright (C) 2017  Zoran Sevarac <sevarac@gmail.com>
 *
 *  This file is part of DeepNetts.
 *
 *  DeepNetts is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.package deepnetts.core;
 */

package deepnetts.data;

import deepnetts.util.ImageUtils;
import deepnetts.util.Tensor;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

/**
 * This class represents example image to train the network.
 * It contains image and label information.
 */
public class ExampleImage implements MLDataItem {

    /**
     * Image dimensions - width and height
     */
    private final int width, height;   // dont need this here , maybe only in dataset

    /**
     * Image label, a concept to map to this image
     */
    private final String label;

    /**
     * Desired network output - maybe its better to use  int - output index with 1 ? lesss memory for huge data sets - TODO: use int here
     */
    private Tensor targetOutput; // output vector depends on number of classes- this could be int in order to save memory

    /**
     * Transformed RGB values of Image pixels
     * used as an input for neural net
     */
    private float[] rgbVector;

    private Tensor rgbTensor;

    private File file;


    
    /**
     * Creates an instance of new example image with specified image and label
     * Loads image from specified file and creates matrix structures with color information
     *
     * @param imgFile image file
     * @param label image label
     * @throws IOException if file is not found or reading file fails from some reason.
     */
    public ExampleImage(File imgFile, String label) throws IOException {
        this.label = label;
        this.file = imgFile;
        BufferedImage image = ImageIO.read(imgFile);
        width = image.getWidth();
        height = image.getHeight();

        createInputFromPixels(image);
    }

    public ExampleImage(BufferedImage image, String label) {
        this.label = label;
        width = image.getWidth();
        height = image.getHeight();

        createInputFromPixels(image);
    }
    
    public ExampleImage(BufferedImage image) {
        this(image, null);
    }    
    
    public ExampleImage(BufferedImage image, String label, int targetWidth, int targetHeight) throws IOException {
        this.label = label;
        width = targetWidth;
        height = targetHeight;  
        
        // if specified image does not fit given dimsnsions scale image
        if (image.getWidth() != targetWidth || image.getHeight() != targetHeight) {
            image = ImageUtils.scaleImage(image, targetWidth, targetHeight);
        }
      
        createInputFromPixels(image);
    }    
    
    private void createInputFromPixels(BufferedImage image) {
        rgbVector = new float[width * height * 3];

        // ako image nije sRGB
        if (image.getType() != BufferedImage.TYPE_INT_ARGB) {
            BufferedImage imageCopy = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_INT_ARGB);
            imageCopy.getGraphics().drawImage(image, 0, 0, null);
            image = imageCopy;
        }
        Raster raster = image.getRaster();
        float[] pixel = null;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                //  int color = image.getRGB(x, y);

                pixel = raster.getPixel(x, y, pixel); // get as butes

                rgbVector[y * width + x] = pixel[0] / 255.0f;
                rgbVector[width * height + y * width + x] = pixel[1] / 255.0f;
                rgbVector[2 * width * height + y * width + x] = pixel[2] / 255.0f;

            }
        }

        rgbTensor = new Tensor(height, width, 3, rgbVector);
    }
    
    public void invert() {
        for (int i = 0; i < rgbVector.length; i++) {
            rgbVector[i] = 1 - rgbVector[i];
        }
    }

    @Override
    public Tensor getTargetOutput() {
        return targetOutput;
    }

    public float[] getRgbVector() {
        return rgbVector;
    }

    // set this internally using utility method
    public final void setTargetOutput(Tensor targetOutput) {
        this.targetOutput = targetOutput;
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public String getLabel() {
        return label;
    }

    @Override
    public Tensor getInput() {
        return rgbTensor;
    }

    public File getFile() {
        return file;
    }

}