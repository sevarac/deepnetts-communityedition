package deepnetts.util;

import java.io.IOException;

/**
 *
 * @author Zoran
 */
public class GenerateRandomNegative {
    public static void main(String[] args) throws IOException { 
      // ImageUtils.generateRandomColoredImages(96, 96, 30, "D:\\datasets\\DukesChoiceDemo\\negative");
       ImageUtils.generateNoisyImage(96, 96, 10, "D:\\datasets\\DukesChoiceDemo\\negative");
      //  ImageSetUtils.createImageIndex("D:\\datasets\\LegoPeopleNoviJecaPreprocessed\\");
    }
}
