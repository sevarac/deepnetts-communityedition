/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package deepnetts.util;

import java.io.IOException;

/**
 *
 * @author Zoran
 */
public class GenerateRandomNegative {
    public static void main(String[] args) throws IOException {
       // ImageUtils.generateRandomColoredImages(96, 96, 460, "D:\\datasets\\LegoPeopleNoviJecaPreprocessed\\negative");
        ImageSetUtils.createImageIndex("D:\\datasets\\LegoPeopleNoviJecaPreprocessed\\");
    }
}
