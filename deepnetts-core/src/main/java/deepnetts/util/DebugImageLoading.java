package deepnetts.util;

import deepnetts.data.ImageSet;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;
import javax.imageio.ImageIO;

public class DebugImageLoading {

    static int imageWidth = 100;
    static int imageHeight = 100;

    static String labelsFile = "D:\\datasets\\fruits\\fruits-360\\Training\\labels.txt";
   static String trainingFile = "D:\\datasets\\fruits\\fruits-360\\Training\\index_test2.txt"; // 1000 cifara - probaj sa 10 000    
    
   // static String trainingFile = "D:\\datasets\\mnist\\train\\train.txt";       
    
    public static void main(String[] args) throws FileNotFoundException, IOException {
//        ImageSet imageSet = new ImageSet(imageWidth, imageHeight);
//        imageSet.loadLabels(new File(labelsFile));
//        imageSet.loadImages(new File(trainingFile));

        BufferedImage bi = ImageIO.read(new File("0_100g.jpg"));
        System.out.println("ImageType:" + bi.getType());    
        System.out.println("DataType: "+bi.getSampleModel().getDataType()); // znaci ovaj ima bajtove
        Raster r = bi.getRaster();
        float[] rgb = null;
        rgb = r.getPixel(5, 5, rgb);
        System.out.println(Arrays.toString(rgb) );
        // tip 5
        // blue je posednja 3. komponenta
        // red je prva
        // green je treca
        
        
    }
    
}
