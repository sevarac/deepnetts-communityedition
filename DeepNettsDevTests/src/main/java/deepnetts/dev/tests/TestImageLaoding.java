/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package deepnetts.dev.tests;

import deepnetts.data.ImageSet;
import deepnetts.util.ImageSetUtils;
import java.io.FileNotFoundException;
import java.io.IOException;

/**
 *
 * @author Zoran
 */
public class TestImageLaoding {
    public static void main(String[] args) throws FileNotFoundException, IOException {
        // Akljucci
        // data set na disku mora da bude u zadatom formatu
        // ako label sili index vec psotoji neka kreira novi sa numeracijom umesto postojeceg
        
        ImageSet imgSet= new ImageSet(28, 28);
      //  imgSet.loadLabels("d:\\datasets\\mnist\\train\\labels.txt"); // da bi ucitao slike mora prvo da ucita labele, To je lose da poziv jedne metode zavisi od druge i to redolsed. MOzda da napravim builder za image set u kome se zadaju dimenzije, labele, image index i preprocesiranje. Definitivno ali verovatno u sledecemkoraku. SAD AD PRORADI

      // ovde je propust jer ce ucitati samo prv ekoji mogu biti samo jedna klasa...
      
       // imgSet.loadImages(new File("d:\\datasets\\mnist\\train\\train-nolabels.txt"));
   //     imgSet.loadImages(new File("d:\\datasets\\mnist\\train\\train-nolabels.txt"), 10);
        
        // testiraj i onu sa brojem da li hvata labele od parent foldera. Naoraviti specifikaciju foldera za ucitavanje slika - how to prepare data?
        ImageSetUtils.createImageIndex("d:\\datasets\\mnist\\train");
        ImageSetUtils.createLabelsIndex("d:\\datasets\\mnist\\train");
        // testiraj generisanje labels i index fajla
    }
}
