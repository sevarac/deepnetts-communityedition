package deepnetts.data;

import deepnetts.util.Tensor;
import java.io.File;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;
import org.junit.Ignore;

/**
 *
 * @author Zoran
 */
public class ImageSetTest {
    
    public ImageSetTest() {
    }
    
    @BeforeClass
    public static void setUpClass() {
    }
    
    @AfterClass
    public static void tearDownClass() {
    }
    
    @Before
    public void setUp() {
    }
    
    @After
    public void tearDown() {
    }

    /**
     * Test of add method, of class ImageSet.
     */
    @Ignore
    public void testAdd() {
        System.out.println("add");
        ExampleImage exImage = null;
        ImageSet instance = null;
        instance.add(exImage);
        fail("The test case is a prototype.");
    }

    /**
     * Test of loadImages method, of class ImageSet.
     */
    @Ignore

    public void testLoadImages_File_boolean() throws Exception {
        System.out.println("loadImages");
        File imageIdxFile = null;
        boolean absPaths = false;
        ImageSet instance = null;
        instance.loadImages(imageIdxFile);
        fail("The test case is a prototype.");
    }

    /**
     * Test of loadImages method, of class ImageSet.
     */
    @Ignore
    public void testLoadImages_String_boolean() throws Exception {
        System.out.println("loadImages");
        String imageIdxFile = "";
        boolean absPaths = false;
        ImageSet instance = null;
        instance.loadImages(imageIdxFile);
        fail("The test case is a prototype.");
    }

    /**
     * Test of loadImages method, of class ImageSet.
     */
    @Ignore
    public void testLoadImages_3args() {
        System.out.println("loadImages");
        File imageIdxFile = new File("D://datasets/mnis//train/train.txt");
        boolean absPaths = false;
        int numOfImages = 1000;
        ImageSet instance = new ImageSet(28, 28);
        instance.loadImages(imageIdxFile, numOfImages);
        fail("The test case is a prototype.");
    }

    /**
     * Test of getLabelsCount method, of class ImageSet.
     */
    @Ignore
    public void testGetLabelsCount() {
        System.out.println("getLabelsCount");
        ImageSet instance = null;
        int expResult = 0;
        int result = instance.getLabelsCount();
        assertEquals(expResult, result);
        fail("The test case is a prototype.");
    }

    /**
     * Test of split method, of class ImageSet.
     */
    @Ignore
    public void testSplit() {
        System.out.println("split");
        double[] partSizes = null;
        ImageSet instance = null;
        ImageSet[] expResult = null;
        ImageSet[] result = instance.split(partSizes);
        assertArrayEquals(expResult, result);
        fail("The test case is a prototype.");
    }

    /**
     * Test of loadLabels method, of class ImageSet.
     */
    @Ignore
    public void testLoadLabels_String() {
        System.out.println("loadLabels");
        String filePath = "";
        ImageSet instance = null;
        String[] expResult = null;
        String[] result = instance.loadLabels(filePath);
        assertArrayEquals(expResult, result);
        fail("The test case is a prototype.");
    }

    /**
     * Test of loadLabels method, of class ImageSet.
     */
    @Ignore
    public void testLoadLabels_File() {
        System.out.println("loadLabels");
        File file = null;
        ImageSet instance = null;
        String[] expResult = null;
        String[] result = instance.loadLabels(file);
        assertArrayEquals(expResult, result);
        fail("The test case is a prototype.");
    }

    /**
     * Test of zeroMean method, of class ImageSet.
     */
    @Ignore
    public void testZeroMean() {
        System.out.println("zeroMean");
        ImageSet instance = null;
        Tensor expResult = null;
        Tensor result = instance.zeroMean();
        assertEquals(expResult, result);
        fail("The test case is a prototype.");
    }


    /**
     * Test of getOutputLabels method, of class ImageSet.
     */
    @Ignore
    public void testGetOutputLabels() {
        System.out.println("getOutputLabels");
        ImageSet instance = null;
        String[] expResult = null;
        String[] result = instance.getOutputLabels();
        assertArrayEquals(expResult, result);
        fail("The test case is a prototype.");
    }
    
}
