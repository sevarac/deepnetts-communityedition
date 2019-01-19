package deepnetts.util;

/**
 * Static utility methods for tensors
 * 
 * @author zoran
 */
public class Tensors {
    
    private Tensors() { }

   
    
    public static Tensor zeros(int cols) {
        return new Tensor(cols, 0f);
    }

    public static Tensor ones(int cols) {
        return new Tensor(cols, 1.0f);
    }    
    
    public static Tensor random(int rows, int cols) {
        Tensor tensor = new Tensor(rows, cols);

        for (int i = 0; i < tensor.getRows(); i++) {
            for (int j = 0; j < tensor.getCols(); j++) {
                tensor.set(i, j, RandomGenerator.getDefault().nextFloat());
            }
        }
        return tensor;
    }
    
}
