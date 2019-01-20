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

        for (int r = 0; r < tensor.getRows(); r++) {
            for (int c = 0; c < tensor.getCols(); c++) {
                tensor.set(r, c, RandomGenerator.getDefault().nextFloat());
            }
        }
        return tensor;
    }
    
    public static Tensor random(int rows, int cols, int depth) {
        Tensor tensor = new Tensor(rows, cols, depth);

        for (int z = 0; z < tensor.getDepth(); z++) {
            for (int r = 0; r < tensor.getRows(); r++) {
                for (int c = 0; c < tensor.getCols(); c++) {
                    tensor.set(r, c, z, RandomGenerator.getDefault().nextFloat());
                }
            }
        }
        return tensor;
    }    
    
    public static Tensor random(int rows, int cols, int depth, int fourthDim) {
        Tensor tensor = new Tensor(rows, cols, depth, fourthDim);

        for (int f = 0; f < tensor.getFourthDim(); f++) {
            for (int z = 0; z < tensor.getDepth(); z++) {
                for (int r = 0; r < tensor.getRows(); r++) {
                    for (int c = 0; c < tensor.getCols(); c++) {
                        tensor.set(r, c, z, f, RandomGenerator.getDefault().nextFloat());
                    }
                }
            }
        }
        return tensor;
    }      
    
}
