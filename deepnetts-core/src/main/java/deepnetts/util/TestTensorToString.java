package deepnetts.util;

public class TestTensorToString {
    public static void main(String[] args) {
        Tensor t = Tensor.create(4, 3, new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f});
        System.out.println(t);
    }

}
