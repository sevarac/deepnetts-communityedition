package deepnetts.net;

/**
 * Neural network architecture types
 * 
 * @author Zoran Sevarac
 */
public enum NetworkType {
    FEEDFORWARD("FEEDFORWARD"), CONVOLUTIONAL("CONVOLUTIONAL");
    
    private final String name;       

    private NetworkType(String s) {
        name = s;
    }    
    
    public boolean equalsName(String otherName) {
        return name.equals(otherName);
    }
    
    public static NetworkType Of(Class networkClass) {
        if (networkClass.equals(FeedForwardNetwork.class)) {
            return FEEDFORWARD;
        } else if (networkClass.equals(ConvolutionalNetwork.class)) {
            return CONVOLUTIONAL;
        }

       throw new RuntimeException("Unknown network type!");       
    }

    @Override
    public String toString() {
       return this.name;
    }        
}
