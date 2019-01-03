package deepnetts.net.train;

/**
 *
 * @author zoran
 */
public interface TrainerProvider<T> {
        public T getTrainer();
        public void setTrainer(T trainer);
}
