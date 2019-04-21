package deepnetts.util;

import java.util.Collection;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class DeepNettsThreadPool {

    private static DeepNettsThreadPool instance;
    private final ExecutorService es;
    private final int threadCount;

    private DeepNettsThreadPool() {
        threadCount = 2;//Runtime.getRuntime().availableProcessors()-1;
        es = Executors.newFixedThreadPool(threadCount);
    }
    
    private DeepNettsThreadPool(int threadCount) {
        this.threadCount = threadCount;//Runtime.getRuntime().availableProcessors()-1;
        es = Executors.newFixedThreadPool(threadCount);
    }    

    public static DeepNettsThreadPool getInstance() {
        if (instance==null) instance = new DeepNettsThreadPool();
        return instance;
    }
    
    public static DeepNettsThreadPool getInstance(int threadCount) {
        if (instance==null) instance = new DeepNettsThreadPool(threadCount);
        return instance;
    }    

    public void run(Collection<Callable<Void>> tasks) throws InterruptedException {
        es.invokeAll(tasks);
    }
    
    public <T> List<Future<T>> runAll(Collection<Callable<T>> tasks) throws InterruptedException {
        return es.invokeAll(tasks);
    }
    
    
    /**
     * Submit a single task to thread pool.
     * @param task
     * @return 
     */
    public Future<?> submit(Callable<?> task) {
        return es.submit(task);
    }

    public void run(Runnable task) {
        es.submit(task);
    }
    
    public void shutdown() {
        es.shutdown();
    }

    final public int getThreadCount() {
        return threadCount;
    }
    
 }