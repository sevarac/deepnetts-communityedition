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
    private ExecutorService es;
    private final int threadCount;

    private DeepNettsThreadPool() {
        threadCount = PhysicalCores.physicalCoreCount()-1;//Runtime.getRuntime().availableProcessors()-1; // or use DeepNetts configuration if autodo autodetect
        es = Executors.newFixedThreadPool(threadCount);
    }
    
    public static DeepNettsThreadPool getInstance() {
        if (instance==null) instance = new DeepNettsThreadPool();
        
        // if es was shutdown create new es
        if (instance.es.isShutdown() || instance.es.isTerminated()) {
            instance.es = Executors.newFixedThreadPool(instance.threadCount);
        }        
        return instance; 
    }
    
    public void run(Collection<Callable<Void>> tasks) throws InterruptedException {
        es.invokeAll(tasks);
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
//        if (es.isTerminated()) {
//            es = Executors.newFixedThreadPool(threadCount);
//        }           
        es.submit(task);
    }
    
    // maintain a list of running trainings  and allow shitdown only when trainings are unsubscribed
    public void shutdown() {
        es.shutdown();
    }

    final public int getThreadCount() {
        return threadCount;
    }
    
 }