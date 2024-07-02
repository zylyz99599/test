package MultiThread;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class FixedThreadPoolExample {
    public static void main(String[] args) {
        // 创建一个具有5个线程的固定线程池
        ExecutorService executorService = Executors.newFixedThreadPool(5);

        // 提交10个任务到线程池
        for (int i = 1; i <= 10; i++) {
            final int taskId = i;
            executorService.submit(() -> {
                System.out.println("Task " + taskId + " is running by " + Thread.currentThread().getName());
                try {
                    Thread.sleep(2000); // 模拟任务执行
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                System.out.println("Task " + taskId + " completed by " + Thread.currentThread().getName());
            });
        }

        // 关闭线程池
        executorService.shutdown();
    }
}
