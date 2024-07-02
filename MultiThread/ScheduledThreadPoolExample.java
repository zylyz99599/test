package MultiThread;

import java.util.concurrent.*;

public class ScheduledThreadPoolExample {
    public static void main(String[] args) {
        // 创建一个具有4个核心线程的ScheduledExecutorService
        ScheduledExecutorService scheduledExecutorService = Executors.newScheduledThreadPool(4);
        // 创建一个一次性任务，延迟3秒后执行
        Runnable oneTimeTask = () -> System.out.println("One-time task executed at: " + System.currentTimeMillis());
//        scheduledExecutorService.schedule(oneTimeTask, 3, TimeUnit.SECONDS);

        // 创建一个周期性任务，初始延迟1秒后，每2秒执行一次
        Runnable periodicTask = () -> System.out.println("Periodic task executed at: " + System.currentTimeMillis());
        scheduledExecutorService.scheduleAtFixedRate(periodicTask, 1, 2, TimeUnit.SECONDS);

        // 创建一个周期性任务，初始延迟1秒后，每次任务结束后延迟2秒再执行下一次
        Runnable fixedDelayTask = () -> {
            System.out.println("Fixed-delay task started at: " + System.currentTimeMillis());
            try {
                Thread.sleep(1000); // 模拟任务执行时间
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            System.out.println("Fixed-delay task ended at: " + System.currentTimeMillis());
        };
//        scheduledExecutorService.scheduleWithFixedDelay(fixedDelayTask, 1, 2, TimeUnit.SECONDS);

        // 运行一段时间后，关闭线程池
        try {
            Thread.sleep(10000); // 让程序运行10秒
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        scheduledExecutorService.shutdown();
    }
}
