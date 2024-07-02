package MultiThread;

import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

public class CyclicBarrierExample {
    public static void main(String[] args) {
        final CyclicBarrier barrier1 = new CyclicBarrier(2);
        final CyclicBarrier barrier2 = new CyclicBarrier(2);

        Thread thread1 = new Thread(() -> {
            try {
                System.out.println("Thread 1 is executing...");
                barrier1.await();  // 到达屏障1
            } catch (InterruptedException | BrokenBarrierException e) {
                Thread.currentThread().interrupt();
            }
        });

        Thread thread2 = new Thread(() -> {
            try {
                barrier1.await();  // 等待thread1到达屏障1
                System.out.println("Thread 2 is executing...");
                barrier2.await();  // 到达屏障2
            } catch (InterruptedException | BrokenBarrierException e) {
                Thread.currentThread().interrupt();
            }
        });

        Thread thread3 = new Thread(() -> {
            try {
                barrier2.await();  // 等待thread2到达屏障2
                System.out.println("Thread 3 is executing...");
            } catch (InterruptedException | BrokenBarrierException e) {
                Thread.currentThread().interrupt();
            }
        });

        thread1.start();
        thread2.start();
        thread3.start();
    }
}
