package MultiThread;

import java.util.concurrent.CountDownLatch;

public class MultiThreadSequenceControl {
    public static void main(String[] args) {
        final CountDownLatch latch1 = new CountDownLatch(1);
        final CountDownLatch latch2 = new CountDownLatch(1);

        Thread thread1 = new Thread(() -> {
            System.out.println("Thread 1 is executing...");
            latch1.countDown();  // Release latch1
        });

        Thread thread2 = new Thread(() -> {
            try {
                latch1.await();  // Wait for thread1 to finish
                System.out.println("Thread 2 is executing...");
                latch2.countDown();  // Release latch2
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });

        Thread thread3 = new Thread(() -> {
            try {
                latch2.await();  // Wait for thread2 to finish
                System.out.println("Thread 3 is executing...");
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });

        thread1.start();
        thread2.start();
        thread3.start();
    }
}
