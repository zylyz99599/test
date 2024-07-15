package MultiThread;

import java.util.concurrent.atomic.AtomicInteger;

public class AtomicIntegerExample {
    private static AtomicInteger counter = new AtomicInteger(0);

    public static void main(String[] args) throws InterruptedException {
        Thread thread1 = new Thread(() -> {
            for (int i = 0; i < 3; i++) {
                int value = counter.incrementAndGet();
//                if (value % 3 == 0) {
                    System.out.println("Thread 1: " + value);
//                }
            }
        });

        Thread thread2 = new Thread(() -> {
            for (int i = 0; i < 2; i++) {
                int value = counter.incrementAndGet();
//                if (value % 2 == 0) {
                    System.out.println("Thread 2: " + value);
//                }
            }
        });

        thread1.start();
        thread2.start();

    }
}
