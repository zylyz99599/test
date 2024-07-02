package MultiThread;

public class ThreadLocalExample {
    private static ThreadLocal<Integer> threadLocal = ThreadLocal.withInitial(() -> 1);

    public static void main(String[] args) {
//        Runnable task = () -> {
//            int value = threadLocal.get();
//            System.out.println(Thread.currentThread().getName() + ": " + value);
//            threadLocal.set(value + 1);
//            System.out.println(Thread.currentThread().getName() + ": " + threadLocal.get());
//        };
//
//        Thread thread1 = new Thread(task);
//        Thread thread2 = new Thread(task);
//        Thread thread3 = new Thread(task);
//
//        thread1.start();
//        thread2.start();
//        thread3.start();
        Integer a = 5;
        String s = String.valueOf(5);
        char b = '5';
        int i = b - '0';
        char ch = (char)(i + '0');
        System.out.println(ch);
    }
}
