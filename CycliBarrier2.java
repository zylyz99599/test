import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.CyclicBarrier;

public class CycliBarrier2 {
    static CyclicBarrier cycliBarrier;
    static CountDownLatch countDownLatch;

    public static void main(String[] args) throws InterruptedException {

//        cycliBarrier = new CyclicBarrier(10, () -> System.out.println("全部就绪，开始辣！"));
//
//        for (int i = 0; i < 10; i++) {
//            new Thread(() -> {
//                try {
//                    Thread.sleep((long) (Math.random() * 3000));
//                    System.out.println(Thread.currentThread().getName() + ",加载完毕！");
//                    cycliBarrier.await();
//                } catch (InterruptedException e) {
//                    e.printStackTrace();
//                } catch (BrokenBarrierException e) {
//                    e.printStackTrace();
//                }
//                System.out.println(Thread.currentThread().getName() + "，终于加入啦！");
//            }, "我tm" + i + "号来啦").start();
//        }

        countDownLatch = new CountDownLatch(3);
        for (int i = 0; i < 3; i++) {
            new Thread(() -> {
                try {
                    Thread.sleep((long) (Math.random() * 5000));
                    System.out.println(Thread.currentThread().getName() + "，到餐厅了");
                    countDownLatch.countDown();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }, "水友" + i + "号").start();
        }
        countDownLatch.await();
        System.out.println("都到齐了，那么服务员上菜!82年可乐先来一瓶!");

    }
}
