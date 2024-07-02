import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicStampedReference;

public class AutomicClassDemo {

    static AtomicStampedReference<Integer> atomicStampedReference = new AtomicStampedReference<>(1, 1);

    public static void main(String[] args) throws InterruptedException {
//        new Thread(() -> {
//            int stamp = atomicStampedReference.getStamp(); // 获取版本号
//            System.out.println("a1=====>" + stamp);
//            try {
//                TimeUnit.SECONDS.sleep(1);
//            } catch (InterruptedException e) {
//                e.printStackTrace();
//            }
//
//            atomicStampedReference.compareAndSet(1, 2, atomicStampedReference.getStamp(),
//                    atomicStampedReference.getStamp() + 1);
//            System.out.println("a2====>"+atomicStampedReference.getStamp());
//            System.out.println(atomicStampedReference.compareAndSet(2,1,atomicStampedReference.getStamp(),
//                    atomicStampedReference.getStamp()+1));
//            System.out.println(atomicStampedReference.getStamp());
//        }, "a").start();
//
//        new Thread(() -> {
//            int stamp = atomicStampedReference.getStamp(); // 获取版本号
//            System.out.println("b1=====>" + stamp);
//            try {
//                TimeUnit.SECONDS.sleep(5);
//            } catch (InterruptedException e) {
//                e.printStackTrace();
//            }
//
//            atomicStampedReference.compareAndSet(2, 4, atomicStampedReference.getStamp(),
//                    atomicStampedReference.getStamp() + 1);
//            System.out.println("a2====>"+atomicStampedReference.getStamp());
//            System.out.println(atomicStampedReference.compareAndSet(4,5,atomicStampedReference.getStamp(),
//                    atomicStampedReference.getStamp()+1));
//            System.out.println(atomicStampedReference.getStamp());
//        }, "b").start();
        Phone3 phone = new Phone3();
        new Thread(()->{
            try {
                phone.sms();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        },"A").start();

        new Thread(()->{
            phone.call();
        },"B").start();

    }
}
class Phone3 {
    public synchronized void sms() throws InterruptedException {
        System.out.println(Thread.currentThread().getName()+"sms");
//        this.wait(1000);
        call();
    }

    public synchronized void call(){
        System.out.println(Thread.currentThread().getName()+"call");
    }
}