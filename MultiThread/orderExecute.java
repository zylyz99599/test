package MultiThread;

import java.util.concurrent.DelayQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class orderExecute {

    public static void main(String[] args) throws InterruptedException {

        Ordered ordered = new Ordered();
        new Thread(()->{
            try {
                ordered.printA();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        },"A").start();

        new Thread(()->{
            try {
                ordered.printB();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        },"B").start();

        new Thread(()->{
            try {
                ordered.printC();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        },"C").start();

    }

}
class Ordered{

    private Lock lock = new ReentrantLock();
    private Condition condition1 = lock.newCondition();
    private Condition condition2 = lock.newCondition();
    private Condition condition3 = lock.newCondition();


    private int number=1;

    public void printA() throws InterruptedException {
        lock.lock();
        while (number!=1){
            condition1.await();
        }
        System.out.println(Thread.currentThread().getName()+"=====>A");
        number=2;
        condition2.signal();

        lock.unlock();
    }

    public void printB() throws InterruptedException {
        lock.lock();

        while (number!=2){
            condition3.await();
        }
        System.out.println(Thread.currentThread().getName()+"=====>B");
        number=3;
        condition3.signal();

        lock.unlock();
    }

    public void printC() throws InterruptedException {
        lock.lock();

        while (number!=3){
            condition3.await();
        }
        System.out.println(Thread.currentThread().getName()+"=====>C");
        number=1;
        condition1.signal();

        lock.unlock();
    }

}