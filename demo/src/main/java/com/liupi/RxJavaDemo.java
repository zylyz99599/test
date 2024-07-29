package com.liupi;

import io.reactivex.Flowable;
import io.reactivex.Scheduler;
import io.reactivex.schedulers.Schedulers;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class RxJavaDemo {

    private static List<Integer> sharedList = new ArrayList<>();
    private static AtomicInteger counter = new AtomicInteger(0);

    public static void main(String[] args) throws InterruptedException {
        Flowable.range(1, 5)
                .observeOn(Schedulers.io())
                .flatMap(num -> {
                    return Flowable.just(num)
                            .observeOn(Schedulers.computation())
                            .map(item -> {
                                sharedList.add(item); // 修改共享状态
                                return item * 2;
                            });
                })
                .subscribeOn(Schedulers.io())

                .subscribe(
                        result -> System.out.println("Result: " + result),
                        Throwable::printStackTrace,
                        () -> System.out.println("Completed"));

        // 等待足够的时间以确保所有线程完成
        Thread.sleep(2000);

        // 输出最终的共享列表和计数器的值
        System.out.println("Shared List: " + sharedList);
        System.out.println("Counter: " + counter.get());

    }
}
