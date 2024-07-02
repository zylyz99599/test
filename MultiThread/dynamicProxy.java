package MultiThread;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;

interface Service{
    void perform();
}

class RealService implements Service{

    @Override
    public void perform() {
        System.out.println("真实类的方法输出~~~~");
    }
}

class ServiceInvokeHandler implements InvocationHandler{

    private final Object object;

    ServiceInvokeHandler(Object object) {
        this.object = object;
    }

    @Override
    public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
        System.out.println("在被代理处理之前的方法输出~~~~");
        Object result = method.invoke(object, args);
        System.out.println("在被代理处理之后的方法输出~~~~");
        return result;
    }
}

public class dynamicProxy {

    public static void main(String[] args) {
        // 创建真实被代理类
        Service realService = new RealService();

        ServiceInvokeHandler serviceInvokeHandler = new ServiceInvokeHandler(realService);

        Service service = (Service) Proxy.newProxyInstance(realService.getClass().getClassLoader(), realService.getClass().getInterfaces(), serviceInvokeHandler);

        service.perform();

    }

}
