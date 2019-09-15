import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Main {
    public static void main(String[] args) {
        INDArray array = Nd4j.linspace(1,15,15).reshape('c',3,5);
        System.out.print(array);
        System.out.print(array.sum(1));
    }
}
