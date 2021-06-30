public class AlgorithmPSO {
    int n = 2;
    double[] y;
    double[] x;
    double[] v;
    double c1 = 2;
    double c2 = 2;
    double pbest[];
    double gbest;
    double vmax = 0.1;

    public void fitnessFunction(){
        for (int i = 0; i < n; i++){
            y[i] = -1 * x[i] * (x[i] - 2);
        }
    }

    public void init(){
        x = new double[n];
        v = new double[n];
        y = new double[n];
        pbest = new double[n];
        
        x[0] = 0.;
        x[1] = 2.;
        v[0] = 0.01;
        v[1] = 0.02;
        fitnessFunction();

        for (int i = 0; i < n; i++){
            pbest[i] = y[i];
            if (y[i] > gbest) gbest = y[i];
        }
        System.out.println("算法开始，起始最优解："+gbest);
        System.out.print("\n");
    }

    public double getMAX(double a, double b) {
        return a>b?a:b;
    }

    public void PSO(int max){
        for (int i = 0; i < max; i++){
            double w = 0.4;
            for (int j = 0; j < n; j++){
                v[j] = w * v[j] 
                + c1 * Math.random() * (pbest[j] - x[j])
                + c2 * Math.random() * (gbest - x[j]);

                if (v[j] > vmax) v[j] = vmax;
                x[j] += v[j];

                if (x[j] > 2) x[j] = 2;
                if (x[j] < 0) x[j] = 0;
            }
            fitnessFunction();
            for (int j = 0; j < n; j++){
                pbest[j] = getMAX(y[j], pbest[j]);
                if (pbest[j] > gbest) gbest = pbest[j];
                System.out.println("粒子n"+j+"：x = "+x[j]+"\t"+"v = "+v[j]);
            }
            System.out.println("第"+(i+1)+"次迭代，全局最优解 gbest = "+gbest);
            System.out.print("\n");
        }
    }

    public static void main(String[] args){
        AlgorithmPSO ts = new AlgorithmPSO();
        ts.init();
        ts.PSO(100);
    }
}