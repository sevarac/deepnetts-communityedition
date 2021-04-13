package deepnetts.data;

import deepnetts.util.Tensor;
import deepnetts.util.Tensors;
import java.util.List;

public class DataSetStats {
    private Tensor min, max, mean, var, std;
    
    
    // nedostajuce vrednosti
    // kako cemo q1, q2, q3, q4 IQR // za ovo moram da ih sortiram ceo data set - mora da podrzava soritranje, implementira metodu
    // sta ako je binarni, za dn su svi numericki. nema veze?
    
    public DataSetStats(Tensor t) {
//        min = t.clone();
//        max = t.clone();
//        mean = t.clone();
//        mean.fill(0);
//        var = t.clone();
//        var.fill(0);
//        std = t.clone();
//        std.fill(0);        
    }
    
    public Tensor getMin() {
        return min;
    }

    public Tensor getMax() {
        return max;
    }

    public Tensor getMean() {
        return mean;
    }

    public Tensor getVar() {
        return var;
    }

    public Tensor getStd() {
        return std;
    }

    // static factory 
    public static DataSetStats createStats(TabularDataSet dataSet) {
        List<MLDataItem> items = dataSet.getItems();
        
        // sta kod onih kod kojih treba frekvencija, tj, da ih prebroji, najcesvce kod outputa
        
        DataSetStats inputStats = new DataSetStats(items.get(0).getInput());
        DataSetStats targetStats = new DataSetStats(items.get(0).getTargetOutput());
        
        // kako da znam da li da ih brojim ili sabiram, da li gledam frekvenciju kod categroy varabl eili numericku vrednost. to bi korisnik morao da kze, tj data set da sadrzi
        // koje kolone su kategorija/frekvencija? mora da ima ColumnType podesavanje iz Vis reca.  Da li je nezavisna binarna kolona, ili deo grup ebinarnih onehot encoded kolona
        // ako je one hot encoded onda je binary encoded categorical   == enum  kad ih enkodira postavi im tip da bude enum??
        // BinryEncodedCategory
        
        Tensor min, max;
        
        for(MLDataItem item : items) {
            Tensor inputTensor = item.getInput(); // a kako ces da sortiras kolonu?
            inputStats.add(item.getInput());
//                Tensors.min(inputStats.getMin(), inputTensor, inputStats.getMin()); // ovo je binarna operacija nad dva tensora, mozda u TensorOpe
//                Tensors.max(inputStats.getMax(), inputTensor, inputStats.getMax());
                inputStats.getMean().add(inputTensor);
                
                // ako je binarana kolona ond aje broj? ColumnType.Binary odakle da uzmem tip kolone - pa iz data seta
                
                // var
                //std
                // q1, q2, q3, median - za ovo moram da sortiram svaku kolonu! odnosno da ih procesiram nezavisno jednu od druge
                
                targetStats.add(item.getTargetOutput());                        
        }     
        
        inputStats.getMean().div(items.size());
        
        for(MLDataItem item : items) {
            Tensor inputTensor = item.getInput();
            // treba da ima novi tensor a ne da bude inplace
            // bolje da ovde napravim Tensor i da ga setujem u statistici nego ovako da budzim 
            inputStats.getStd(); // sqrt od varijance   
        //    Tensors.sqrt(inputStats.getVar(), s);
        }
        
        // mozda dodati metpdi getValues    u MLDataItem    a d ainput i output vracaju samo one koje treba koji su prethodno setovani
        return inputStats; // kako vratiti i output stats? dva tensora? jedino neka klasa koja ima statistiku z aoba tensora
    }
        
    
    
    public void add(Tensor tensor) {

        mean.add(tensor);
        // mean.div(tensorNum); // za ovo mi treba n
        // median, q1, q2, q3, q4 mora da ih soritram - mozda da mu prosledim celu kolekciju pa ovde to da se resava...
        //var.add(tensor.sub(mean).sqr()); // sqrt div n-1
        // std = min(min, tensor);
    }
    
    
    
}
