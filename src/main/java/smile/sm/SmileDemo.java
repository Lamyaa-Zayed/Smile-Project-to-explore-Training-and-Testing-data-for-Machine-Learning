package smile.sm;

import org.apache.commons.csv.CSVFormat;
import smile.data.DataFrame;
import smile.data.vector.BaseVector;
import smile.io.Read;
import smile.plot.swing.*;
import smile.plot.swing.Canvas;

import java.awt.*;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.net.URISyntaxException;

public class SmileDemo {
    String trainPath = "src/main/resources/data/Mobile_train.csv";
    String testPath = "src/main/resources/data/Mobile_test.csv";

    public SmileDemo() {
    }

    public static void main(String[] args) {
        SmileDemo sd = new SmileDemo ();
        DataFrame trainData = sd.readCSV (sd.trainPath);
        DataFrame testData = sd.readCSV (sd.testPath);
        sd.getTrainDataSummery (trainData);
        sd.processTrainData (trainData);
        sd.plotData (trainData);
    }

    public DataFrame readCSV(String path) {
        CSVFormat format = CSVFormat.DEFAULT.withFirstRecordAsHeader ();
        DataFrame df = null;
        try {
            df = Read.csv (path, format);
        } catch (IOException e) {
            e.printStackTrace ();
        } catch (URISyntaxException e) {
            e.printStackTrace ();
        }
        System.out.println (df.summary ());
        return df;
    }
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    public void getTrainDataSummery(DataFrame data) {
        DataFrame summary = data.summary ();
        DataFrame selectedColumns = data.select ("battery_power", "n_cores");
        System.out.println (summary);
        System.out.println (data.slice (0, 5));
        System.out.println (data.select (5));
        System.out.println (selectedColumns.slice (0, 10));
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

    public DataFrame processTrainData(DataFrame data){
       DataFrame nonNullData= data.omitNullRows ();
        System.out.println ("Number of non Null rows is: "+nonNullData.nrows ());
        BaseVector talk_timeDF=nonNullData.column ("talk_time");

        return nonNullData;
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    public void plotData(DataFrame data) {
        if (data != null) {
          DataFrame selectedDF = data.select ("clock_speed","int_memory");
            Canvas canvas = Histogram.of (selectedDF.doubleVector (0).array ()).canvas ();
            try {
                canvas.window();
            } catch (InterruptedException e) {
                e.printStackTrace ();
            } catch (InvocationTargetException e) {
                e.printStackTrace ();
            }
        }
    }
}
