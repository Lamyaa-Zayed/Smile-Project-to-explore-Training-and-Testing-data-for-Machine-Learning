package smile.sm;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.net.URISyntaxException;
import java.util.Arrays;
import java.util.Collection;
import java.util.Map;
import java.util.stream.Collectors;
import org.apache.commons.csv.CSVFormat;
import smile.classification.RandomForest;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.data.measure.NominalScale;
import smile.data.vector.IntVector;
import smile.io.Read;
import smile.plot.swing.Histogram;

public class SmileRF2Demo {

    public static int[] encodeCategory(DataFrame df, String columnName) {
        String[] values = df.stringVector(columnName).distinct().toArray(new String[] {});
        int[] pclassValues = df.stringVector(columnName).factorize(new NominalScale(values)).toIntArray();
        return pclassValues;
    }
    public static void main(String[] args) throws InvocationTargetException, InterruptedException, IOException, URISyntaxException {
        DataFrame titanic = Read.csv("src/main/resources/data/data.csv", CSVFormat.DEFAULT.withDelimiter(';')
                .withHeader("Name","Pclass","Age","Sex","Survived")
                .withSkipHeaderRecord(true));

       titanic = titanic.merge(IntVector.of("PclassValue", encodeCategory(titanic, "Pclass")));
        titanic = titanic.merge(IntVector.of("SexValue", encodeCategory(titanic, "Sex")));
        eda(titanic);
        titanic = titanic.drop("Name");
        titanic = titanic.drop("Sex");
        titanic = titanic.drop("Pclass");
        titanic = titanic.omitNullRows();
        System.out.println(titanic.schema());
        System.out.println(titanic.summary());
        RandomForest model = RandomForest.fit(Formula.lhs("Survived"), titanic);
        System.out.println("feature importance:");
        System.out.println(Arrays.toString(model.importance()));
        System.out.println(model.metrics ());
        //TODO load test data to validate model
    }
    private static void eda(DataFrame titanic) throws InterruptedException, InvocationTargetException {
        titanic.summary();
        DataFrame titanicSurvived = DataFrame.of(titanic.stream().filter(t -> t.get("Survived").equals(1)));
        DataFrame titanicNotSurvived = DataFrame.of(titanic.stream().filter(t -> t.get("Survived").equals(0)));
        titanicNotSurvived.omitNullRows().summary();
        titanicSurvived = titanicSurvived.omitNullRows();
        titanicSurvived.summary();
        int size = titanicSurvived.size();
        System.out.println(size);
        Double averageAge = titanicSurvived.stream()
                .mapToDouble(t -> t.isNullAt("Age" ) ? 0.0 : t.getDouble("Age"))
                .average()
                .orElse(0);
        System.out.println(averageAge.intValue());
        Map map = titanicSurvived.stream()
                .collect(Collectors.groupingBy(t -> Double.valueOf(t.getDouble("Age")).intValue(), Collectors.counting()));

        double[] breaks = ((Collection<Integer>)map.keySet())
                .stream()
                .mapToDouble(l -> Double.valueOf(l))
                .toArray();

        int[] valuesInt = ((Collection<Long>) map.values())
                .stream().mapToInt(i -> i.intValue())
                .toArray();

//    Histogram.of(titanicSurvived.doubleVector("Age").toDoubleArray(), values.length, true)
//          .canvas().setAxisLabels("Age","Count")
//          .setTitle("Age frequencies among surviving passengers" )
//          .window();
        Histogram.of(titanicSurvived.intVector("PclassValue").toIntArray(),4, true)
                .canvas().setAxisLabels("Classes","Count")
                .setTitle("Pclass values frequencies among surviving passengers" )
                .window();
        //Histogram.of(values, map.size(), false).canvas().window();
        titanicSurvived.schema();
    }
}