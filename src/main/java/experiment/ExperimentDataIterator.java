package experiment;

import lombok.AccessLevel;
import lombok.Setter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.rng.CpuNativeRandom;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

public class ExperimentDataIterator implements MultiDataSetIterator {

    @Setter(AccessLevel.NONE)
    private int miniBatchSize;
    @Setter(AccessLevel.NONE)
    private int timeSeriesLength;
    @Setter(AccessLevel.NONE)
    private boolean hasNext;

    public ExperimentDataIterator() {
        this.miniBatchSize = 10;
        this.timeSeriesLength = 5;

        reset();
    }

    @Override
    public MultiDataSet next(int num) {
        INDArray features[] = new INDArray[]{Nd4j.rand(new int[]{num, 2, timeSeriesLength}, 0, 1, new CpuNativeRandom())};
        INDArray labels[] = new INDArray[]{Nd4j.rand(new int[]{num, 2, timeSeriesLength}, 0, 1, new CpuNativeRandom())};

        hasNext = false;

        MultiDataSet data = new org.nd4j.linalg.dataset.MultiDataSet(features, labels);

        return data;
    }

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor multiDataSetPreProcessor) {
    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public void reset() {
        hasNext = true;
    }

    @Override
    public boolean hasNext() {
        return hasNext;
    }

    @Override
    public MultiDataSet next() {
        return next(miniBatchSize);
    }
}