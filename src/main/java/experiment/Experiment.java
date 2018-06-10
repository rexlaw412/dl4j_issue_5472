package experiment;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.listener.EarlyStoppingListener;
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

@Slf4j
public class Experiment {
    private static EarlyStoppingGraphTrainer getTrainer(ComputationGraph computationGraph, MultiDataSetIterator trainDataSetIterator, MultiDataSetIterator testDataSetIterator, int epochTimeFrame) {
        EarlyStoppingConfiguration.Builder esCongBuilder = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(epochTimeFrame))
                .scoreCalculator(new DataSetLossCalculator(testDataSetIterator, true))
                .evaluateEveryNEpochs(1)
                .modelSaver(new InMemoryModelSaver());

        EarlyStoppingListener<ComputationGraph> listener = new EarlyStoppingListener<ComputationGraph>() {
            public void onStart(EarlyStoppingConfiguration<ComputationGraph> earlyStoppingConfiguration, ComputationGraph computationGraph) { }
            public void onEpoch(int epochNum, double score, EarlyStoppingConfiguration<ComputationGraph> earlyStoppingConfiguration, ComputationGraph computationGraph) { }
            public void onCompletion(EarlyStoppingResult<ComputationGraph> earlyStoppingResult) { }
        };

        return new EarlyStoppingGraphTrainer(esCongBuilder.build(), computationGraph, trainDataSetIterator, listener);
    }

    private static ComputationGraph getComputationGraphic(int in, int out) {
        ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder()
                .updater(new Adam(0.01))
                .activation(Activation.RELU)
                .graphBuilder()
                .addInputs("IN")
                .setInputTypes(InputType.recurrent(in))
                .addLayer("AUTOENCODER",
                        new VariationalAutoencoder.Builder()
                                .encoderLayerSizes(64)
                                .decoderLayerSizes(64)
                                .nOut(7)
                                .pzxActivationFunction(Activation.IDENTITY)
                                .reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID.getActivationFunction())).build(),
                        "IN")
                .addLayer("RNN", new GravesLSTM.Builder().nOut(128).build(), "AUTOENCODER")//you may comment out this line, or next line for testing
                .addLayer("OUT", new RnnOutputLayer.Builder()
                        .nOut(out)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT).build(), "RNN")
                .setOutputs("OUT")
                .pretrain(true)
                .backprop(true);

        ComputationGraphConfiguration config = builder.build();
        ComputationGraph net = new ComputationGraph(config);
        net.init();

        return net;
    }

    private static ComputationGraph getComputationGraphicCrash(int in, int out) {
        ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder()
                .updater(new Adam(0.01))
                .activation(Activation.RELU)
                .graphBuilder()
                .addInputs("IN")
                .setInputTypes(InputType.recurrent(in))
                .addLayer("AUTOENCODER",
                        new VariationalAutoencoder.Builder()
                                .encoderLayerSizes(64)
                                .decoderLayerSizes(64)
                                .nOut(7)
                                .pzxActivationFunction(Activation.IDENTITY)
                                .reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID.getActivationFunction())).build(),
                        "IN")
                .addLayer("RNN", new Bidirectional(Bidirectional.Mode.ADD, new GravesLSTM.Builder().nOut(128).build()), "AUTOENCODER")
                .addLayer("OUT", new RnnOutputLayer.Builder()
                        .nOut(out)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT).build(), "RNN")
                .setOutputs("OUT")
                .pretrain(true)
                .backprop(true);

        ComputationGraphConfiguration config = builder.build();
        ComputationGraph net = new ComputationGraph(config);
        net.init();

        return net;
    }

    public static void main(String[] args) {
        ExperimentDataIterator iterator = new ExperimentDataIterator();

        EarlyStoppingGraphTrainer earlyStoppingGraphTrainer;
        ComputationGraph net = getComputationGraphic(2, 2);
        earlyStoppingGraphTrainer = getTrainer(net, iterator, iterator, 2);
        earlyStoppingGraphTrainer.fit();
        System.out.println("First Model finish.");

        ComputationGraph netCrash = getComputationGraphicCrash(2, 2);
        net.fit(iterator.next());
        System.out.println("Sceond Model finish.(Not use EarlyStoppingGraphTrainer)");

        earlyStoppingGraphTrainer = getTrainer(netCrash, iterator, iterator, 10);
        earlyStoppingGraphTrainer.fit();
        System.out.println("Sceond Model finish.(Use EarlyStoppingGraphTrainer)");
    }
}