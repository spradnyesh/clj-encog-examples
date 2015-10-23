(ns clj-encog-examples.neural.recurrent.common
  (:import [org.encog.neural.pattern FeedForwardPattern]
           [org.encog.engine.network.activation ActivationSigmoid]
           [org.encog.neural.networks BasicNetwork]
           [org.encog.neural.networks.training TrainingSetScore]
           [org.encog.neural.networks.training.anneal NeuralSimulatedAnnealing]
           [org.encog.neural.networks.training.propagation.back Backpropagation]
           [org.encog.ml.train.strategy Greedy HybridStrategy StopTrainingStrategy]))

(defn create-feedforward-network [hidden]
  (doto (cast BasicNetwork
              (.generate
               (doto (FeedForwardPattern.)
                 (.setActivationFunction (ActivationSigmoid.))
                 (.setInputNeurons 1)
                 (.addHiddenLayer hidden)
                 (.setOutputNeurons 1))))
    (.reset)))

(defn train-network [what network training-set]
  (let [score (TrainingSetScore. training-set)
        train-main (Backpropagation. network training-set 0.000001 0.0)
        stop (StopTrainingStrategy.)]
    (doto train-main
      (. addStrategy (Greedy.))
      (. addStrategy (HybridStrategy. (NeuralSimulatedAnnealing. network score 10 2 100)))
      (. addStrategy stop))
    (loop [i 1, stp nil]
      (if stp
        (. train-main getError)
        (do (. train-main iteration)
            (println "Training " what ", Epoch #" i
                     ", Error: " (. train-main getError))
            (recur (inc i) (. stop shouldStop)))))))
