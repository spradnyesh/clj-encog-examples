(ns clj-encog-examples.neural.recurrent.elman-xor
  (:require [clj-encog-examples.neural.xor.common :as c]
            [clj-encog-examples.neural.xor.temporal :as t])
  (:import [org.encog.neural.pattern ElmanPattern FeedForwardPattern]
           [org.encog.engine.network.activation ActivationSigmoid]
           [org.encog.neural.networks BasicNetwork]
           [org.encog.neural.networks.training TrainingSetScore]
           [org.encog.neural.networks.training.anneal NeuralSimulatedAnnealing]
           [org.encog.neural.networks.training.propagation.back Backpropagation]
           [org.encog.ml.train.strategy Greedy HybridStrategy StopTrainingStrategy]))

(defn create-elman-network []
  (doto (cast BasicNetwork
              (.generate
               (doto (ElmanPattern.)
                 (.setActivationFunction (ActivationSigmoid.))
                 (.setInputNeurons 1)
                 (.addHiddenLayer 6)
                 (.setOutputNeurons 1))))
    (.reset)))

(defn create-feedforward-network []
  (doto (cast BasicNetwork
              (.generate
               (doto (FeedForwardPattern.)
                 (.setActivationFunction (ActivationSigmoid.))
                 (.setInputNeurons 1)
                 (.addHiddenLayer 6)
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

(defn main []
  (let [training-set (t/generate 120)
        elman-ntwrk (create-elman-network)
        basic-ntwrk (create-feedforward-network)]
    (println "Best error rate with Elman Network: "
             (train-network "Elman" elman-ntwrk training-set))
    (println "Best error rate with Feedforward Network: "
             (train-network "FeedForward" basic-ntwrk training-set))
    (println "Elman should be able to get into the 10% range,\nfeedforward should not go below 25%.\nThe recurrent Elment net can learn better in this case.")
    (println "If your results are not as good, try rerunning, or perhaps training longer."))
  (c/shutdown))
