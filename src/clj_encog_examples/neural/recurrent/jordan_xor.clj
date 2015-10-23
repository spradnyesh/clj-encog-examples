(ns clj-encog-examples.neural.recurrent.jordan-xor
  (:require [clj-encog-examples.neural.xor.common :as c]
            [clj-encog-examples.neural.recurrent.common :as rc]
            [clj-encog-examples.neural.xor.temporal :as t])
  (:import [org.encog.neural.pattern JordanPattern]
           [org.encog.engine.network.activation ActivationSigmoid]
           [org.encog.neural.networks BasicNetwork]))

(defn create-jordan-network []
  (doto (cast BasicNetwork
              (.generate
               (doto (JordanPattern.)
                 (.setActivationFunction (ActivationSigmoid.))
                 (.setInputNeurons 1)
                 (.addHiddenLayer 2)
                 (.setOutputNeurons 1))))
    (.reset)))

(defn main []
  (let [training-set (t/generate 120)
        jordan-ntwrk (create-jordan-network)
        basic-ntwrk (rc/create-feedforward-network 2)]
    (println "Best error rate with Jordan Network: "
             (rc/train-network "Jordan" jordan-ntwrk training-set))
    (println "Best error rate with Feedforward Network: "
             (rc/train-network "FeedForward" basic-ntwrk training-set))
    (println "Jordan will perform only marginally better than feedforward.\nThe more output neurons, the better performance a Jordan will give."))
  (c/shutdown))
