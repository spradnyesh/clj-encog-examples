(ns clj-encog-examples.neural.recurrent.elman-xor
  (:require [clj-encog-examples.neural.xor.common :as c]
            [clj-encog-examples.neural.recurrent.common :as rc]
            [clj-encog-examples.neural.xor.temporal :as t])
  (:import [org.encog.neural.pattern ElmanPattern]
           [org.encog.engine.network.activation ActivationSigmoid]
           [org.encog.neural.networks BasicNetwork]))

(defn create-elman-network []
  (doto (cast BasicNetwork
              (.generate
               (doto (ElmanPattern.)
                 (.setActivationFunction (ActivationSigmoid.))
                 (.setInputNeurons 1)
                 (.addHiddenLayer 6)
                 (.setOutputNeurons 1))))
    (.reset)))

(defn main []
  (let [training-set (t/generate 120)
        elman-ntwrk (create-elman-network)
        basic-ntwrk (rc/create-feedforward-network 6)]
    (println "Best error rate with Elman Network: "
             (rc/train-network "Elman" elman-ntwrk training-set))
    (println "Best error rate with Feedforward Network: "
             (rc/train-network "FeedForward" basic-ntwrk training-set))
    (println "Elman should be able to get into the 10% range,\nfeedforward should not go below 25%.\nThe recurrent Elment net can learn better in this case.")
    (println "If your results are not as good, try rerunning, or perhaps training longer."))
  (c/shutdown))
