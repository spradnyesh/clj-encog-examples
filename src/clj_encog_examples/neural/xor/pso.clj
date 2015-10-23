(ns clj-encog-examples.neural.xor.pso
  (:require [clj-encog-examples.neural.xor.common :as c])
  (:import [org.encog.util.simple EncogUtility]
           [org.encog.ml.data.basic BasicMLDataSet]
           [org.encog.neural.networks BasicNetwork]
           [org.encog.neural.networks.training TrainingSetScore]
           [org.encog.neural.networks.training.pso NeuralPSO]
           [org.encog.mathutil.randomize NguyenWidrowRandomizer]))

(defn main []
  (let [training-set (BasicMLDataSet. c/INPUT c/IDEAL)
        network (doto (. EncogUtility (simpleFeedForward 2 2 0 1 false)) (.reset))
        score (TrainingSetScore. training-set)
        randomizer (NguyenWidrowRandomizer.)
        train (NeuralPSO. network randomizer score 20)]
    (. EncogUtility (trainToError train 0.01))
    (println "Neural Network Results:")
    (. EncogUtility (evaluate (cast BasicNetwork (. train getMethod))
                              training-set)))
  (c/shutdown))
