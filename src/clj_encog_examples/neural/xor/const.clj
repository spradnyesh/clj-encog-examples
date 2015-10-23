(ns clj-encog-examples.neural.xor.const
  (:require [clj-encog-examples.neural.xor.common :as c])
  (:import [org.encog.ml.data.basic BasicMLDataSet]
           [org.encog.mathutil.randomize ConsistentRandomizer]
           [org.encog.neural.networks.training.propagation.back Backpropagation]))

(defn main []
  (let [training-set (BasicMLDataSet. c/INPUT c/IDEAL)
        network (c/create-network 2)]
    (. (ConsistentRandomizer. -1 1 500) (randomize network))
    (println (. network dumpWeights))
    (let [train (Backpropagation. network training-set 0.7 0.3)]
      (. train fixFlatSpot false)
      (c/do-training train)
      (c/print-results network training-set)))
  (c/shutdown))
