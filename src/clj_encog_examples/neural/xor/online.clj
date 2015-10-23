(ns clj-encog-examples.neural.xor.online
  (:require [clj-encog-examples.neural.xor.common :as c])
  (:import [org.encog.util.simple EncogUtility]
           [org.encog.ml.data.basic BasicMLDataSet]
           [org.encog.neural.networks.training.propagation.back Backpropagation]))

(defn main []
  (let [training-set (BasicMLDataSet. c/INPUT c/IDEAL)
        network (doto (. EncogUtility (simpleFeedForward 2 2 0 1 false)) (.reset))
        train (Backpropagation. network training-set 0.07 0.02)]
    (. train (setBatchSize 1))
    ;; output goes to repl (and *not* shown in cider)
    (. EncogUtility (trainToError train 0.01))
    (. EncogUtility (evaluate network training-set)))
  (c/shutdown))
