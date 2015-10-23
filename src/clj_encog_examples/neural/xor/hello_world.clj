(ns clj-encog-examples.neural.xor.hello-world
  (:require [clj-encog-examples.neural.xor.common :as c])
  (:import [org.encog.ml.data.basic BasicMLDataSet]
           [org.encog.neural.networks.training.propagation.resilient ResilientPropagation]))

(defn main []
  (let [training-set (BasicMLDataSet. c/INPUT c/IDEAL)
        network (c/create-network 5)
        train (ResilientPropagation. network training-set)]
    (c/do-training train)
    (c/print-results network training-set))
  (c/shutdown))
