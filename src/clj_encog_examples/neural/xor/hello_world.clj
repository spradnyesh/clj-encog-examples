(ns clj-encog-examples.neural.xor.hello-world
  (:import [org.encog Encog]
           [org.encog.ml.data.basic BasicMLDataSet]
           [org.encog.neural.networks BasicNetwork]
           [org.encog.neural.networks.layers BasicLayer]
           [org.encog.engine.network.activation ActivationSigmoid]
           [org.encog.neural.networks.training.propagation.resilient ResilientPropagation]))

(def INPUT (into-array (map double-array [[0.0 0.0][1.0 0.0][0.0 1.0][1.0 1.0]])))
(def IDEAL (into-array (map double-array [[0.0][1.0][1.0][0.0]])))

(defn create-network []
  (doto (BasicNetwork.)
    (. addLayer (BasicLayer. nil true 2))
    (. addLayer (BasicLayer. (ActivationSigmoid.) true 5))
    (. addLayer (BasicLayer. (ActivationSigmoid.) false 1))
    (.. getStructure finalizeStructure)
    (. reset)))

(defn main []
  (let [training-set (BasicMLDataSet. INPUT IDEAL)
        network (create-network)
        train (ResilientPropagation. network training-set)]
    (loop [i 0, error 1]
      (if (<= error 0.01)
        (. train finishTraining)
        (do (. train iteration)
            (let [err (.getError train)]
              (println "Epoch #" (inc i) ", Error: " err)
              (recur (inc i) err)))))
    (println "Neural Network Results:")
    (dotimes [i (. training-set size)]
      (let [pair (.get training-set i)
            output (. network compute (. pair getInput))]
        (println (.. pair getInput (getData 0))
                 "," (.. pair getInput (getData 1))
                 ", actual =" (. output getData 0)
                 ", ideal =" (. (. pair getIdeal) getData 0)))))
  (.. Encog getInstance shutdown))
