(ns clj-encog-examples.neural.xor.common
  (:import [org.encog Encog]
           [org.encog.neural.networks BasicNetwork]
           [org.encog.neural.networks.layers BasicLayer]
           [org.encog.engine.network.activation ActivationSigmoid]))

(def INPUT (into-array (map double-array [[0.0 0.0][1.0 0.0][0.0 1.0][1.0 1.0]])))
(def IDEAL (into-array (map double-array [[0.0][1.0][1.0][0.0]])))

(defn create-network [hidden-1]
  (doto (BasicNetwork.)
    (. addLayer (BasicLayer. nil true 2))
    (. addLayer (BasicLayer. (ActivationSigmoid.) true hidden-1))
    (. addLayer (BasicLayer. (ActivationSigmoid.) false 1))
    (.. getStructure finalizeStructure)
    (. reset)))

(defn do-training [train]
  (loop [i 0, error 1]
    (if (<= error 0.01)
      (. train finishTraining)
      (do (. train iteration)
          (let [err (.getError train)]
            (println "Epoch #" (inc i) ", Error: " err)
            (recur (inc i) err))))))

(defn print-results [network training-set]
  (println "Neural Network Results:")
  (dotimes [i (. training-set size)]
    (let [pair (.get training-set i)
          output (. network compute (. pair getInput))]
      (println (.. pair getInput (getData 0))
               "," (.. pair getInput (getData 1))
               ", actual =" (. output getData 0)
               ", ideal =" (. (. pair getIdeal) getData 0)))))

(defn shutdown []
  (.. Encog getInstance shutdown))
