(ns clj-encog-examples.neural.xor.partial
  (:require [clj-encog-examples.neural.xor.common :as c])
  (:import [org.encog.util.simple EncogUtility]
           [org.encog.neural.data.basic BasicNeuralDataSet]
           [org.encog.neural.networks BasicNetwork]))

(defn main []
  (let [training-set (BasicNeuralDataSet. c/INPUT c/IDEAL)
        network (doto (. EncogUtility (simpleFeedForward 2 10 10 1 false))
                  (.reset)
                  (.enableConnection 0 0 0 false)
                  (.enableConnection 0 1 3 false)
                  (.enableConnection 1 1 1 false)
                  (.enableConnection 1 4 4 false))]
    (println "Final output:")
    ;; output goes to repl (and *not* shown in cider)
    (. EncogUtility (trainToError network training-set 0.01))
    (. EncogUtility (evaluate network training-set))

    (println "Training should leave hidden neuron weights at zero.")
    (println "First removed neuron weight:" (. network (getWeight 0 0 0)))
    (println "First removed neuron weight:" (. network (getWeight 0 1 3)))
    (println "First removed neuron weight:" (. network (getWeight 1 1 1)))
    (println "First removed neuron weight:" (. network (getWeight 1 4 4))))
  (c/shutdown))
