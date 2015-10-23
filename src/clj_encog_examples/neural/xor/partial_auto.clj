(ns clj-encog-examples.neural.xor.partial-auto
  (:require [clj-encog-examples.neural.xor.common :as c])
  (:import [org.encog.util Format]
           [org.encog.util.simple EncogUtility]
           [org.encog.neural.data.basic BasicNeuralDataSet]
           [org.encog.neural.networks BasicNetwork]
           [org.encog.neural.networks.structure AnalyzeNetwork]))

(defn main []
  (let [training-set (BasicNeuralDataSet. c/INPUT c/IDEAL)
        network (doto (. EncogUtility (simpleFeedForward 2 10 10 1 false)) (.reset))]
    (. EncogUtility (trainToError network training-set 0.01))
    (let [analyze (AnalyzeNetwork. network)
          remove (/ (.. analyze getWeights getHigh) 50)]
      (println (. analyze toString))
      (println "Remove connections below:"
               (. Format (formatDouble remove 5)))
      (.setProperty network (. BasicNetwork TAG_LIMIT)
                    remove)
      (.. network getStructure finalizeLimit)

      (println (. (AnalyzeNetwork. network) toString))

      (println "Final output:")
      (. EncogUtility (evaluate network training-set))))
  (c/shutdown))
