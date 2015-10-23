(ns clj-encog-examples.neural.xor.temporal
  (:import [org.encog.ml.data.basic BasicMLDataSet]))

(def SEQUENCE (double-array [1.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0 0.0]))

(defn generate [c]
  (let [cnt (count SEQUENCE)]
    (loop [i c, input [], ideal []]
      (if (zero? i)
        (BasicMLDataSet. (into-array (map double-array input))
                         (into-array (map double-array ideal)))
        (recur (dec i)
               (conj input [(nth SEQUENCE (mod i cnt))])
               (conj ideal [(nth SEQUENCE (mod (inc i) cnt))]))))))
