(ns clj-encog-examples.neural.xor.factory
  (:require [clj-encog-examples.neural.xor.common :as c]
            [clojure.string :as str])
  (:import [org.encog.util.simple EncogUtility]
           [org.encog.ml.data.basic BasicMLDataSet]
           [org.encog.ml.factory MLMethodFactory MLTrainFactory]
           [org.encog.ml MLResettable MLRegression]
           [org.encog.ml.train.strategy RequiredImprovementStrategy]
           [org.encog.neural.networks.training.propagation.manhattan ManhattanPropagation]))

;; TODO: relu not working

(def feedforward-a "?:B->SIGMOID->4:B->SIGMOID->?")
(def feedforward-relu "?:B->RELU->5:B->LINEAR->?")
(def biasless-a "?->SIGMOID->4->SIGMOID->?")
(def svmc-a "?->C->?")
(def svmr-a "?->R->?")
(def rbf-a "?->gaussian(c=4)->?")
(def pnnc-a "?->C(kernel=gaussian)->?")
(def pnnr-a "?->R(kernel=gaussian)->?")

(defn process [method-name method-architecture trainer-name trainer-args output-neurons]
  (let [method (. (MLMethodFactory.)
                  (create method-name method-architecture 2 output-neurons))
        dataset (BasicMLDataSet. c/INPUT c/IDEAL)
        train (. (MLTrainFactory.)
                 (create method dataset trainer-name trainer-args))]
    (when (and (instance? MLResettable method)
               (not (instance? ManhattanPropagation train)))
      (. train (addStrategy (RequiredImprovementStrategy. 500))))
    ;; output goes to repl (and *not* shown in cider)
    (. EncogUtility (trainToError train 0.01))
    (. EncogUtility (evaluate (cast MLRegression (. train getMethod)) dataset))
    ;; finally, write out what we did
    (println "Machine Learning Type: " method-name)
    (println "Machine Learning Architecture: " method-architecture)
    (println "Training Method: " trainer-name)
    (println "Training Args: " trainer-args)))

(defn backprop []
  (process MLMethodFactory/TYPE_FEEDFORWARD
           feedforward-a
           MLTrainFactory/TYPE_BACKPROP
           ""
           1))

(defn rprop []
  (process MLMethodFactory/TYPE_FEEDFORWARD
           feedforward-a
           MLTrainFactory/TYPE_RPROP
           ""
           1))

(defn relu []
  (process MLMethodFactory/TYPE_FEEDFORWARD
           feedforward-relu
           MLTrainFactory/TYPE_RPROP
           ""
           1))

(defn biasless []
  (process MLMethodFactory/TYPE_FEEDFORWARD
           biasless-a
           MLTrainFactory/TYPE_RPROP
           ""
           1))

(defn svm-classify []
  (process MLMethodFactory/TYPE_SVM
           svmc-a
           MLTrainFactory/TYPE_SVM
           ""
           1))

(defn svm-regression []
  (process MLMethodFactory/TYPE_SVM
           svmr-a
           MLTrainFactory/TYPE_SVM
           ""
           1))

(defn svm-search-regression []
  (process MLMethodFactory/TYPE_SVM
           svmr-a
           MLTrainFactory/TYPE_SVM_SEARCH
           ""
           1))

(defn anneal []
  (process MLMethodFactory/TYPE_FEEDFORWARD
           feedforward-a
           MLTrainFactory/TYPE_ANNEAL
           ""
           1))

(defn genetic []
  (process MLMethodFactory/TYPE_FEEDFORWARD
           feedforward-a
           MLTrainFactory/TYPE_GENETIC
           ""
           1))

(defn lma []
  (process MLMethodFactory/TYPE_FEEDFORWARD
           feedforward-a
           MLTrainFactory/TYPE_LMA
           ""
           1))

(defn manhattan []
  (process MLMethodFactory/TYPE_FEEDFORWARD
           feedforward-a
           MLTrainFactory/TYPE_MANHATTAN
           "lr=0.0001"
           1))

(defn nm []
  (process MLMethodFactory/TYPE_FEEDFORWARD
           feedforward-a
           MLTrainFactory/TYPE_NELDER_MEAD
           ""
           1))

(defn scg []
  (process MLMethodFactory/TYPE_FEEDFORWARD
           feedforward-a
           MLTrainFactory/TYPE_SCG
           ""
           1))

(defn rbf []
  (process MLMethodFactory/TYPE_RBFNETWORK
           rbf-a
           MLTrainFactory/TYPE_RPROP
           ""
           1))

(defn svd []
  (process MLMethodFactory/TYPE_RBFNETWORK
           rbf-a
           MLTrainFactory/TYPE_SVD
           ""
           1))

(defn pnn-c []
  (process MLMethodFactory/TYPE_PNN
           pnnc-a
           MLTrainFactory/TYPE_PNN
           ""
           2))

(defn pnn-r []
  (process MLMethodFactory/TYPE_PNN
           pnnr-a
           MLTrainFactory/TYPE_PNN
           ""
           1))

(defn qprop []
  (process MLMethodFactory/TYPE_FEEDFORWARD
           feedforward-a
           MLTrainFactory/TYPE_QPROP
           ""
           1))

(defn usage []
  (println "Usage:(clj-encog-examples.neural.xor.factory/main mode)
Where mode is one of:\n")

  (println "backprop - Feedforward biased with backpropagation")
  (println "rprop - Feedforward biased with resilient propagation")
  ;; (println "relu - Feedforward biased with resilient propagation & ReLu activation")
  (println "biasless - Feedforward biasless with resilient")
  (println "svm-c - Support vector machine classification")
  (println "svm-r - Support vector machine regression")
  (println "svm-search-r - Support vector machine search regression")
  (println "anneal - Simulated annealing")
  (println "genetic - Genetic")
  (println "lma - Levenberg Marquadt")
  (println "manhattan - Manhattan Update")
  (println "nm - Nelder Mead")
  (println "scg - Scaled Conjugate Gradient")
  (println "rbf - Radial Basis Function with RPROP")
  (println "svd - Radial Basis Function with SVD")
  (println "pnn-c - Probabalistic Neural Network Classification")
  (println "pnn-r - Probabalistic Neural Network Regression")
  (println "qprop - Quick Propagation"))

(defn main [& arg]
  (if-not arg
    (usage)
    (condp = (str/lower-case (first arg))
      "backprop" (backprop)
      "rprop" (rprop)
      ;; "relu" (relu)
      "biasless" (biasless)
      "svm-c" (svm-classify)
      "svm-r" (svm-regression)
      "svm-search-r" (svm-search-regression)
      "anneal" (anneal)
      "genetic" (genetic)
      "lma" (lma)
      "manhattan" (manhattan)
      "nm" (nm)
      "scg" (scg)
      "rbf" (rbf)
      "svd" (svd)
      "pnn-c" (pnn-c)
      "pnn-r" (pnn-r)
      "qprop" (qprop)
      :t (usage)))
  (c/shutdown))
