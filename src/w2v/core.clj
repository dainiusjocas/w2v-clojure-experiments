(ns w2v.core
  (:require [clojure.string :as string]
            [clojure.java.io :as io])
  (:import [org.deeplearning4j.text.sentenceiterator BasicLineIterator]
           [org.deeplearning4j.models.word2vec Word2Vec$Builder]
           [org.deeplearning4j.text.tokenization.tokenizer.preprocessor CommonPreprocessor]
           [org.deeplearning4j.text.tokenization.tokenizerfactory DefaultTokenizerFactory]
           [org.deeplearning4j.models.embeddings.loader WordVectorSerializer]
           [org.nd4j.linalg.factory Nd4j]
           [org.nd4j.linalg.api.buffer DataBuffer$Type]
           [org.datavec.api.util ClassPathResource]
           [org.deeplearning4j.plot BarnesHutTsne$Builder]))

(defn get-model []                                              ;; `vec` in the original Java
  (-> (Word2Vec$Builder.)
      (.minWordFrequency 5)
      (.iterations 1)
      (.layerSize 100)
      (.seed 42)
      (.windowSize 5)
      (.iterate (BasicLineIterator. "resources/raw_sentences.txt"))
      (.tokenizerFactory (doto (DefaultTokenizerFactory.)
                           (.setTokenPreProcessor (CommonPreprocessor.))))
      (.build)))

(defn fit-model [model]
  (.fit model))

(defn vector-matrix [model word] (.getWordVectorMatrix model word))

(defn nearest [model word] (.wordsNearest model word 10))