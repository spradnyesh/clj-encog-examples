(defproject clj-encog-examples "0.3.1"
  :description "encog-examples rewritten in clojure"
  :url "https://github.com/spradnyesh/clj-encog-examples"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.6.0"]
                 [org.encog/encog-core "3.3.0"]]
  :profiles {:repl {:dependencies [[org.clojure/tools.nrepl "0.2.10"]]
                    :plugins [[cider/cider-nrepl "0.10.0-SNAPSHOT" :exclusions [org.clojure/tools.nrepl]]]}})
