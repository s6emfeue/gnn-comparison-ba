#### If you happen to use or modify this code, please remember to cite our paper:

[Federico Errica, Marco Podda, Davide Bacciu, Alessio Micheli: *A Fair Comparison of Graph Neural Networks for Graph Classification*](https://openreview.net/pdf?id=HygDF6NFPB). *Proceedings of the 8th International Conference on Learning Representations (ICLR 2020).*

    @inproceedings{errica_fair_2020,
	    title = {A fair comparison of graph neural networks for graph classification},
	    booktitle = {Proceedings of the 8th {International} {Conference} on {Learning} {Representations} ({ICLR})},
	    author = {Errica, Federico and Podda, Marco and Bacciu, Davide and Micheli, Alessio},
	    year = {2020}
    }
--

#### Diese Datei dient als Überblick über alle veränderten Dateien im Zuge der BA GNN Benchmarking

Neue Dateien und Ordner:

- /requirements_new.txt
- /install_test.sh
- /get_data_info.ipynb (Informationen über Datensätze)
- /experiments_comparison.ipynb (Visualisierungen für die BA)
- /config_SparseCIN.yml (Konfiguartion für CIN)
- /config_GraphSAGE_ba.yml (Neue Konfiguartion für GraphSAGE)
- /config_DGCNN_ba.yml (Neue Konfiguartion für DGCNN)
- /config_GIN_ba.yml (Neue Konfiguartion für GIN)
- /Changes_for_BA.md
- /models/graph_classifiers/CIN_GNN (Architektur CIN)
- /datasets/cin_data.py (Datenvorbereitung CIN)


Geänderte Dateien:
- /PrepareDatasets.py (Neue Datensätze und Features preprocessing)
- /Launch_Experiments.py (Neue Datensätze)
- /models/gnn_wrapper/NetWrapper.py (CIN-Coding hinzugefügt (auskommentiert))
- /experiments/EndToEndExperiments.py (Tests für CIN (auskommentiert))
- /datasets/__init__.py (Neue Datensätze)
- /datasets/manager.py (BenchmarkDatasetManager, Neue Datensätze)
- /config/base.py (Neue Datensätze, Neuer Algorithmus)
