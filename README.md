# Install modules
intall PyTorch 1.8.2, torch-geometric 1.7.0. Other versions may also work.
# Run the programs
1) run `gen_files.py` to generate TU format graph files

  change `main` parameters:
  `folder` points to the database graph data directory.
  `pickle_f` points to the pickle file which contains the file name to graph type dictionary.
  `n_sub_label = True`: use specific node labels; `n_sub_label = False`: use general node labels

2) run `my_mutag_gin.py` to get graph classification accuracy
