import ember

# ember hard codes the size of data.
# it's best to just let it convert all 1.1M points. This only takes like 30 minutes
# only have to do this once.
# data points are stored in the data directory and are mmaped into memory when training
ember.create_vectorized_features("./data/ember/")
ember.create_metadata("./data/ember/")
