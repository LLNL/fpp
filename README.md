# Function Preserving Projection (FPP)

A linear projection technique for finding a 2D view that capture interpretable pattern of the given function in a high-dimensional domain. The function can be univariate or multivariate, continuous (regression) or discrete (classification). The details of the method can be found in the corresponding paper: https://arxiv.org/pdf/1909.11804.pdf

### Dependency:
  tensorflow (<=1.15.0), numpy

### File Description:
  fpp.py - function preserving projection class

  fpp_example.ipynb - fpp usage examples

### Test Dataset Description:

Circle_in_5D_cube.npy - synthetic dataset where a 2D circle pattern of the function exists in a 5D space.

Circle_in_30D.npy - synthetic dataset where a 2D circle pattern of the function exists in a 30D space.

### Fpp Class Usage:

  ```python
  # X - function domain
  # f - function range
  # epoches - training epoches
  # batchSize - traiing batch size
  # proj_mat - projection matrix
  # embedding - the 2D embedding coordinate
  # loss - the loss on the entire training dataset

  ###### regression task #####
  model = fpp()
  model.setup(X, f)
  model.train()
  proj_mat, embedding, loss, _ = model.eval()

  ###### classification task #####
  model = fpp()
  model.setupMultiClass(X, f) #f should be a one-hot encoding of the class
  model.train(epoches, batchSize)
  proj_mat, embedding, loss, _ = model.eval()
  ```


Reviewed and released under LLNL-CODE-791217

Author(s): Shusen Liu (liu42@llnl.gov), Rushil Anirudh (anirudh1@llnl.gov)
