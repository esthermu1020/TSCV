from BlogHelper.Dependencies import *
from sklearn.model_selection._split import _BaseKFold

class TSCVGenerator(_BaseKFold):
    """Time Series cross-validator"""
    
    def __init__(self, data, n_splits=5,split_ratio = 0.7,
                 one_train_size=72, one_val_size = 12, test_size = 24*30,
                 lookback_width=15, target_width=1, prediction_offset=12, BATCH_SIZE=32,
                 sequence_stride=1, sampling_rate=1,train_from_front = False,
                 target_columns=None, iftscv=True):
        '''
        | train | val | train | val | ... | Test
        '''
      
        # init the sample generator 
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.dataset = data
        self.one_train_size = one_train_size 
        self.one_val_size = one_val_size
        self.test_size = test_size
        self.iftscv = iftscv
        self.split_ratio = split_ratio
        self.train_from_front=train_from_front # weather train set uses all data from the front
        self.target_columns = target_columns


        # Time series parameters
        self.sequence_stride = sequence_stride
        self.sampling_rate = sampling_rate
        self.lookback_width = lookback_width
        self.prediction_offset = prediction_offset

        # slice defination
        self.input_slice = slice(0, self.lookback_width, sampling_rate)
        self.input_indices = np.arange(self.lookback_width)[self.input_slice]
        self.sequence_size = len(self.input_indices)
        self.BATCH_SIZE=BATCH_SIZE

        # |       total_window_size=6        |
        # |lookback_width|prediction_offset=2|
        # | 1， 2， 3， 4 |     5， (6)       ｜
        # |lookback_width|      target_width=1|
        # prediction offset tracks the last predicted value (-1)


        # Sample window parameters    
        if (self.lookback_width)%sampling_rate > 0:
          print('Adjusting lookback to a multiple of sampling rate:', self.lookback_width)
        self.total_window_size = self.lookback_width + prediction_offset 
 
        # Target slicing parameters
        # Index for one window, start is included, end is not
        self.train_start = 0
        if not self.iftscv:
          self.one_train_size = int(self.dataset.shape[0] * self.split_ratio) # 0.7
          self.one_val_size = int((self.dataset.shape[0] - self.one_train_size) * self.split_ratio)
          self.test_size = self.dataset.shape[0] - self.one_train_size - self.one_val_size        
          print('train:(val+test) = ', round(self.one_train_size/(self.one_val_size + self.test_size),2))
          print('val:test = ', round(self.one_val_size/self.test_size, 2))
    
        self.val_start = self.train_start + self.one_train_size
        self.tol_length = self.one_train_size + self.one_val_size # for one window |train|val|

        self.target_width = target_width
        self.target_start = self.total_window_size - self.target_width
        self.target_slice = slice(self.target_start, None)
        self.target_indices = np.arange(self.total_window_size)[self.target_slice]    
        
        train_dfs, val_dfs, test_dfs = self.tosets()
        self.train_dfs = train_dfs
        self.val_dfs = val_dfs
        self.test_dfs = test_dfs
        self.test_dfs = self.dataset.iloc[-self.test_size-self.target_start-self.target_width + 1:] if not test_dfs else test_dfs
        self.column_indices = {name: i for i, name in
                              enumerate(self.test_dfs.columns)}


    def split(self, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X = self.dataset
        X, y, groups = indexable(X, y, groups)
        n_samples = X.shape[0]
        self.n_samples = n_samples
        n_splits = self.n_splits
        n_folds = n_splits + 1
        self.train_indices = np.arange(self.tol_length)[self.train_start:self.val_start]
        self.val_indices = np.arange(self.tol_length)[self.val_start:self.tol_length]
        self.test_indices = np.arange(self.n_samples)[-self.test_size:]
        indices = np.arange(n_samples) #(n_samples, )
        self.indices = indices

        val_starts = range(self.val_start, n_samples, self.tol_length)
        self.val_starts = val_starts[:-1]

        for val_start in val_starts:
              val_end = val_start + self.one_val_size # not include in val
              train_end = val_start
              train_start = 0 if self.train_from_front else  train_end - self.one_train_size
              yield (
                    indices[train_start:train_end],
                    indices[val_start-self.target_start-self.target_width + 1 : val_end],
                    None
                )
              
    def tosets(self):   
          train, val = [], []
          for train_index, val_index, _ in self.split():
              train.append(self.dataset.iloc[train_index])
              val.append(self.dataset.iloc[val_index]) 
          return (train, val, None)

    def __repr__(self):
            return '\n'.join([
            f'Total sequence size: {self.tol_length}',
            f'=========',
            f'Train sequence size: {len(self.train_indices)} ',
            f'Train indices: {self.train_indices[:5]}...{self.train_indices[-5:]}',
            f'Val sequence size: {len(self.val_indices)}',
            f'Val indices: {self.val_indices[:5]}...{self.val_indices[-5:]}',
            f'Test sequence size: {self.test_size}',
            f'Test indices: {self.test_indices[:5]}...{self.test_indices[-5:]}',
            f'=========',
            f'Total sequence size: {self.total_window_size}',
            f'Total input size: {self.sequence_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.target_indices}',
            f'Label column name(s): {self.target_columns}'])


    def make_dataset(self, data, targets=True, shuffle=False):

          BATCH_SIZE=self.BATCH_SIZE
          BATCH_SIZE = BATCH_SIZE[0] if isinstance(BATCH_SIZE, tuple) else BATCH_SIZE
          input_data = data.values
          target_coln_id = 0
          if targets:
            target_data = data[self.target_columns].iloc[self.target_start:].values
          else:
            target_data = None
         
          def _timeseries_dataset_from_array(data=input_data, target_coln_id = target_coln_id):
                  n_obs, n_features=data.shape
                  X, Y = [], []
                  for i in range(n_obs - self.target_start - self.target_width + 1):
                              Y.append(data[(i + self.target_start):(i + self.target_start + self.target_width), target_coln_id])
                              X.append(data[i:(i + self.lookback_width)])
                  
                  X, Y = np.array(X), np.array(Y)
                  
                  X = np.reshape(X, (X.shape[0], self.lookback_width, n_features))
                  return X, Y

          ds = _timeseries_dataset_from_array(
                  data=input_data,
                  target_coln_id = target_coln_id
                )
          return ds

    @property
    def train(self):
      Xtrain, ytrain = None, None
      for train_df in self.train_dfs:
          Xtrain = self.make_dataset(train_df)[0] if Xtrain is None \
            else np.concatenate([Xtrain, self.make_dataset(train_df)[0]], axis=0)
          ytrain = self.make_dataset(train_df)[1] if ytrain is None \
            else np.concatenate([ytrain, self.make_dataset(train_df)[1]], axis=0)
      return Xtrain, ytrain
      

    @property
    def val(self):
      Xval, yval = None, None
      for val_df in self.val_dfs:
          Xval = self.make_dataset(val_df)[0] if Xval is None \
            else np.concatenate([Xval, self.make_dataset(val_df)[0]], axis=0)
          yval = self.make_dataset(val_df)[1] if yval is None \
            else np.concatenate([yval, self.make_dataset(val_df)[1]], axis=0)
      return Xval, yval

    @property
    def test(self): 
      return self.make_dataset(self.test_dfs)[0] , self.make_dataset(self.test_dfs)[1]