from BlogHelper.Dependencies import *

# scaler
class QuantileScaler():
      def fit(self, df):
          """
          Return quantile range and median of all features

          Set qrange be 1 if feature has very small range, otherwise calculate
          the quantile range with specified quantile
          """
          eps = 0.1 # set the threshold
          self.df_median = df.median()
          df_q30 = df.quantile(0.3)
          df_q70 = df.quantile(0.7)
          qrange = np.array(df_q70-df_q30)
          if np.isscalar(qrange):
                if qrange < eps:
                      qrange = 1.0
          elif isinstance(qrange, np.ndarray):
                constant_mask = qrange < np.ones_like(qrange)*eps
                qrange[constant_mask] = 1.0
          self.qrange = qrange
          self.df_qrange = pd.Series(index = self.df_median.index, data=qrange)
          return self.qrange, self.df_median

      def transform(self, df):
          """
          Perform scaling
          """
          qrange, df_median = self.fit(df)
          return (df - df_median)/qrange
          

# window generator base
def _build_repr(self):
    # XXX This is copied from BaseEstimator's get_params
    cls = self.__class__
    init = getattr(cls.__init__, "deprecated_original", cls.__init__)
    # Ignore varargs, kw and default values and pop self
    init_signature = signature(init)
    # Consider the constructor parameters excluding 'self'
    if init is object.__init__:
        args = []
    else:
        args = sorted(
            [
                p.name
                for p in init_signature.parameters.values()
                if p.name != "self" and p.kind != p.VAR_KEYWORD
            ]
        )
    class_name = self.__class__.__name__
    params = dict()
    for key in args:
        # We need deprecation warnings to always be on in order to
        # catch deprecated param values.
        # This is set in utils/__init__.py but it gets overwritten
        # when running under python3 somehow.
        warnings.simplefilter("always", FutureWarning)
        try:
            with warnings.catch_warnings(record=True) as w:
                value = getattr(self, key, None)
                if value is None and hasattr(self, "cvargs"):
                    value = self.cvargs.get(key, None)
            if len(w) and w[0].category == FutureWarning:
                # if the parameter is deprecated, don't show it
                continue
        finally:
            warnings.filters.pop(0)
        params[key] = value

    return "%s(%s)" % (class_name, _pprint(params, offset=len(class_name)))


class BaseCrossValidator(metaclass=ABCMeta):
    """Base class for all cross-validators
    Implementations must define `_iter_test_masks` or `_iter_test_indices`.
    """

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        """
        X, y, groups = indexable(X, y, groups)
        indices = np.arange(_num_samples(X))
        for test_index in self._iter_test_masks(X, y, groups):
            train_index = indices[np.logical_not(test_index)]
            test_index = indices[test_index]
            yield train_index, test_index

    def _iter_test_masks(self, X=None, y=None, groups=None):
        """Generates boolean masks corresponding to test sets.
        By default, delegates to _iter_test_indices(X, y, groups)
        """
        for test_index in self._iter_test_indices(X, y, groups):
            test_mask = np.zeros(_num_samples(X), dtype=bool)
            test_mask[test_index] = True
            yield test_mask

    def _iter_test_indices(self, X=None, y=None, groups=None):
        """Generates integer indices corresponding to test sets."""
        raise NotImplementedError

    @abstractmethod
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator"""

    def __repr__(self):
        return _build_repr(self)

class BaseKFold(BaseCrossValidator, metaclass=ABCMeta):
    """Base class"""

    @abstractmethod
    def __init__(self, n_splits, *, shuffle, random_state):
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError(
                "The number of folds must be of Integral type. "
                "%s of type %s was passed." % (n_splits, type(n_splits))
            )
        n_splits = int(n_splits)

        if n_splits <= 1:
            raise ValueError(
                "k-fold cross-validation requires at least one"
                " train/test split by setting n_splits=2 or more,"
                " got n_splits={0}.".format(n_splits)
            )

        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be True or False; got {0}".format(shuffle))

        if not shuffle and random_state is not None:  # None is the default
            raise ValueError(
                "Setting a random_state has no effect since shuffle is "
                "False. You should leave "
                "random_state to its default (None), or set shuffle=True.",
            )

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        if self.n_splits > n_samples:
            raise ValueError(
                (
                    "Cannot have number of splits n_splits={0} greater"
                    " than the number of samples: n_samples={1}."
                ).format(self.n_splits, n_samples)
            )

        for train, test in super().split(X, y, groups):
            yield train, test

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator
        """
        return self.n_splits

class BaseModel:
    def __init__(self, look_back, num_features, loss, layer1=512, layer2=256, target_width=1,
    clip_delta = 1, qrange=1, target_median = 0):
        self.look_back = look_back
        self.num_features = num_features
        self.loss = loss
        self.layer1 = layer1
        self.layer2 = layer2
        self.target_width = target_width
        self.clip_delta = clip_delta
        self.qrange = qrange
        self.target_median =  target_median
        self.create_model()
        self.history = None

    def plot_window(self, data_window, y_scaler, l, r, plot_real=True, legend=True):
        y_base = y_scaler.inverse_transform( self.model.predict( data_window ) ).flatten()
        if l is None:
            l = 0
        if r is None:
            r = y_base.shape[0]
            plt.plot(range(l, r), y_base[l:r], label='base', alpha=1.0)
        if plot_real:
            y_real = []
            X, Y = data_window
            for i in range(len(Y)):
                target = Y[i]
                y_real = y_real + list(np.reshape(target, (-1)))
            y_real = y_scaler.inverse_transform( np.reshape(y_real, (1, -1)) ).flatten()
            plt.plot(range(l, r), y_real[l:r], label='real', alpha=1.0)
        if legend:
            plt.legend()

    def create_model(self):
        '''
        default setting of lstm model activation function
        activation="tanh",
        recurrent_activation="sigmoid",
        '''
        model = Sequential()
        model.add(LSTM(self.layer1, input_shape=(self.look_back, self.num_features), return_sequences=True))
        model.add(Dropout(0.05))
        
        model.add(LSTM(self.layer2, return_sequences=True))
        model.add(Dropout(0.1))

        model.add(LSTM(self.layer2))
        model.add(Dropout(0.1))

        model.add(Dense(self.target_width))
        self.model = model
        
    def compile_and_fit(self, data_window, epochs, batch_size, verbose, learning_rate, loss_weights=None, patience=4):
        if loss_weights is None:
            if isinstance(self.loss, list):
                loss_weights = [1 for _ in range(len(self.loss))]
            else:
                loss_weights = [1]

        self.model.compile(loss=self.loss,
                           loss_weights=loss_weights,
                           optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                           metrics = ['MAE']
                           )
        X, Y = data_window.train
        Xval, Yval = data_window.val
        batch_size = batch_size[0] if isinstance(batch_size, tuple) else batch_size
        self.history = self.model.fit(X, Y,
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      validation_data=(Xval, Yval), 
                                      callbacks=[EarlyStopping(monitor='val_loss', 
                                                               patience=patience, 
                                                               mode='min',  # training will stop when loss stopped decreasing
                                                               restore_best_weights=True #restore model weights from the epoch with the best loss
                                                               )
                                                 ],
                                      verbose=verbose,
                                      shuffle=False)
        return self.history