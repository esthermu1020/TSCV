from BlogHelper.Dependencies import *
from BlogHelper.Modeling_Helpers import QuantileScaler, BaseModel
from BlogHelper.Windows_Helpers import TSCVGenerator

class experiment:
  ''' 
  This create an experiment class
  '''

  def __init__(self, dataset, lr = 0.00006, BATCH_SIZE = 32, MAX_EPOCHS = 50,
               prediction_offset = 24, lookback_width = 12,clip_delta = 1,
               target_column = ['load_mw_actual'], target_width = 1,  
               showgraph=False, hour_pst = 9, split_ratio = 0.7, iftscv = True,
               verbose=0, patience=5, loss=None,
               one_train_size=72, one_val_size = 12, test_size = 24*30,
               train_from_front = False):
    
      # data parameters
      self.target_column=target_column
      self.dataset=dataset # non-scaled, non-shifted dataset
      self.hour_pst = hour_pst
    
      # model hyperparameters
      self.lr=float(lr)
      self.BATCH_SIZE=BATCH_SIZE
      self.patience=patience
      self.MAX_EPOCHS = MAX_EPOCHS
      self.showgraph=showgraph
      # loss parameters
      self.loss = loss
      self.clip_delta = clip_delta

      # window parameters
      self.lookback_width = lookback_width
      self.target_width = target_width
      self.prediction_offset = prediction_offset
      self.one_train_size = one_train_size
      self.one_val_size = one_val_size
      self.test_size = test_size
      # cv parameter
      self.train_from_front = train_from_front
      self.split_ratio = split_ratio
      self.iftscv=iftscv

      self.create_data_window() # chop data -> data window
      self.calculate_target_stats()
      self.create_model_init() # define the model
      self.experiment(verbose=verbose, showgraph=showgraph)

  def create_data_window(self):
      """
      Create a data window
      """
      # Scaler and Shifter
      rbscaler = QuantileScaler() #(q1=0.2, q2=0.8)
      self.rbscaler=rbscaler

      scaled_data = rbscaler.transform(self.dataset)
      self.scaled_data=scaled_data

      BATCH_SIZE=self.BATCH_SIZE
      hour_pst=self.hour_pst
      data_window = TSCVGenerator(data=self.scaled_data, 
                                          lookback_width=self.lookback_width, 
                                          target_width=self.target_width, 
                                          prediction_offset=self.prediction_offset, 
                                          BATCH_SIZE=BATCH_SIZE,
                                          target_columns=self.target_column,
                                          one_train_size = self.one_train_size,
                                          train_from_front = self.train_from_front,
                                          one_val_size = self.one_val_size,
                                          test_size = self.test_size,
                                          iftscv = self.iftscv,
                                          split_ratio = self.split_ratio
                                        )
      self.data_window = data_window

  def calculate_target_stats(self,verbose=True):
      target_column = self.target_column
      rbscaler = self.rbscaler
      self.numcol = len(self.scaled_data.columns)-1 # number of feature columns

      # recover target value's range, median      
      qrange = rbscaler.df_qrange[target_column[0]]
      self.qrange = qrange
      target_median = rbscaler.df_median[target_column[0]]
      self.target_median = target_median
      if verbose:
        print('qrange:', qrange, 'median:', target_median)

  def create_model_init(self):
      lr = self.lr
      clip_delta = self.clip_delta
      model = BaseModel(look_back=self.lookback_width,target_width=self.target_width,
                            num_features=len(self.scaled_data.columns),
                            loss=[tf.keras.losses.MeanAbsoluteError()],
                            qrange = self.qrange, target_median = self.target_median)
      self.model = model
      if self.showgraph:
        display(tf.keras.utils.plot_model(model.model, show_shapes=True))

  def experiment(self, showgraph=False, verbose=0, quantile=False):
      """
      Run the experiment.
      - Showgraph: If true, plot the model's architecture
      - verbose: If 0 don't show the modelling details, if 1 show all details. Default 0.
      """
      model =  self.model
      lr = self.lr
      from time import time
      start = time()
      model.compile_and_fit(data_window=self.data_window,
                            epochs=self.MAX_EPOCHS,
                            batch_size=self.BATCH_SIZE,
                            verbose=verbose,
                            learning_rate=lr,
                            patience=self.patience)
      delta=time()-start
      print('Total Modelling Time: ',delta)
      self.delta=delta
      self.model=model


  def make_prediction(self, verbose=False, plot=False, end=False):
      """
      Return MAE, RMSE, MAPE, a result table containing pred, and real value.
      """
      qrange = self.qrange
      target_median = self.target_median
      model = self.model

      Xtest, Ytest = self.data_window.test
      y_real = []
      y_real = np.array(Ytest)*qrange+target_median  
      self.y_real=y_real # recover the real y

      # predict value
      y_pred = model.model.predict(Xtest)
      y_pred = np.array(y_pred)*qrange+target_median
      self.y_pred = y_pred

      print(y_real.shape, y_pred.shape)

      # use to handle target_width != 1
      self.y_pred = np.array(y_pred.flatten()).reshape(-1, self.target_width)
      print(y_pred.shape) 

      # print the metrics
      self.MAE = np.mean(np.abs((self.y_pred-self.y_real.reshape(-1,self.target_width))), axis=0)
      if verbose:
        print('MAE', self.MAE)
   
      if self.target_width == 1:
        result_df = pd.DataFrame(
          np.concatenate([self.y_real, self.y_pred], axis = 1), 
          columns = ['real', 'q50']
          )
        result_df.index = self.dataset.iloc[-abs(self.data_window.test_size):].index
        result_df.index.names = ['index']

      else:

        result_df = pd.DataFrame(
          np.concatenate([self.y_real, self.y_pred], axis = 1), 
          columns = ['real', 'q50']
          )
        result_df.index = self.dataset.iloc[-abs(self.data_window.test_size):].index
        result_df.index.names = ['index']

      self.result_df = result_df

      self.fetch_plot_mae() 

      if plot:
         fig, ax = plt.subplots(figsize=(10,5))
         result_df.plot(alpha=0.5,figsize=(20, 5), color = ('red', 'green', 'blue', 'orange'),ax=ax)
         pd.DataFrame(result_df.iloc[:,0]).reset_index().plot(x ='index', y ='real',kind='scatter',s=2,ax=ax, color='red')
         
      return self.MAE, result_df

  def fetch_plot_mae(self):
    MAE_train = np.array(self.model.history.history['MAE']) * self.qrange
    
    self.MAE_train = MAE_train
    MAE_val = np.array(self.model.history.history['val_MAE']) * self.qrange
    self.MAE_val = MAE_val
    Epochs = len(MAE_train)
    MAEs = pd.concat(
        [pd.DataFrame({'MAE': MAE_train, 'type': 'train'}),
        pd.DataFrame({'MAE': MAE_val, 'type': 'val'})]
    )
    import seaborn as sns
    sns.set(rc={'figure.figsize':(10,4)})
    g = sns.violinplot(x="type", y="MAE", data=MAEs) \
          .set(title=f'Train & Val MAE for {Epochs} epochs',
               ylim=(MAEs['MAE'].values.min()-1000, 
                  MAEs['MAE'].values.max()+1000))
    plt.text(-0.2, MAE_train.mean() + np.std(MAE_train), f"Train MAE : {round(MAE_train.mean(),2)} \n  (std={round(np.std(MAE_train),2)})", 
            horizontalalignment='left', size='medium', color='blue', weight='semibold')
    plt.text(0.7, MAE_val.mean() + np.std(MAE_train), f"Val MAE : {round(MAE_val.mean(),2)} \n  (std={round(np.std(MAE_val),2)})",  
            horizontalalignment='left', size='medium', color='blue', weight='semibold')

  
  def plotly_plot_result(self, save=False):    
        result_df = self.result_df.loc[:'2022-06-26 23:00:00']
        # Create figure
        fig = go.Figure()
        # Set title
        fig.update_layout(
                    title_text="Predicted vs Actual electric load (mw)"
                )
        # set color
        colors = px.colors.qualitative.Bold.copy()
        # print(colors)
        real = result_df.columns[0]
        fig.add_trace(
                      go.Scatter(x=list(result_df.index), 
                                y=list(result_df[real]),
                                line_width=1,
                                line_color=colors[0],
                                name = real))

        fig.add_trace(
                      go.Scatter(x=list(result_df.index), 
                                y=list(result_df['q50']),
                                line_color=colors[1],
                                opacity= 0.8,
                                # fill='toself',
                                name = 'q50'))
        fig['data'][0]['showlegend']=False

        fig.add_trace(
                      go.Scatter(x=list(result_df.index), 
                                y=list(result_df[real]),
                                line_width=2,
                                line_color=colors[0],
                                name = real))
        
        
        
        fig.update_xaxes(tickangle=90, tickfont=dict(size=9))
        fig.update_xaxes(showticklabels=True )    #Disable xticks 
        fig.update_xaxes(nticks=40)

        # Add range slider
        fig.update_layout(
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=list([
                                dict(count=10,
                                    label="10d",
                                    step="day",
                                    stepmode="backward"),
                                dict(count=1,
                                    label="1m",
                                    step="month",
                                    stepmode="backward"),
                                dict(count=1,
                                    label="1y",
                                    step="year",
                                    stepmode="todate"),
                                dict(step="all")
                            ])
                        ),
                        rangeslider=dict(
                            visible=True
                        ),
                        type="date"
                    )
                )
        return fig