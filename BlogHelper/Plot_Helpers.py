from Blog.BlogHelper.Dependencies import *

def plotly_plot_dataset(dataset, write_file=False, path = '/content/drive/MyDrive/Colab Notebooks/Blog/output/input_data.html'):
      # Create figure
      fig = make_subplots(rows=3, cols=1)
      # Set title
      fig.update_layout(
                          title_text="Evolution of Actual and Caiso's Predicted Electric Demand (MW)"
                      )
      # set color
      colors = px.colors.qualitative.Bold.copy()
      fig.add_trace(
                            go.Scatter(x=list(dataset.index), 
                                      y=list(dataset['load_mw_actual']),
                                      line_width=1,
                                      line_color=colors[2],
                                      name = 'Actual Load (MW)'),
                            row=1, col=1)
              
      fig.add_trace(
                            go.Scatter(x=list(dataset.index), 
                                      y=list(dataset['load_mw_dam']),
                                      line_width=1,
                                      line_color=colors[1],
                                      name = 'Demand DAM (MW)'),
                            row=2, col=1)
              
      fig.add_trace(
                            go.Scatter(x=list(dataset.index), 
                                      y=list(dataset['load_mw_rtpd']),
                                      line_width=1,
                                      line_color=colors[3],
                                      name = 'Demand RTPD (MW)'),
                            row=3, col=1)
              
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
                                  visible=False
                              ),
                              type="date"
                          ),
                          xaxis2=dict(
                                rangeslider=dict(visible=False),
                                type="date",
                            ),
                          xaxis3=dict(
                                rangeslider=dict(visible=False),
                                type="date",
                            ),
                      )
      fig.update_xaxes(matches='x')
      if write_file:
        print(f'white file to {path}')
        fig.write_html(path)
      fig.show()
      return fig
      

def draw_summary_table_data_preview(dataset, width=1200, height=570):
    # draw the summary statistics table
    fig = go.Figure(data=[go.Table(
        columnwidth = [250, 100],
        header=dict(values=list(dataset.describe().T.reset_index().columns),
                    fill_color='paleturquoise',
                    align='center'),
        cells=dict(values=round(dataset.describe().T.reset_index(),2).T.values,
                  fill_color=['paleturquoise'] +['lightcyan', '#e7fafa' ]*4,
                  line_color='white',
                  align=['left','center']))
    ])

    fig.update_layout(width=1200, height=570)
    fig.show()
    return fig


def draw_stats_table(ddd, width = 900, height = 300, fill_color1= 'paleturquoise',fill_color2= ['paleturquoise'] +['#e7fafa']*4, line_color = 'white'):
      fig = go.Figure(data=[go.Table(
          header=dict(values=list(ddd.reset_index().columns),
                      fill_color=fill_color1,
                      line_color='white',
                      align='center'),
          cells=dict(values=round(ddd.reset_index(),5).T.values,
                    fill_color=fill_color2,
                    line_color= line_color,
                    align=['left','center']))
      ])
      fig.update_layout(width=width, height=height)
      fig.show()
      return fig


def plotly_plot_variance(tmp_train, tmp_val, tmp_test):
    import seaborn as sns
    mae_train, mae_val, mae_test = np.array(tmp_train), np.array(tmp_val), np.array(tmp_test)
    num_trials = len(mae_train)
    MAEs = pd.concat([
        pd.DataFrame({'MAE': mae_train, 'type': 'train'}),
        pd.DataFrame({'MAE': mae_val, 'type': 'val'}),
        pd.DataFrame({'MAE': mae_test, 'type': 'test'})
    ])
    sns.set(rc={'figure.figsize':(10,4),
                'axes.facecolor':'#e7fafa', 
                'figure.facecolor':'lightcyan'})
    # sns.set_style("whitegrid")
    stdmax = max(np.std(mae_train), np.std(mae_val), np.std(mae_test))
    ax = sns.violinplot(x="type", y="MAE", data=MAEs)
    ax.set(title=f'Train & Val & Test MAE for {num_trials} trials')
    plt.text(-0.3, mae_train.mean() + 3*stdmax, f"Train MAE : {round(mae_train.mean(),2)} \n  (std={round(np.std(mae_train),2)})", 
                horizontalalignment='left', size='medium', color='red', weight='semibold')
    plt.text(0.7, mae_val.mean() + 2*stdmax, f"Val MAE : {round(mae_val.mean(),2)} \n  (std={round(np.std(mae_val),2)})",  
                horizontalalignment='left', size='medium', color='red', weight='semibold')
    plt.text(1.6, mae_test.mean() + 2*stdmax, f"Test MAE : {round(mae_test.mean(),2)} \n  (std={round(np.std(mae_test),2)})",  
                horizontalalignment='left', size='medium', color='red', weight='semibold')
    fig = ax.get_figure()
    return fig

def plotly_plot_variance_comparison(tmp_test, tmp_test1, tmp_test2):
    import seaborn as sns
    mae_a, mae_b, mae_c = np.array(tmp_test), np.array(tmp_test1), np.array(tmp_test2)
    num_trials = len(mae_a)
    MAEs = pd.concat([
            pd.DataFrame({'MAE': mae_a, 'type': 'threewayholdout'}),
            pd.DataFrame({'MAE': mae_b, 'type': 'blockchainingcv'}),
            pd.DataFrame({'MAE': mae_c, 'type': 'forwardchainingcv'})
        ])
    sns.set(rc={'figure.figsize':(10,4),
                    'axes.facecolor':'#e7fafa', 
                    'figure.facecolor':'lightcyan'})
    # sns.set_style("whitegrid")
    stdmax = max(np.std(mae_a), np.std(mae_b), np.std(mae_c))
    ax = sns.violinplot(x="type", y="MAE", data=MAEs)
    ax.set(title=f'Distribution of MAEs by algorithms for {num_trials} trials')
    plt.text(-0.3, mae_a.mean() - 5*stdmax, f"Three-way Holdout :\n    mean:{round(mae_a.mean(),2)}", 
                    horizontalalignment='left', size='medium', color='red', weight='semibold')
    plt.text(0.7, mae_b.mean() - 3*stdmax, f"Block Chaining CV : \n    mean:{round(mae_b.mean(),2)}",  
                    horizontalalignment='left', size='medium', color='red', weight='semibold')
    plt.text(1.6, mae_c.mean() + 3* stdmax, f"Forward chaining CV : \n    mean:{round(mae_c.mean(),2)}",  
                    horizontalalignment='left', size='medium', color='red', weight='semibold')
    fig = ax.get_figure()
    return fig