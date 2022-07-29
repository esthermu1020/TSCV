# Improving LSTM Performance Using Time Series Cross Validation

[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)

This repository is meant to assist readers to understand the idea behind the blog better by running the experimental code on themselves. 

<br/>

<details open>
<summary>
 <strong>✨ Descriptions</strong>
</summary>
  
<br/>
  
  We have implemented a LSTM (Long Short-term Memory)-based forecasting model to predict electricity demand in California 24 hours (one day) in advance, and attempted to further improve the prediction performance. The blog presented an important and powerful method: **Time series cross validation (TSCV)**, including ***forward chaining TSCV*** and ***block chaining TSCV***. They are trained to compare with the result obtained from the baseline model selection technique: three-way holdout validation.
  
  The flow of our experiment is shown below:
  
 <img src="https://github.com/esthermu1020/blog_examples/blob/main/images/base/myflow.png" width="700" height="400" />


 <br/>
 </details>

<details open>
<summary>
 <strong>⚓️ Model</strong>
</summary>
  
<br/>

   Our DL model architecture consists of one input layer, 3 LSTM layers, 3 dropout layers, and 1 output layer, as shown below. Although there are important  considerations to take into account when choosing a particular architecture, we focused only on experimenting with TSCV methods using a fixed LSTM model architecture, and will discuss model architecture and hyperparameter tuning in a separate blog. 

   On the right hand size, we also presented the way we did for data feeding. We used 12 hours historic data (`lookback_width=12`) to predict one-hour electric load (`target_width=1`) of 24 hours ahead (`prediction_offset=24`).

<img src="https://github.com/esthermu1020/blog_examples/blob/main/images/base/basemodel_architecture.png" width="700" height="400" />

</details>

 <br/>
 
<details open>
<summary>
 <strong> ☕️ Repository Directory</strong>
</summary>
 
 <br/>
 
 The picture below shows the structure of the `BlogHelper` package. Make sure you have installed [ ALL the dependencies](https://github.com/esthermu1020/blog_examples/blob/main/BlogHelper/Dependencies.py) as required.
 
 <img src="https://github.com/esthermu1020/blog_examples/blob/main/images/dir.png" width="500" height="600" />
 
 <br/>
 
 The experimental result of each validation method are stored at:
 
 - Link to the baseline model [ 🌾Three way Holdout](https://github.com/esthermu1020/blog_examples/tree/main/images/3wayholdout)
 
 - Link to [ ⚡️Forward Chaining CV](https://github.com/esthermu1020/blog_examples/tree/main/images/forwardchaining)
 
 - Link to [ 💫Block Chaining CV ](https://github.com/esthermu1020/blog_examples/tree/main/images/blockchaining)

 </details>
 <br/>
 
 Compare the model performances in terms of variation of modeling KPIs and residuals at
 
  - Link to [ 🌪Algorithm comparisons](https://github.com/esthermu1020/blog_examples/tree/main/images/comparison)
  
   <br/>
   
  <details open>
<summary>
 <strong> 🚀 How To Repeat the Experiment?</strong>
</summary>
 
 <br/>
 
 1. clone `BlogHelper` to your google drive
 
 - Note: You could simply run `!git clone https://github.com/esthermu1020/blog_examples.git` in your colab code chunk
 
 2. Make sure all packages mentioned in `./blog_examples/BlogHelper/dependencies.py` are installed via [pip](https://pip.pypa.io/en/stable/user_guide/)
 
 3. run experiments follow the instructions in `Blog_to_share.ipynb`
 
  </details>
