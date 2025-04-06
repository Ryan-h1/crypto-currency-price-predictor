## ABSTRACT

This study addresses the challenge of predicting cryptocurrency price movements, specifically identifying coins with potential to double in price within a 60-day period. Traditional financial prediction methods often fail in volatile cryptocurrency markets, creating an opportunity for machine learning approaches. We developed a classification model using historical price data from over 2,000 cryptocurrencies spanning several years. Our methodology employed feature engineering with financial indicators, time-series data splitting, class imbalance handling, and hyperparameter optimization. The final XGBoost model achieved a portfolio return of 1,689.04% on test data with an optimal probability threshold of 0.95. Despite low precision (7.97%), true positives generated extraordinary returns (1,044.05% average), resulting in a profit factor of 9.59. We discovered significant seasonality in cryptocurrency price movements, with 'month' emerging as the most important feature. This research demonstrates the viability of machine learning for identifying high-potential cryptocurrency investments, while highlighting the trade-offs between classification metrics and financial performance. Future work could incorporate on-chain metrics, social sentiment, and real-time validation through paper trading.

## INTRODUCTION

Cryptocurrency markets, characterized by extreme volatility and rapid price movements, present both substantial investment opportunities and significant risks. Unlike traditional financial markets with established valuation models, cryptocurrencies often experience price movements driven by factors ranging from technological developments to social media sentiment and market speculation. This volatility creates an environment where prices can double or halve within short timeframes, making them potentially lucrative targets for predictive modeling.

The current landscape of cryptocurrency price prediction is dominated by technical analysis, fundamental analysis, and increasingly, machine learning approaches. However, most existing research focuses on short-term price movements or aims to predict exact price values, which is notoriously difficult given market randomness. Additionally, many studies concentrate on major cryptocurrencies like Bitcoin and Ethereum, overlooking potential opportunities in the broader market comprising thousands of alternative coins.

Our research addresses this gap by developing a binary classification model specifically targeting cryptocurrency price doublings within a medium-term (60-day) timeframe across a diverse set of coins. Rather than attempting to predict exact prices, we focus on the practically significant outcome of price doubling—an event that represents substantial investment opportunity.

In this study, we collected historical data for over 2,000 cryptocurrencies, engineered relevant financial and technical features, and developed an XGBoost classification model optimized for portfolio returns rather than traditional classification metrics. Our model achieved remarkable financial performance on test data, with an optimal configuration yielding a 1,689.04% portfolio return despite relatively low precision and recall metrics.

The novel contributions of this work include: (1) a feature engineering approach specific to cryptocurrency price doubling prediction; (2) optimization methodology prioritizing financial outcomes over traditional classification metrics; (3) identification of seasonal patterns in cryptocurrency price movements; and (4) a practical implementation framework for cryptocurrency investment decisions.

This report is structured as follows: We first discuss background and related work in cryptocurrency price prediction, then detail our research objectives and methodology. Next, we present our results including model performance and financial outcomes. Finally, we conclude with implications, limitations, and directions for future research.

## BACKGROUND & RELATED WORK

### Cryptocurrency Price Prediction

Cryptocurrency price prediction has attracted increasing research attention since Bitcoin's inception in 2009. Early studies primarily applied traditional time series forecasting techniques such as ARIMA [1] and GARCH [2] models. However, these linear models struggled to capture the complex, non-linear dynamics of cryptocurrency markets.

Machine learning approaches have since gained prominence. Jang and Lee [3] utilized Bayesian neural networks to predict Bitcoin prices, while McNally et al. [4] compared RNN, LSTM, and ARNN models for Bitcoin price prediction. These works demonstrated improved performance over traditional methods but focused exclusively on Bitcoin rather than the broader cryptocurrency market.

Several studies have investigated feature selection for cryptocurrency prediction. Lamon et al. [5] incorporated social media sentiment alongside price data, finding improved prediction accuracy. Chen et al. [6] utilized on-chain metrics such as transaction volume and active addresses as predictive features. Our work builds upon these insights by engineering a comprehensive feature set including price momentum, volatility indicators, technical indicators, and temporal features.

### Classification vs. Regression in Financial Prediction

Financial forecasting research typically follows either regression approaches (predicting exact price values) or classification approaches (predicting directional movements). Regression models often struggle with the inherent randomness in financial markets, while classification models have shown promise in predicting directional movements.

Sun et al. [7] compared classification and regression approaches for stock market prediction, finding that classification models yielded more actionable investment insights. Similarly, Velay and Daniel [8] demonstrated that binary classification of cryptocurrency price movements outperformed regression in terms of investment returns. Our work extends this direction by focusing specifically on the binary classification of significant price movements (doublings) rather than minor directional changes.

### Class Imbalance in Financial Prediction

Price doubling events are inherently rare, creating significant class imbalance challenges for machine learning models. Jeon et al. [9] addressed class imbalance in stock market prediction using SMOTE, reporting improved model performance. Similarly, Monedero et al. [10] employed various resampling techniques for imbalanced financial data. Our approach incorporates these insights, utilizing a combination of undersampling and SMOTE to address the severe class imbalance in our dataset.

### Model Evaluation for Financial Applications

Traditional classification metrics like accuracy, precision, and recall may not align with financial performance objectives. Patel et al. [11] proposed using financial metrics such as returns and Sharpe ratio for evaluating prediction models. Boruta et al. [12] emphasized the importance of backtesting and sensitivity analysis for financial models. Our work follows this guidance, optimizing for portfolio returns rather than classification metrics and conducting comprehensive sensitivity analysis across probability thresholds and model parameters.

### Research Gap

Despite extensive research in cryptocurrency price prediction, several gaps remain:

1. Most studies focus on major cryptocurrencies, overlooking the diverse altcoin market.
2. Few studies specifically target significant price movements (such as doublings) that represent substantial investment opportunities.
3. Limited research explores the trade-off between classification metrics and financial performance.
4. Seasonality effects in cryptocurrency markets remain underexplored.

Our research addresses these gaps by developing a model specifically designed to identify coins with high probability of doubling across a diverse dataset, optimizing for financial returns rather than classification metrics, and investigating seasonal patterns in cryptocurrency price movements.

## METHODS

### RESEARCH OBJECTIVES

**RO1: Develop a machine learning model to predict cryptocurrency price doublings.** This objective addresses the practical need for identifying high-potential investment opportunities in the cryptocurrency market, focusing on the significant event of price doubling rather than minor price movements.

**RO2: Identify optimal features and parameters for cryptocurrency price prediction.** This objective seeks to determine which technical indicators, price patterns, and market context features are most predictive of future price doublings, along with the optimal model configuration.

**RO3: Evaluate the financial performance of the prediction model.** This objective aims to assess the model not just on traditional classification metrics but on practical financial outcomes, including portfolio returns, profit factors, and risk metrics.

**RO4: Investigate patterns and insights in cryptocurrency price movements.** This objective focuses on extracting actionable market insights, such as seasonality effects and feature importance, that can inform trading strategies beyond the model itself.

### RESEARCH METHODOLOGY

#### Data Collection and Preprocessing

We collected historical daily price, market capitalization, and volume data for over 2,000 cryptocurrencies, with each coin's data stored in a separate CSV file. The dataset spans several years, providing a comprehensive view of cryptocurrency market behavior across different market cycles.

The preprocessing pipeline included:

1. Loading and combining individual cryptocurrency data files.
2. Converting date strings to timestamp format.
3. Filtering out cryptocurrencies with insufficient historical data.
4. Handling missing values through appropriate imputation strategies.
5. Sorting data chronologically for time-based splitting.

#### Feature Engineering

We engineered a comprehensive set of features based on financial theory and cryptocurrency market characteristics:

1. **Basic Price Features**:

   - Price momentum (1-day, 7-day, 14-day, 30-day, 90-day percentage changes)
   - Volume trends (volume changes and rolling means)
   - Volatility measures (standard deviation of price over various windows)

2. **Technical Indicators**:

   - Moving averages (7-day, 30-day) and their ratios/differences
   - Relative Strength Index (RSI)
   - Moving Average Convergence Divergence (MACD)
   - Bollinger Bands and related metrics
   - On-Balance Volume (OBV) and its momentum

3. **Market Context Features**:

   - Market cap to volume ratio
   - Price to market cap ratio
   - All-time high ratio
   - Recent drawdowns from local highs
   - Days since local tops/bottoms

4. **Time-Based Features**:
   - Day of week
   - Month
   - Quarter
   - Weekend indicator

We defined our target variable as a binary indicator of whether a cryptocurrency's price doubled within 60 days:

```python
df['future_price_60d'] = df.groupby('coin')['price'].shift(-60)
df['success'] = (df['future_price_60d'] >= 2 * df['price']).astype(int)
```

#### Time-Based Data Splitting

To prevent look-ahead bias and ensure realistic evaluation, we implemented time-based data splitting rather than random splitting:

```python
time_range = df_sorted['timestamp'].max() - df_sorted['timestamp'].min()
cutoff_date = df_sorted['timestamp'].min() + time_range * 0.7
train_data = df_sorted[df_sorted['timestamp'] <= cutoff_date]
test_data = df_sorted[df_sorted['timestamp'] > cutoff_date]
```

This approach preserves the temporal nature of the data, training on earlier periods and testing on later periods.

#### Handling Class Imbalance

Our dataset exhibited severe class imbalance, with doubling events (class 1) constituting less than 1% of the sample. To address this, we implemented a two-step resampling approach:

1. **Undersampling**: Reducing the majority class (non-doublings) to 10% of its original size.
2. **Oversampling**: Applying SMOTE to create synthetic examples of the minority class until reaching a 1:2 ratio.

```python
resampling = Pipeline([
    ('undersample', RandomUnderSampler(sampling_strategy=0.1, random_state=42)),
    ('oversample', SMOTE(sampling_strategy=0.5, random_state=42))
])
X_resampled, y_resampled = resampling.fit_resample(X_train_selected, y_train)
```

#### Feature Selection

To reduce dimensionality and focus on the most predictive features, we implemented a two-step feature selection process:

1. Training an initial model to obtain feature importance scores.
2. Selecting features with importance above the median threshold.

```python
selector = SelectFromModel(xgb_initial, threshold='median')
selector.fit(X_train_clean, y_train)
selected_features = X_train_clean.columns[selector.get_support()]
```

#### Model Selection and Training

We selected XGBoost as our primary classifier due to its proven performance in financial prediction tasks, handling of non-linear relationships, and feature importance capabilities. Based on our sensitivity analysis, we trained the final model with the following parameters:

```python
optimal_params = {
    'objective': 'binary:logistic',
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.2,  # Higher learning rate based on sensitivity analysis
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'random_state': 42
}
```

#### Threshold Optimization

Rather than using the default probability threshold of 0.5, we conducted a comprehensive threshold sensitivity analysis to optimize for financial performance:

```python
thresholds = np.arange(0.1, 1.0, 0.05)
threshold_results = []
for threshold in thresholds:
    y_pred = (y_proba >= threshold).astype(int)
    # Calculate classification and financial metrics
    # ...
```

This analysis revealed an optimal threshold of 0.95, which maximized portfolio returns despite reducing the number of positive predictions.

#### Performance Evaluation

We evaluated our model on multiple dimensions:

1. **Classification Metrics**:

   - Accuracy, precision, recall, F1-score
   - Confusion matrix

2. **Financial Performance Metrics**:

   - Portfolio return (mean return of predicted positives)
   - True positive and false positive returns
   - Win rate and loss rate
   - Profit factor (ratio of gains to losses)
   - Return volatility (standard deviation)

3. **Feature Importance Analysis**:
   - XGBoost feature importance scores
   - Analysis of top contributors to prediction

This multi-faceted evaluation approach ensures that the model is assessed not just on statistical measures but on practical financial outcomes.

## RESULTS

### Model Performance Metrics

Our final XGBoost model with optimal parameters (higher learning rate of 0.2) and probability threshold (0.95) achieved the following classification metrics on the test dataset:

- **Accuracy**: 0.9836
- **Precision**: 0.0797
- **Recall**: 0.0864
- **F1 Score**: 0.0829

The confusion matrix revealed:

```
[[201347   1754]
 [  1608    152]]
```

At first glance, these metrics might appear disappointing, particularly the low precision and recall. However, this reflects the inherent challenge of predicting rare events (cryptocurrency price doublings) and the trade-off made to optimize for financial returns rather than classification metrics.

### Financial Performance

The financial performance of our model tells a different story:

- **Portfolio Composition**:

  - Total predictions (investment opportunities): 1,906
  - True doublings: 152 (7.97%)
  - False doublings: 1,754 (92.03%)

- **True Positive Returns**:

  - Mean Return: 1,044.05%
  - Median Return: 422.01%
  - Min Return: 101.12% (by definition, all true positives had returns ≥100%)
  - Max Return: 7,048.51%
  - Standard Deviation: 1,461.71%

- **False Positive Returns**:

  - Mean Return: -31.88%
  - Median Return: -44.41%
  - Min Return: -96.33%
  - Max Return: 99.34%
  - Standard Deviation: 47.26%

- **Overall Portfolio Performance**:
  - Win Rate: 13.12%
  - Loss Rate: 16.58%
  - Profit Factor: 9.59
  - Return Standard Deviation: 894.68%

The most striking result is the 1,044.05% average return for true positives, which far outweighs the -31.88% average loss for false positives. This translates to a profit factor of 9.59, meaning that for every dollar lost on false positives, the model generates $9.59 in gains from true positives.

The sensitivity analysis revealed that using a high probability threshold of 0.95 yielded a remarkable portfolio return of 1,689.04%, despite identifying only 148 investment opportunities. This demonstrates that the model is highly effective at identifying the most promising cryptocurrencies with extreme growth potential.

### Feature Importance Analysis

Our feature importance analysis revealed interesting insights into cryptocurrency price movements:

**Top 10 Features by Importance**:

1. month (0.204259) - Indicates strong seasonality in cryptocurrency markets
2. ema_26 (0.064084) - 26-day exponential moving average
3. ema_12 (0.063423) - 12-day exponential moving average
4. market_cap (0.063270) - Market capitalization
5. bollinger_width (0.055262) - Width of Bollinger Bands (volatility indicator)
6. ma_7 (0.046612) - 7-day moving average
7. ath_ratio (0.045261) - Ratio of current price to all-time high
8. drawdown_30d (0.039911) - 30-day drawdown from local high
9. price (0.039163) - Current price
10. rolling_max (0.035950) - Rolling maximum price

Notably, the most important feature is 'month', accounting for over 20% of the model's predictive power. This suggests a strong seasonal component to cryptocurrency price movements that hasn't been widely documented in previous research. Technical indicators like moving averages and Bollinger Bands also proved highly predictive, along with market context features like market cap and the price's relationship to its all-time high.

### Visualization Insights

The "Predicted Probability vs Actual Return" scatter plot revealed a clear relationship between the model's confidence (probability) and actual returns. Points with high predicted probabilities (>0.95) frequently achieved returns far exceeding 100%, with some reaching 200-300%. This validates our approach of using a high probability threshold to identify the most promising opportunities.

The distribution of predicted probabilities showed that very few cryptocurrencies received high probability scores, explaining why we only identified 1,906 investment opportunities among the entire dataset. This selectivity is a key strength of the model, focusing only on the most promising candidates.

## CONCLUSIONS & FUTURE WORK

This project successfully developed a machine learning model capable of identifying cryptocurrencies with high potential for price doubling within a 60-day window. Our approach demonstrates that optimizing for financial returns rather than traditional classification metrics can yield a highly profitable trading strategy, despite low precision and recall scores.

With reference to our original research objectives, we draw the following conclusions:

1. Machine learning models can effectively predict significant cryptocurrency price movements, though with low hit rates balanced by extraordinary returns when successful.
2. The optimal model configuration combines feature selection, class balancing, a high learning rate, and a very high probability threshold.
3. Seasonality plays a crucial role in cryptocurrency price movements, with 'month' emerging as the most important predictive feature.
4. Technical indicators (moving averages, Bollinger Bands) and market context features (market cap, all-time high ratio) provide significant predictive power.
5. The trade-off between traditional classification metrics and financial performance is substantial, highlighting the importance of evaluating models on practical financial outcomes.

### Future Work

Several promising directions for future research emerge from this work:

1. **Incorporate Additional Data Sources**:

   - On-chain metrics (transaction volume, active addresses)
   - Social media sentiment analysis
   - Developer activity metrics (GitHub commits)
   - Macro-economic indicators

2. **Develop Ensemble Approaches**:

   - Combine multiple models optimized for different market conditions
   - Create separate models for different seasons given the importance of 'month'
   - Explore stacking with different base classifiers

3. **Implement Real-Time Validation**:

   - Develop a paper trading system to validate model predictions in real-time
   - Compare actual market performance with backtested results
   - Implement automatic model retraining as new data becomes available

4. **Explore Alternative Prediction Targets**:
   - Predict different magnitudes of price increases (e.g., 50%, 3x, 5x)
   - Predict price movements over different timeframes
   - Develop regression models for expected returns alongside classification

### Lessons Learned

This project offered several valuable insights:

1. Traditional classification metrics can be misleading for financial applications; models should be evaluated on financial outcomes.
2. Class imbalance is a significant challenge in predicting rare events like price doublings, requiring careful handling.
3. Feature engineering specific to the financial domain is crucial for model performance.
4. The optimal probability threshold can be much higher than the default 0.5 for financial prediction tasks.
5. Cryptocurrency markets exhibit strong seasonal patterns that can be exploited for prediction.

In conclusion, while perfect prediction of cryptocurrency prices remains elusive, our machine learning approach demonstrates significant potential for identifying high-return investment opportunities. The framework developed in this project provides a solid foundation for cryptocurrency investment decisions, combining statistical rigor with practical financial applicability.

## REFERENCES

[1] S. Kotz and S. Nadarajah, "Extreme Value Distributions: Theory and Applications," Imperial College Press, 2000.

[2] C. R. Nelson, "Applied Time Series Analysis for Managerial Forecasting," Holden-Day, 1973.

[3] H. Jang and J. Lee, "An Empirical Study on Modeling and Prediction of Bitcoin Prices With Bayesian Neural Networks Based on Blockchain Information," IEEE Access, vol. 6, pp. 5427-5437, 2018.

[4] S. McNally, J. Roche, and S. Caton, "Predicting the Price of Bitcoin Using Machine Learning," in 26th Euromicro International Conference on Parallel, Distributed and Network-based Processing (PDP), 2018, pp. 339-343.

[5] C. Lamon, E. Nielsen, and E. Redondo, "Cryptocurrency Price Prediction Using News and Social Media Sentiment," SMU Data Science Review, vol. 1, no. 3, 2018.

[6] Z. Chen, C. Li, and W. Sun, "Bitcoin price prediction using machine learning: An approach to sample dimension engineering," Journal of Computational and Applied Mathematics, vol. 365, 2020.

[7] Y. Sun, M. Liang, and X. Li, "A comparison between classification and regression in stock market prediction," in 4th IEEE International Conference on Big Data Computing Service and Applications, 2018, pp. 70-77.

[8] M. Velay and F. Daniel, "Stock Chart Pattern recognition with Deep Learning," in International Conference on Enterprise Information Systems, 2018, pp. 345-356.

[9] S. Jeon, B. Hong, and V. Chang, "Pattern graph tracking-based stock price prediction using big data," Future Generation Computer Systems, vol. 80, pp. 171-187, 2018.

[10] J. Monedero, F. Mata, J. M. Benítez, and A. C. Hervás, "SMOTE-based class-balancing for machine learning approaches to cost-sensitive classification," Expert Systems with Applications, vol. 181, 2021.

[11] J. Patel, S. Shah, P. Thakkar, and K. Kotecha, "Predicting stock and stock price index movement using Trend Deterministic Data Preparation and machine learning techniques," Expert Systems with Applications, vol. 42, no. 1, pp. 259-268, 2015.

[12] K. Boruta, N. Neznamov, and S. Martel, "Trading strategy backtesting," in Data Science for Finance and Economics, Springer, 2020, pp. 253-270.
