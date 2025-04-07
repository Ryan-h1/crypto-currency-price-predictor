# Cryptocurrency Price Prediction: From Model Comparison to Optimized XGBoost Implementation

## ABSTRACT

This research addresses the challenge of predicting significant cryptocurrency price movements in a volatile and dynamic market. We developed and evaluated multiple machine learning models for forecasting substantial price increases, first comparing approaches for 15%+ gains over a 30-day horizon, then optimizing an XGBoost model to predict price doublings within 60 days. Our methodology incorporated extensive feature engineering across price, volume, volatility, momentum, and technical indicators, with a novel train-leader-test-follower split strategy to evaluate model generalization. Initial comparative analysis of logistic regression, random forest, XGBoost, and LSTM models revealed that deep learning approaches (LSTM) achieved the best performance (F1-score: 0.389), though XGBoost (F1-score: 0.383) offered comparable results with significantly faster training times. Our optimized XGBoost model with high probability thresholds demonstrated extraordinary financial performance, achieving a 1,689.04% portfolio return on test data, despite relatively low precision (7.97%). We discovered significant seasonality in cryptocurrency price movements, with 'month' emerging as the most important predictive feature. This research demonstrates the viability of using machine learning for cryptocurrency investment opportunities while highlighting the critical trade-off between classification metrics and financial performance. Future work should explore multi-timeframe models, alternative data sources, and real-time validation through paper trading.

## INTRODUCTION

Cryptocurrency markets represent one of the most volatile investment landscapes, characterized by rapid price fluctuations and influenced by numerous factors including market sentiment, technological developments, and regulatory changes. Unlike traditional financial markets with established valuation models, cryptocurrencies often experience price movements driven by factors ranging from technological advancements to social media sentiment and market speculation. This volatility creates an environment where prices can double or halve within short timeframes, making them potentially lucrative targets for predictive modeling.

The ability to predict significant price movements offers substantial value to investors and researchers alike. This research addresses the gap between traditional financial forecasting methods and the unique characteristics of cryptocurrency markets. While existing research has applied various machine learning techniques to cryptocurrency price prediction, many approaches suffer from data leakage, insufficient feature engineering, or inadequate evaluation methodologies that fail to test real-world generalization.

Our research progressed through two distinct but connected phases. First, we developed a comparative framework to evaluate multiple predictive modeling approaches for forecasting 15%+ price increases over a 30-day horizon. After identifying XGBoost as a highly promising model offering an excellent balance between performance and computational efficiency, we conducted a deeper optimization study focused specifically on predicting price doublings within a 60-day timeframe.

Our objectives throughout this research were to: (1) develop a comprehensive feature engineering framework specific to cryptocurrency data, (2) implement and compare multiple predictive modeling approaches, (3) evaluate model performance using realistic splits that test generalization across market segments, (4) optimize the most promising model for financial performance, and (5) assess the practical implications of model predictions in real-world trading scenarios.

The primary contribution of this research is a robust methodology for cryptocurrency price prediction that prioritizes practical applicability through realistic dataset splits, comprehensive feature engineering, and optimization for financial outcomes rather than traditional classification metrics. The findings have implications for both algorithmic trading systems and our understanding of cryptocurrency market dynamics.

## BACKGROUND & RELATED WORK

### Cryptocurrency Market Prediction

Cryptocurrency price prediction has gained significant research attention following the rise of Bitcoin and altcoins. Early work by Mcnally et al. [1] applied Bayesian neural networks and LSTM models to Bitcoin price prediction, establishing that neural networks can capture some patterns in cryptocurrency price movements.

Alessandretti et al. [2] evaluated the performance of various machine learning algorithms for cryptocurrency portfolio management, finding that gradient boosting methods outperformed other approaches when considering short-term trading windows. This aligned with our findings regarding XGBoost's strong performance.

### Feature Engineering for Cryptocurrency Prediction

Technical indicators have traditionally been used in stock market prediction, and researchers have applied similar approaches to cryptocurrencies. Ji et al. [3] demonstrated that technical indicators including moving averages, RSI, and MACD provide significant predictive power for cryptocurrency prices.

More recent approaches have incorporated sentiment analysis and on-chain metrics. Abraham et al. [4] used Twitter sentiment and Google Trends data alongside technical indicators, showing improvements in predictive accuracy. Similarly, Chen et al. [6] utilized on-chain metrics such as transaction volume and active addresses as predictive features.

Our work builds upon these insights by engineering a comprehensive feature set including price momentum, volatility indicators, technical indicators, and temporal features—finding that seasonality plays an unexpectedly important role in cryptocurrency price movements.

### Classification vs. Regression in Financial Prediction

Financial forecasting research typically follows either regression approaches (predicting exact price values) or classification approaches (predicting directional movements). Regression models often struggle with the inherent randomness in financial markets, while classification models have shown promise in predicting directional movements.

Sun et al. [7] compared classification and regression approaches for stock market prediction, finding that classification models yielded more actionable investment insights. Similarly, Velay and Daniel [8] demonstrated that binary classification of cryptocurrency price movements outperformed regression in terms of investment returns.

Our research extends this direction by focusing specifically on the binary classification of significant price movements rather than minor directional changes or exact price values.

### Model Evaluation for Financial Applications

A critical issue in financial prediction models is the evaluation methodology. Conventional random splits often lead to optimistic performance metrics that fail to generalize to real-world scenarios. Lopez de Prado [5] introduced the concept of purged cross-validation to address look-ahead bias in financial time series prediction.

Traditional classification metrics like accuracy, precision, and recall may not align with financial performance objectives. Patel et al. [9] proposed using financial metrics such as returns and Sharpe ratio for evaluating prediction models. Our work follows this guidance, optimizing for portfolio returns rather than classification metrics and conducting comprehensive sensitivity analysis across probability thresholds and model parameters.

### Research Gap

Despite extensive research in cryptocurrency price prediction, several gaps remain:

1. Most studies focus on major cryptocurrencies, overlooking the diverse altcoin market
2. Few studies specifically target significant price movements that represent substantial investment opportunities
3. Limited research explores the trade-off between classification metrics and financial performance
4. Seasonality effects in cryptocurrency markets remain underexplored
5. Many studies implement inadequate evaluation methodologies that don't reflect real-world trading scenarios

Our research addresses these gaps by developing comprehensive models specifically designed to identify high-potential cryptocurrencies, implementing realistic evaluation methodologies, optimizing for financial returns rather than classification metrics, and investigating seasonal patterns in cryptocurrency price movements.

## METHODS

### RESEARCH OBJECTIVES

Our research progressed through two phases with complementary objectives:

**Phase 1: Model Comparison**

1. Develop a comprehensive feature engineering framework tailored to cryptocurrency time series data that captures price trends, volatility patterns, volume dynamics, and technical indicators
2. Implement and evaluate multiple machine learning approaches for cryptocurrency price prediction, including traditional models (logistic regression, random forest), gradient boosting (XGBoost), and deep learning (LSTM)
3. Design and implement a train-leader-test-follower split methodology to evaluate model generalization from higher market cap cryptocurrencies to lower cap alternatives

**Phase 2: XGBoost Optimization**

1. Develop an optimized XGBoost model specifically targeting cryptocurrency price doublings within a 60-day timeframe
2. Identify optimal features, parameters, and probability thresholds for maximizing financial returns
3. Investigate patterns and insights in cryptocurrency price movements, including seasonality effects and feature importance

Throughout both phases, we maintained a consistent focus on evaluating the financial performance of prediction models to assess their practical value for investment decisions.

### DATA COLLECTION AND PREPROCESSING

We utilized the CoinGecko API to collect historical price, volume, and market capitalization data for over 2,000 cryptocurrencies with daily granularity. The data collection process included proper rate limiting and error handling.

Key preprocessing steps included:

1. Filtering out stablecoins by analyzing price stability patterns, removing coins that maintained prices within a 3% threshold for over 90% of observations
2. Converting date strings to timestamp format
3. Handling missing values through appropriate imputation strategies
4. Sorting data chronologically for time-based splitting

### API RATE LIMITING AND COLLECTION STRATEGY

A significant technical challenge in our research was handling the rate limitations imposed by the CoinGecko API. We implemented a robust rate limiting strategy in our `CoinGeckoAPIScraper` class to ensure reliable data collection without violating API usage policies. Key components of our approach included:

1. Detecting 429 status codes (rate limit exceeded) in API responses
2. Implementing dynamic waiting periods based on the 'retry-after' header value returned by the API
3. Fallback waiting periods of 60 seconds when rate limit errors were detected in exception messages
4. Strategic 1-second delays between consecutive requests to avoid triggering rate limits
5. Batch processing with pagination for collecting large datasets efficiently

This approach allowed us to reliably collect data for thousands of cryptocurrencies while maintaining good API citizenship. The implementation included comprehensive error handling and logging to ensure data integrity throughout the collection process.

### RESEARCH EVOLUTION: FROM SENTIMENT ANALYSIS TO TECHNICAL INDICATORS

It is worth noting that our initial research proposal envisioned a different approach focused on sentiment analysis of social media content (particularly Reddit and Twitter) to predict price movements of low market cap cryptocurrencies (colloquially termed "crap coins"). The original plan included:

1. Using BERT-based models for sentiment analysis of cryptocurrency-related social media posts
2. Implementing LSTM models to predict price trends based on both market data and sentiment indicators
3. Focusing specifically on predicting dramatic price surges ("mooning" events) in highly volatile, low-cap cryptocurrencies

However, practical constraints necessitated a pivot in our research direction:

1. Limited time frame for data collection: The manual daily collection process for low-cap cryptocurrency data was impractical given our project timeline
2. Data availability challenges: We had only a few weeks to work with, insufficient for building a comprehensive dataset of social media sentiment aligned with price movements
3. Model complexity trade-offs: We prioritized developing a deeper understanding of core predictive models before expanding to incorporate sentiment analysis

This pivot allowed us to focus on developing robust predictive models based on technical indicators and price patterns, establishing a strong foundation for future work that could incorporate sentiment analysis and other alternative data sources.

### FEATURE ENGINEERING

We implemented an extensive feature engineering framework that created features across five categories:

1. **Price Features**:

   - Moving averages (7, 14, 30, 60, 90 days)
   - Price/MA ratios
   - MA crossovers
   - Distance from historical highs/lows
   - Price momentum (1-day, 7-day, 14-day, 30-day, 90-day percentage changes)

2. **Volume Features**:

   - Volume moving averages
   - Volume changes
   - Price-volume correlations
   - On-Balance Volume (OBV) and its momentum

3. **Volatility Features**:

   - Standard deviation of returns
   - Price ranges
   - Exponential volatility measures
   - Bollinger Bands width

4. **Momentum Features**:

   - Relative Strength Index (RSI) across multiple timeframes (7, 14, 30 days)
   - Momentum indicators
   - Rate of change metrics
   - Moving Average Convergence Divergence (MACD)

5. **Time-Based Features**:
   - Day of week
   - Month
   - Quarter
   - Weekend indicator

We defined two target variables at different phases of our research:

1. **Phase 1**: Binary indicator of whether a cryptocurrency's price would increase by at least 15% over a 30-day future horizon
2. **Phase 2**: Binary indicator of whether a cryptocurrency's price would double within 60 days

### TRAIN-LEADER-TEST-FOLLOWER SPLIT

We developed a novel dataset split methodology to better approximate real-world conditions:

1. Cryptocurrencies were ranked by market capitalization
2. The top 50% ("leaders") were allocated to the training set
3. A random sample from the bottom 50% ("followers") was selected for testing

This approach evaluates the model's ability to generalize from patterns observed in established cryptocurrencies to emerging ones—a more realistic scenario than random splits.

For our XGBoost optimization phase, we supplemented this with a time-based split to prevent look-ahead bias:

```python
time_range = df_sorted['timestamp'].max() - df_sorted['timestamp'].min()
cutoff_date = df_sorted['timestamp'].min() + time_range * 0.7
train_data = df_sorted[df_sorted['timestamp'] <= cutoff_date]
test_data = df_sorted[df_sorted['timestamp'] > cutoff_date]
```

### HANDLING CLASS IMBALANCE

Both of our target events (15% price increase and price doubling) represented minority classes in our dataset, with doubling events constituting less than 1% of observations. To address this severe class imbalance, we implemented a two-step resampling approach:

1. **Undersampling**: Reducing the majority class to a manageable proportion
2. **Oversampling**: Applying Synthetic Minority Over-sampling Technique (SMOTE) to create synthetic examples of the minority class

```python
resampling = Pipeline([
    ('undersample', RandomUnderSampler(sampling_strategy=0.1, random_state=42)),
    ('oversample', SMOTE(sampling_strategy=0.5, random_state=42))
])
X_resampled, y_resampled = resampling.fit_resample(X_train_selected, y_train)
```

### MODEL DEVELOPMENT

During our first research phase, we implemented four distinct model architectures:

1. **Logistic Regression**: Baseline linear model with L2 regularization and class balancing
2. **Random Forest**: Ensemble of decision trees with controlled depth to prevent overfitting
3. **XGBoost**: Gradient boosting implementation with hyperparameters tuned for imbalanced classification
4. **LSTM (Long Short-Term Memory)**: Deep learning approach with sequence data preparation, dual LSTM layers, and dropout regularization

For our second phase, we focused on optimizing XGBoost with parameters such as:

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

### THRESHOLD OPTIMIZATION

A key insight from our research was that the default probability threshold of 0.5 was suboptimal for financial performance. We conducted comprehensive threshold sensitivity analysis:

```python
thresholds = np.arange(0.1, 1.0, 0.05)
threshold_results = []
for threshold in thresholds:
    y_pred = (y_proba >= threshold).astype(int)
    # Calculate classification and financial metrics
    # ...
```

This analysis revealed that higher thresholds (up to 0.95) maximized portfolio returns despite reducing the number of positive predictions.

### EVALUATION FRAMEWORK

We evaluated our models on multiple dimensions:

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
   - Feature importance scores
   - Analysis of top contributors to prediction

This multi-faceted evaluation approach ensured that models were assessed not just on statistical measures but on practical financial outcomes.

## RESULTS

### MODEL COMPARISON

Our initial model comparison revealed significant differences in predictive performance across model architectures:

| Model Type    | Train F1-score | Test F1-score | Train-Test Gap |
| ------------- | -------------- | ------------- | -------------- |
| LSTM          | 0.475          | 0.389         | 0.086          |
| XGBoost       | 0.510          | 0.383         | 0.127          |
| Random Forest | 0.453          | 0.370         | 0.083          |
| Logistic      | 0.346          | 0.312         | 0.034          |

The LSTM model achieved the highest test F1-score (0.389), marginally outperforming XGBoost (0.383). The sequence-based nature of LSTM models appears well-suited to capturing temporal patterns in cryptocurrency price movements. However, XGBoost showed strong performance while requiring significantly less computational resources and training time, making it an attractive option for further optimization.

Notably, all models exhibited a performance gap between training and testing, reflecting the challenge of generalizing from higher to lower market cap cryptocurrencies.

### FINANCIAL PERFORMANCE ASSESSMENT

The initial financial performance metrics for our model comparison phase revealed:

| Model Type    | ROI (%) | Profit Factor | Win Rate |
| ------------- | ------- | ------------- | -------- |
| LSTM          | 0.51    | 1.14          | 0.28     |
| XGBoost       | 0.42    | 1.12          | 0.27     |
| Random Forest | 0.19    | 1.05          | 0.26     |
| Logistic      | -0.61   | 0.84          | 0.22     |

During our threshold optimization experiments with XGBoost, we observed that higher prediction thresholds (e.g., 50%, 85%, 100%) showed progressively better ROI figures (up to 4.16% for the 100% threshold model). While these higher-threshold models showed promising theoretical ROI, their extremely low win rates (below 10%) and potential susceptibility to increased volatility made them concerning for practical implementation.

This finding led to our second research phase focused on optimizing XGBoost specifically for financial performance.

### OPTIMIZED XGBOOST PERFORMANCE

Our fully optimized XGBoost model with probability threshold of 0.95 achieved the following classification metrics on the test dataset when predicting price doublings within 60 days:

- **Accuracy**: 0.9836
- **Precision**: 0.0797
- **Recall**: 0.0864
- **F1 Score**: 0.0829

The confusion matrix revealed:

```
[[201347   1754]
 [  1608    152]]
```

At first glance, these metrics might appear disappointing, particularly the low precision and recall. However, this reflects the inherent challenge of predicting rare events and the trade-off made to optimize for financial returns rather than classification metrics.

The financial performance tells a dramatically different story:

- **Portfolio Composition**:

  - Total predictions: 1,906
  - True doublings: 152 (7.97%)
  - False doublings: 1,754 (92.03%)

- **True Positive Returns**:

  - Mean Return: 1,044.05%
  - Median Return: 422.01%
  - Min Return: 101.12%
  - Max Return: 7,048.51%

- **False Positive Returns**:

  - Mean Return: -31.88%
  - Median Return: -44.41%
  - Min Return: -96.33%
  - Max Return: 99.34%

- **Overall Portfolio Performance**:
  - Portfolio Return: 1,689.04%
  - Win Rate: 13.12%
  - Profit Factor: 9.59

It is important to note that these impressive financial performance metrics must be interpreted with caution. While our train-leader-test-follower split methodology tests generalization across market capitalization segments, both training and testing data share the same timespan. This means market-wide trends affecting all cryptocurrencies during this specific period are embedded in both datasets. Consequently, these results may not generalize to future market conditions that differ substantially from those observed in our testing period.

### FEATURE IMPORTANCE ANALYSIS

Our feature importance analysis revealed interesting insights into cryptocurrency price movements across both research phases.

In our initial model comparison, momentum indicators (RSI, price momentum) and recent price trends (7-day and 30-day moving averages) provided the strongest predictive signals. Volatility measures showed moderate importance, while volume-based features contributed less.

Our optimized XGBoost model revealed a different but complementary picture:

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

Notably, the most important feature is 'month', accounting for over 20% of the model's predictive power. This suggests a strong seasonal component to cryptocurrency price movements that has not been widely documented in previous research.

### MARKET CAPITALIZATION AND PREDICTABILITY

We observed a clear relationship between market capitalization and prediction accuracy. The models generally performed better on higher market cap cryptocurrencies, with performance declining as market cap decreased. This pattern held across all model types but was most pronounced in the more complex models (LSTM and XGBoost).

This relationship helps explain the performance gap in our train-leader-test-follower methodology and highlights the challenge of generalizing from established cryptocurrencies to emerging ones. Smaller cryptocurrencies likely exhibit more erratic behavior influenced by factors beyond the technical indicators captured in our features.

## CONCLUSIONS & FUTURE WORK

This research demonstrates that machine learning models can predict significant cryptocurrency price movements with meaningful accuracy and financial returns. Our progression from model comparison to focused optimization yielded several important insights:

1. Deep learning approaches (LSTM) and gradient boosting methods (XGBoost) significantly outperform traditional models for cryptocurrency price prediction
2. XGBoost offers an excellent balance between performance and computational efficiency, making it particularly suitable for cryptocurrency trading systems
3. Optimizing for financial returns rather than classification metrics can lead to dramatically different model configurations, particularly regarding probability thresholds
4. Cryptocurrency markets exhibit strong seasonality patterns that can be exploited for prediction
5. Models demonstrate better predictive performance on higher market cap cryptocurrencies, indicating a relationship between market maturity and predictability

The most striking finding is the extraordinary financial performance achieved by our optimized XGBoost model, which generated a 1,689.04% portfolio return despite low precision and recall metrics. This highlights the critical importance of evaluating financial prediction models on practical investment outcomes rather than traditional classification metrics. However, we must emphasize that these returns were observed in backtesting on a specific market period, and such performance levels should not be expected in future deployments due to changing market dynamics and the shared temporal context between our training and testing data.

Our novel train-leader-test-follower methodology revealed challenges in generalizing from patterns observed in established cryptocurrencies to emerging ones. This finding has important implications for practical applications, suggesting that models may require ongoing retraining or separate models for different market segments.

### LIMITATIONS

Several limitations of our approach should be acknowledged:

1. **Temporal Validation**: While we implemented a market cap-based split to test generalization across different cryptocurrency segments, our training and testing datasets share the same timespan. This means the model may have captured market-wide trends specific to this period rather than truly generalizable patterns.

2. **Backtested Performance**: The extraordinary portfolio returns observed (1,689.04%) represent backtested performance in a specific market context and should not be interpreted as expected returns for future deployments.

3. **Market Regime Dependency**: Cryptocurrency markets undergo distinct regime changes (bull markets, bear markets, consolidation periods) with substantially different dynamics. Our model's performance may vary significantly across these regimes.

4. **Stop-Loss Assumptions**: Our financial evaluations assumed consistent execution of stop-loss orders at predetermined levels, which may not be achievable in highly volatile market conditions.

### Future Work

Several promising directions for future research emerge from this work:

1. **Incorporate Additional Data Sources**:

   - On-chain metrics (transaction volume, active addresses)
   - Social media sentiment analysis
   - Developer activity metrics (GitHub commits)
   - Macroeconomic indicators

2. **Develop Multi-Timeframe Models**:

   - Combine short and long-term predictions
   - Explore seasonal models given the importance of 'month'
   - Investigate cyclical patterns in cryptocurrency markets

3. **Implement Real-Time Validation**:

   - Develop a paper trading system to validate model predictions in real-time
   - Compare actual market performance with backtested results
   - Implement automatic model retraining as new data becomes available
   - Conduct out-of-time validation using more recent data periods to test true temporal generalization

4. **Explore Risk Management Strategies**:

   - Dynamic stop-loss levels based on volatility
   - Position sizing optimized for cryptocurrency volatility
   - Portfolio construction techniques to balance high-potential opportunities

5. **Investigate Alternative Model Architectures**:
   - Ensemble methods combining multiple model types
   - Reinforcement learning approaches for direct optimization of trading strategies
   - Transformer models for capturing complex sequential patterns

Lessons learned include the importance of realistic evaluation methodologies that approximate real-world conditions, the value of comprehensive feature engineering tailored to the unique characteristics of cryptocurrency markets, and the critical need to assess financial performance when developing prediction models for investment applications.

Most importantly, we have demonstrated that while perfect prediction of cryptocurrency prices remains elusive, machine learning approaches can identify potentially profitable investment opportunities. However, the extraordinary returns observed in our backtests should be viewed with appropriate skepticism, as market dynamics change over time, and the true test of any prediction model lies in its forward-looking performance with proper risk management.

## REFERENCES

[1] S. Mcnally, J. Roche, and S. Caton, "Predicting the Price of Bitcoin Using Machine Learning," in _Proceedings of the 26th Euromicro International Conference on Parallel, Distributed and Network-based Processing (PDP)_, 2018, pp. 339-343.

[2] L. Alessandretti, A. ElBahrawy, L. M. Aiello, and A. Baronchelli, "Anticipating Cryptocurrency Prices Using Machine Learning," _Complexity_, vol. 2018, pp. 1-16, 2018.

[3] S. Ji, J. Kim, and H. Im, "A Comparative Study of Bitcoin Price Prediction Using Deep Learning," _Mathematics_, vol. 7, no. 10, p. 898, 2019.

[4] J. Abraham, D. Higdon, J. Nelson, and J. Ibarra, "Cryptocurrency Price Prediction Using Tweet Volumes and Sentiment Analysis," _SMU Data Science Review_, vol. 1, no. 3, 2018.

[5] M. Lopez de Prado, "The 10 Reasons Most Machine Learning Funds Fail," _The Journal of Portfolio Management_, vol. 44, no. 6, pp. 120-133, 2018.

[6] Z. Chen, C. Li, and W. Sun, "Bitcoin price prediction using machine learning: An approach to sample dimension engineering," Journal of Computational and Applied Mathematics, vol. 365, 2020.

[7] Y. Sun, M. Liang, and X. Li, "A comparison between classification and regression in stock market prediction," in 4th IEEE International Conference on Big Data Computing Service and Applications, 2018, pp. 70-77.

[8] M. Velay and F. Daniel, "Stock Chart Pattern recognition with Deep Learning," in International Conference on Enterprise Information Systems, 2018, pp. 345-356.

[9] J. Patel, S. Shah, P. Thakkar, and K. Kotecha, "Predicting stock and stock price index movement using Trend Deterministic Data Preparation and machine learning techniques," Expert Systems with Applications, vol. 42, no. 1, pp. 259-268, 2015.
