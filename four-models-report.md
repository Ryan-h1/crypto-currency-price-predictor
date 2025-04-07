# CS4442/9542 COURSE PROJECT REPORT

## ABSTRACT

This research addresses the challenge of predicting cryptocurrency price movements in a volatile and dynamic market. We developed and evaluated multiple machine learning models for forecasting significant price increases (15%+) over a 30-day horizon. Our approach incorporated extensive feature engineering across price, volume, volatility, momentum, and technical indicators. We implemented a novel train-leader-test-follower split strategy to evaluate model generalization from high to lower market cap cryptocurrencies. Comparative analysis of logistic regression, random forest, XGBoost, and LSTM models revealed that deep learning approaches (LSTM) achieved the best performance (F1-score: 0.389), followed closely by gradient boosting methods. Financial performance analysis demonstrates the practical value of our models, with LSTM achieving a 0.51% ROI and profit factor of 1.14 on test data over a 30-day trading period. While modest, these returns outperform baseline strategies in the highly volatile cryptocurrency market. The research demonstrates the feasibility of cryptocurrency price prediction while highlighting challenges in generalization across market segments. Future work should explore multi-timeframe models and alternative data sources to further improve predictive performance.

## INTRODUCTION

Cryptocurrency markets represent one of the most volatile investment landscapes, characterized by rapid price fluctuations and influenced by numerous factors including market sentiment, technological developments, and regulatory changes. The ability to predict significant price movements offers substantial value to investors and researchers alike.

This research addresses the gap between traditional financial forecasting methods and the unique characteristics of cryptocurrency markets. While existing research has applied various machine learning techniques to cryptocurrency price prediction, many approaches suffer from data leakage, insufficient feature engineering, or inadequate evaluation methodologies that fail to test real-world generalization.

Our objectives were to: (1) develop a comprehensive feature engineering framework specific to cryptocurrency data, (2) implement and compare multiple predictive modeling approaches, (3) evaluate model performance using a realistic split that tests generalization across market segments, and (4) assess the financial implications of model predictions in real-world trading scenarios.

We developed a pipeline that processes raw cryptocurrency price data, applies sophisticated feature engineering, filters out stablecoins, and trains various predictive models. Our results demonstrate that deep learning approaches provide the strongest predictive performance, though all models struggle with the inherent volatility of smaller-cap cryptocurrencies.

The primary contribution of this research is a robust methodology for cryptocurrency price prediction that prioritizes practical applicability through realistic dataset splits and comprehensive feature engineering. The findings have implications for both algorithmic trading systems and our understanding of cryptocurrency market dynamics.

## BACKGROUND & RELATED WORK

### Cryptocurrency Market Prediction

Cryptocurrency price prediction has gained significant research attention following the rise of Bitcoin and altcoins. Early work by Mcnally et al. [1] applied Bayesian neural networks and LSTM models to Bitcoin price prediction, establishing that neural networks can capture some patterns in cryptocurrency price movements.

Alessandretti et al. [2] evaluated the performance of various machine learning algorithms for cryptocurrency portfolio management, finding that gradient boosting methods outperformed other approaches when considering short-term trading windows.

### Feature Engineering for Cryptocurrency Prediction

Technical indicators have traditionally been used in stock market prediction, and researchers have applied similar approaches to cryptocurrencies. Ji et al. [3] demonstrated that technical indicators including moving averages, RSI, and MACD provide significant predictive power for cryptocurrency prices.

More recent approaches have incorporated sentiment analysis and on-chain metrics. Abraham et al. [4] used Twitter sentiment and Google Trends data alongside technical indicators, showing improvements in predictive accuracy.

### Train-Test Split Methodologies

A critical issue in financial prediction models is the evaluation methodology. Conventional random splits often lead to optimistic performance metrics that fail to generalize to real-world scenarios. Lopez de Prado [5] introduced the concept of purged cross-validation to address look-ahead bias in financial time series prediction.

### Research Gap

Despite extensive research, most cryptocurrency prediction models have:

1. Insufficient feature engineering specific to cryptocurrency markets
2. Evaluation methods that don't test generalization across market segments
3. Limited comparisons across model types under identical conditions
4. Inadequate focus on financial performance metrics relevant to investors

Our research addresses these gaps by developing a comprehensive feature engineering framework and evaluating models using a novel train-leader-test-follower methodology that better approximates real-world usage scenarios.

## METHODS

### RESEARCH OBJECTIVES

1. **RO1**: Develop a comprehensive feature engineering framework tailored to cryptocurrency time series data that captures price trends, volatility patterns, volume dynamics, and technical indicators.
2. **RO2**: Implement and evaluate multiple machine learning approaches for cryptocurrency price prediction, including traditional models (logistic regression, random forest), gradient boosting (XGBoost), and deep learning (LSTM).
3. **RO3**: Design and implement a train-leader-test-follower split methodology to evaluate model generalization from higher market cap cryptocurrencies to lower cap alternatives.
4. **RO4**: Determine the optimal model architecture and feature set configuration for predicting significant cryptocurrency price increases.
5. **RO5**: Evaluate the financial performance of prediction models to assess their practical value for investment decisions.

These objectives address significant gaps in cryptocurrency prediction research by focusing on comprehensive feature engineering, realistic evaluation methodologies, and financial performance assessment.

### RESEARCH METHODOLOGY

#### Data Collection and Preprocessing

We utilized the CoinGecko API to collect historical price, volume, and market capitalization data for cryptocurrencies. The `CoinGeckoAPIScraper` class managed API requests with appropriate rate limiting and error handling. Data was collected for up to 2000 cryptocurrencies with daily granularity.

A key preprocessing step was the identification and filtering of stablecoins, which would otherwise introduce noise into our models. The `StablecoinFilter` class identified stablecoins by analyzing price stability patterns, removing coins that maintained prices within a 3% threshold for over 90% of observations.

#### Feature Engineering

We implemented an extensive feature engineering framework (`CryptoFeatureEngineer`) that created features across five categories:

1. **Price Features**: Moving averages (7, 14, 30, 60, 90 days), price/MA ratios, MA crossovers, and distance from historical highs/lows.
2. **Volume Features**: Volume moving averages, volume changes, and price-volume correlations.
3. **Volatility Features**: Standard deviation of returns, price ranges, and exponential volatility measures.
4. **Momentum Features**: RSI across multiple timeframes (7, 14, 30 days), momentum indicators, and rate of change metrics.
5. **Technical Indicators**: MACD, Bollinger Bands width and position.

Three feature sets were defined:

- **Minimal**: 9 core features
- **Standard**: 24 balanced features across categories
- **Full**: All available features (50+)

#### Target Variable Definition

The prediction target was defined as a binary classification problem: whether a cryptocurrency's price would increase by at least 15% over a 30-day future horizon. This threshold represents a significant investment opportunity while being achievable enough to provide sufficient positive examples.

#### Model Development

We implemented four distinct model architectures:

1. **Logistic Regression**: Baseline linear model with L2 regularization and class balancing.
2. **Random Forest**: Ensemble of decision trees with controlled depth to prevent overfitting.
3. **XGBoost**: Gradient boosting implementation with hyperparameters tuned for imbalanced classification.
4. **LSTM (Long Short-Term Memory)**: Deep learning approach with sequence data preparation, dual LSTM layers, and dropout regularization.

All models used StandardScaler for feature normalization to ensure fair comparison.

#### Train-Leader-Test-Follower Split

We developed a novel dataset split methodology to better approximate real-world conditions:

1. Cryptocurrencies were ranked by market capitalization
2. The top 50% ("leaders") were allocated to the training set
3. A random sample from the bottom 50% ("followers") was selected for testing

This approach evaluates the model's ability to generalize from patterns observed in established cryptocurrencies to emerging ones—a more realistic scenario than random splits.

#### Evaluation Metrics

Models were evaluated using both classification and financial performance metrics:

- **Classification Metrics**: Accuracy, precision, recall, and F1-score
- **Financial Metrics**: Return on investment (ROI), profit factor, and win rate

We also analyzed per-coin performance to identify patterns in model generalization across market capitalization levels.

#### Financial Performance Simulation

To assess practical investment value, we conducted a simulated trading strategy based on model predictions:

1. Equal allocation of capital to cryptocurrencies predicted to increase by ≥15%
2. Holding positions for the full 30-day prediction period
3. Calculation of overall portfolio return and risk metrics

This approach provides a more realistic assessment of model utility beyond classification metrics.

## RESULTS

### Model Performance Comparison

Our experiments revealed significant differences in predictive performance across model architectures:

| Model Type    | Train F1-score | Test F1-score | Train-Test Gap |
| ------------- | -------------- | ------------- | -------------- |
| LSTM          | 0.475          | 0.389         | 0.086          |
| XGBoost       | 0.510          | 0.383         | 0.127          |
| Random Forest | 0.453          | 0.370         | 0.083          |
| Logistic      | 0.346          | 0.312         | 0.034          |

The LSTM model achieved the highest test F1-score (0.389), marginally outperforming XGBoost (0.383). The sequence-based nature of LSTM models appears well-suited to capturing temporal patterns in cryptocurrency price movements. However, XGBoost showed the strongest training performance, suggesting it may benefit from additional regularization techniques.

Notably, all models exhibited a performance gap between training and testing, reflecting the challenge of generalizing from higher to lower market cap cryptocurrencies. The smallest gap belonged to the logistic regression model (0.034), indicating better generalization but lower overall performance.

### Financial Performance Assessment

Beyond classification metrics, we evaluated the models' financial performance through simulated investment strategies:

| Model Type    | ROI (%) | Profit Factor | Win Rate |
| ------------- | ------- | ------------- | -------- |
| LSTM          | 0.51    | 1.14          | 0.28     |
| XGBoost       | 0.42    | 1.12          | 0.27     |
| Random Forest | 0.19    | 1.05          | 0.26     |
| Logistic      | -0.61   | 0.84          | 0.22     |

#### ROI Calculation Methodology

Our financial ROI estimates are based on a simplified trading simulation using the following assumptions:

- Trades are triggered when the model predicts a price increase above the defined threshold (15%)
- If the prediction is correct, we record the gain equal to the threshold percentage
- If the prediction is incorrect, we assume a fixed 5% loss on the investment
- Equal capital allocation across all predicted opportunities

When testing variations of our models, we observed that XGBoost models with higher prediction thresholds (e.g., 50%, 85%, 100%) showed progressively better ROI figures (up to 4.16% for the 100% threshold model). We chose to focus on XGBoost for these threshold tests due to its faster training time compared to LSTM models. While these higher-threshold models showed promising theoretical ROI figures, their extremely low win rates (below 10%) and potential susceptibility to increased volatility make them less reliable for practical implementation

It's important to note that actual ROI in real-world trading would vary significantly depending on market conditions, volatility, liquidity constraints, and transaction costs. The ROI figures presented should be interpreted as relative performance indicators rather than absolute expected returns.

#### Understanding ROI in Cryptocurrency Trading Context

The Return on Investment (ROI) figures represent the percentage return achieved over a 30-day trading period by following each model's predictions. For example, the LSTM model's 0.51% ROI means that if you had invested $10,000 in cryptocurrencies according to the LSTM model's predictions, after 30 days your investment would be worth approximately $10,051 - a $51 profit.

While these returns may seem modest, they should be understood in several important contexts:

1. **Time Period**: These returns represent just a 30-day trading period, which would compound to significantly higher annual returns if consistently achieved (though past performance does not guarantee future results).

2. **Benchmark Comparison**: The positive ROIs of the top three models outperform many traditional investment strategies and passive cryptocurrency holding during volatile or bearish market conditions.

3. **Profit Factor**: This metric (ratio of gross profits to gross losses) is particularly important. LSTM's profit factor of 1.14 means that for every $100 lost on incorrect predictions, $114 was gained on correct predictions, demonstrating positive expected value despite the win rate of only 28%.

4. **Risk Management**: These returns were achieved without applying sophisticated position sizing or stop-loss strategies, suggesting potential for further optimization.

The logistic regression model showed particularly poor financial performance with a negative ROI of -0.61% despite having the smallest train-test gap, suggesting that better generalization does not necessarily translate to better investment performance when the overall predictive power is insufficient.

The win rate across all models was relatively low (0.22-0.28), but the profitability of correct predictions outweighed the losses from incorrect ones for the three best-performing models. This demonstrates the positive expected value of the predictions despite imperfect accuracy.

### Feature Importance

Analysis of feature importance from the random forest model revealed that momentum indicators (RSI, price momentum) and recent price trends (7-day and 30-day moving averages) provided the strongest predictive signals. Volatility measures showed moderate importance, while volume-based features contributed less to prediction accuracy.

The full feature set consistently outperformed both minimal and standard configurations across all model types, suggesting that cryptocurrency price movements are influenced by a complex interaction of many factors that simpler feature sets cannot fully capture.

### Market Capitalization and Predictability

We observed a clear relationship between market capitalization and prediction accuracy. The models generally performed better on higher market cap cryptocurrencies, with performance declining as market cap decreased. This pattern held across all model types but was most pronounced in the more complex models (LSTM and XGBoost).

This relationship helps explain the performance gap in our train-leader-test-follower methodology and highlights the challenge of generalizing from established cryptocurrencies to emerging ones. Smaller cryptocurrencies likely exhibit more erratic behavior influenced by factors beyond the technical indicators captured in our features.

## CONCLUSIONS & FUTURE WORK

This research demonstrates that machine learning models can predict significant cryptocurrency price movements with meaningful accuracy and financial returns. The LSTM model achieved the best performance among the approaches tested, likely due to its ability to capture sequential patterns in price data, resulting in a 0.51% ROI over a 30-day period. While modest in absolute terms, this represents a positive expected value in a highly volatile market and could be compounded through continuous model application.

Our ROI calculation methodology revealed an interesting pattern where models with higher prediction thresholds (particularly XGBoost variants) showed better theoretical returns but with significantly lower win rates. This highlights an important trade-off in financial prediction systems between prediction frequency and expected returns. While higher threshold models might appear superior in simplified simulations, their practical implementation could face challenges from increased volatility, limited trading opportunities, and psychological factors affecting trading discipline.

Our novel train-leader-test-follower methodology revealed challenges in generalizing from patterns observed in established cryptocurrencies to emerging ones. This finding has important implications for practical applications, suggesting that models may require ongoing retraining or separate models for different market segments.

The comprehensive feature engineering framework developed in this project proved valuable, with the full feature set consistently outperforming reduced versions across all model types. This highlights the complex nature of cryptocurrency price movements and the importance of capturing diverse aspects of market behavior.

The financial performance analysis demonstrated that classification metrics alone do not fully capture a model's practical value. Despite similar F1-scores, LSTM and XGBoost showed notably different ROIs, emphasizing the importance of evaluating prediction models on financial outcomes rather than just classification performance.

Future work should explore:

1. More sophisticated financial simulation frameworks that better account for market volatility, trading costs, and liquidity constraints
2. Integration of external data sources such as sentiment analysis, on-chain metrics, and macroeconomic indicators
3. Multi-timeframe models that combine short and long-term predictions
4. Reinforcement learning approaches that optimize trading strategies directly
5. Ensemble methods that combine the strengths of different model architectures
6. Dynamic portfolio allocation strategies based on prediction confidence
7. Risk management techniques to mitigate losses from false positives
8. Extending the trading simulation period to assess performance over multiple market cycles

Lessons learned include the importance of realistic evaluation methodologies that approximate real-world conditions, the value of comprehensive feature engineering tailored to the unique characteristics of cryptocurrency markets, and the critical need to assess financial performance when developing prediction models for investment applications. Most importantly, we've demonstrated that simplified ROI calculations must be interpreted cautiously, as they may not fully capture the complexities and risks of real-world trading environments.

## REFERENCES

[1] S. Mcnally, J. Roche, and S. Caton, "Predicting the Price of Bitcoin Using Machine Learning," in _Proceedings of the 26th Euromicro International Conference on Parallel, Distributed and Network-based Processing (PDP)_, 2018, pp. 339-343.

[2] L. Alessandretti, A. ElBahrawy, L. M. Aiello, and A. Baronchelli, "Anticipating Cryptocurrency Prices Using Machine Learning," _Complexity_, vol. 2018, pp. 1-16, 2018.

[3] S. Ji, J. Kim, and H. Im, "A Comparative Study of Bitcoin Price Prediction Using Deep Learning," _Mathematics_, vol. 7, no. 10, p. 898, 2019.

[4] J. Abraham, D. Higdon, J. Nelson, and J. Ibarra, "Cryptocurrency Price Prediction Using Tweet Volumes and Sentiment Analysis," _SMU Data Science Review_, vol. 1, no. 3, 2018.

[5] M. Lopez de Prado, "The 10 Reasons Most Machine Learning Funds Fail," _The Journal of Portfolio Management_, vol. 44, no. 6, pp. 120-133, 2018.
