# Quantitative Trading Strategies

## Introduction

Quantitative trading involves using mathematical models, statistical analysis, and computational algorithms to identify trading opportunities and execute trades. These strategies rely on data analysis rather than human judgment, enabling systematic and emotionless decision-making.

## Fundamental Concepts

### The Quantitative Trading Process

1. **Strategy Development**: Design hypothesis and trading rules
2. **Backtesting**: Test strategy on historical data
3. **Optimization**: Fine-tune parameters
4. **Validation**: Out-of-sample testing
5. **Implementation**: Live trading with risk controls
6. **Monitoring**: Track performance and adapt

### Key Performance Metrics

**Return Metrics:**
- Total Return
- Annualized Return
- Compound Annual Growth Rate (CAGR)

**Risk Metrics:**
- Volatility (Standard Deviation)
- Maximum Drawdown
- Value at Risk (VaR)

**Risk-Adjusted Metrics:**
- **Sharpe Ratio**: (Return - Risk-Free Rate) / Volatility
  - > 1: Good
  - > 2: Very good
  - > 3: Excellent
- **Sortino Ratio**: Like Sharpe but uses downside deviation
- **Calmar Ratio**: CAGR / Maximum Drawdown

**Trading Metrics:**
- Win Rate: Percentage of profitable trades
- Profit Factor: Gross Profit / Gross Loss
- Average Trade Return
- Maximum Consecutive Losses

## Mean Reversion Strategies

### Concept

Mean reversion assumes that prices tend to return to their historical average over time.

**Statistical Basis:**
- Prices deviate from mean due to temporary factors
- Over time, deviations correct themselves
- Works best in range-bound markets

### Pairs Trading

Trading two correlated securities.

**Process:**
1. Identify highly correlated pairs (e.g., Coca-Cola and PepsiCo)
2. Calculate the spread (difference in prices)
3. When spread widens beyond threshold:
   - Long the underperforming asset
   - Short the outperforming asset
4. Close position when spread reverts to mean

**Statistical Approach:**
```
Spread = log(Price_A) - β × log(Price_B)
Z-Score = (Current_Spread - Mean_Spread) / Std_Spread
```

**Entry Signal:**
- Z-Score > +2: Short spread (long B, short A)
- Z-Score < -2: Long spread (long A, short B)

**Exit Signal:**
- Z-Score crosses 0

**Advantages:**
- Market-neutral (hedged)
- Lower risk than directional trading
- Works in various market conditions

**Challenges:**
- Correlations can break down
- Requires careful pair selection
- Transaction costs can erode profits

### Bollinger Band Mean Reversion

Using Bollinger Bands to identify overbought/oversold conditions.

**Bollinger Bands:**
- Middle Band: 20-day Simple Moving Average
- Upper Band: Middle Band + (2 × Standard Deviation)
- Lower Band: Middle Band - (2 × Standard Deviation)

**Trading Rules:**
- Buy when price touches lower band
- Sell when price touches upper band
- Exit at middle band or opposite band

**Enhancements:**
- Confirm with RSI or other indicators
- Use volume for confirmation
- Adjust band width for volatility

### Statistical Arbitrage (Stat Arb)

Portfolio approach to mean reversion.

**Methodology:**
1. Create a basket of long and short positions
2. Use factor models to identify mispricings
3. Exploit temporary deviations from fair value
4. Portfolio is typically beta-neutral

**Common Approaches:**
- Residual returns arbitrage
- Factor-based arbitrage
- Index arbitrage

## Momentum Strategies

### Concept

Momentum assumes that securities that have performed well (poorly) in the recent past will continue to perform well (poorly) in the near future.

**Behavioral Basis:**
- Herding behavior
- Underreaction to news
- Slow information diffusion

### Time-Series Momentum

Exploiting an asset's own past returns.

**Simple Moving Average Crossover:**
- **Golden Cross**: Short-term MA crosses above long-term MA → Buy
- **Death Cross**: Short-term MA crosses below long-term MA → Sell

**Popular Combinations:**
- 50-day / 200-day (long-term)
- 20-day / 50-day (medium-term)
- 5-day / 20-day (short-term)

**Advantages:**
- Simple to implement
- Works across asset classes
- Captures trends effectively

**Limitations:**
- Lagging indicator
- Whipsaws in choppy markets
- Misses early trend reversals

### Cross-Sectional Momentum

Ranking securities by relative performance.

**Process:**
1. Calculate returns over lookback period (e.g., 3-12 months)
2. Rank securities by returns
3. Long top quintile (winners)
4. Short bottom quintile (losers)
5. Rebalance periodically (monthly)

**Refinements:**
- Skip most recent month (avoid reversals)
- Weight by momentum strength
- Adjust for volatility
- Filter by liquidity

**Example Portfolio:**
- Top 20% by 6-month return → Long
- Bottom 20% by 6-month return → Short
- Equal weight or momentum-weighted
- Monthly rebalancing

### Breakout Strategies

Trading price breakouts from established ranges.

**Channel Breakout:**
- Track highest high and lowest low over N days
- Buy on break above highest high
- Sell on break below lowest low
- Popular parameter: 20-day or 55-day channels

**Support/Resistance Breakout:**
- Identify key price levels
- Enter on break with volume confirmation
- Use stops below/above breakout level

**Volatility Breakout:**
- Buy when price moves beyond X standard deviations
- Combines trend and volatility signals

## Arbitrage Strategies

### Definition

Exploiting price discrepancies between related securities or markets.

### Index Arbitrage

Trading discrepancies between index and its components.

**Example:**
- S&P 500 futures price vs. basket of 500 stocks
- Buy underpriced side, sell overpriced side
- Profit when prices converge

**Requirements:**
- Fast execution
- Low transaction costs
- Sophisticated infrastructure

### Merger Arbitrage

Trading announced mergers and acquisitions.

**Cash Merger:**
- Company A to be acquired by Company B for $50/share
- Company A trading at $48/share (2% spread)
- Buy Company A
- Profit when deal closes and price goes to $50

**Stock Merger:**
- Company A to receive 0.5 shares of Company B
- Expected value: $50
- Current price: $48
- Long Company A, Short 0.5 shares of Company B

**Risk Factors:**
- Deal may not close
- Deal terms may change
- Timing uncertainty
- Regulatory approvals

### Convertible Arbitrage

Trading convertible bonds and underlying stocks.

**Strategy:**
- Buy undervalued convertible bond
- Short corresponding amount of stock
- Profit from:
  - Mispricing correction
  - Bond yield
  - Volatility changes

**Risk Management:**
- Delta hedging: Adjust short position as stock price moves
- Monitor credit risk of bond issuer

## Factor-Based Strategies

### Factor Investing Framework

Systematic exposure to sources of return.

**Common Factors:**

1. **Value**: 
   - Cheap stocks outperform expensive ones
   - Metrics: P/E, P/B, P/S ratios
   
2. **Momentum**: 
   - Recent winners continue winning
   - Metrics: 3-12 month returns

3. **Quality**:
   - Profitable, stable companies outperform
   - Metrics: ROE, earnings stability, low debt

4. **Size**:
   - Small-cap premium over large-cap
   - Metrics: Market capitalization

5. **Low Volatility**:
   - Low-risk stocks outperform high-risk
   - Metrics: Standard deviation, beta

6. **Dividend Yield**:
   - High dividend stocks outperform
   - Metrics: Dividend yield, payout ratio

### Multi-Factor Models

Combining multiple factors for diversification.

**Approaches:**

1. **Sequential Screening**:
   - Filter by Value (top 50%)
   - Then by Momentum (top 50% of remaining)
   - Then by Quality (top 50% of remaining)

2. **Composite Scoring**:
   - Assign scores for each factor
   - Weight factors (e.g., 40% Value, 30% Momentum, 30% Quality)
   - Rank by composite score

3. **Optimization**:
   - Maximize factor exposures
   - Subject to constraints (sector limits, tracking error)

**Example:**
```
Score = 0.4 × Value_Z-Score + 0.3 × Momentum_Z-Score + 0.3 × Quality_Z-Score
```

### Smart Beta Strategies

Rules-based strategies that target specific factors.

**Common Smart Beta Indices:**
- Equal-weight indices (reduce concentration)
- Fundamental-weight (weight by revenue, dividends)
- Minimum variance (reduce portfolio volatility)
- Risk parity (equal risk contribution)

## Machine Learning in Trading

### Supervised Learning

Predicting future returns or price movements.

**Algorithms:**
- Linear Regression
- Logistic Regression (classification)
- Random Forests
- Gradient Boosting Machines
- Neural Networks

**Feature Examples:**
- Technical indicators (RSI, MACD, Bollinger Bands)
- Price patterns
- Volume metrics
- Sentiment scores
- Fundamental ratios

**Challenges:**
- Overfitting to noise
- Non-stationarity of markets
- Limited training data
- Regime changes

**Best Practices:**
- Walk-forward validation
- Regularization techniques
- Ensemble methods
- Out-of-sample testing

### Unsupervised Learning

Discovering patterns without labeled data.

**Applications:**

1. **Clustering**:
   - Group similar stocks
   - Identify market regimes
   - Portfolio construction

2. **Dimensionality Reduction**:
   - PCA for feature reduction
   - Identify dominant factors
   - Noise reduction

3. **Anomaly Detection**:
   - Identify unusual market behavior
   - Detect trading opportunities
   - Risk monitoring

### Reinforcement Learning

Learning optimal trading policies through trial and error.

**Advantages:**
- Learns from market feedback
- Adapts to changing conditions
- Can handle complex state spaces

**Popular Algorithms:**
- Q-Learning
- Deep Q-Networks (DQN)
- Policy Gradient Methods
- Actor-Critic Methods

**Challenges:**
- Sample efficiency
- Exploration vs. exploitation
- Reward function design
- Non-stationarity

## High-Frequency Trading (HFT)

### Characteristics

- Extremely short holding periods (seconds to microseconds)
- Large number of trades
- Low profit per trade
- Requires sophisticated technology

### Common HFT Strategies

**Market Making:**
- Provide liquidity by posting bid and ask quotes
- Profit from bid-ask spread
- High volume, low margin per trade

**Latency Arbitrage:**
- Exploit price differences due to information delays
- Requires fastest connections
- Co-location in exchange data centers

**Statistical Arbitrage:**
- High-frequency version of stat arb
- Very short-term mean reversion
- Thousands of trades per day

**Momentum Ignition:**
- Initiate rapid buying/selling to trigger momentum
- Controversial and may be considered manipulation

### Technology Requirements

- Ultra-low latency connections
- Co-location services
- Custom hardware (FPGAs)
- Direct market access
- Sophisticated order management systems

## Risk Management in Quantitative Trading

### Position Sizing

**Fixed Fractional Method:**
```
Position_Size = (Account_Equity × Risk_Percentage) / (Entry_Price × Stop_Loss_Percentage)
```

**Kelly Criterion:**
```
f* = (p × b - q) / b
```
Where:
- f* = fraction of capital to bet
- p = probability of winning
- q = probability of losing
- b = win/loss ratio

### Portfolio Risk Controls

1. **Concentration Limits**: Maximum per position
2. **Sector Limits**: Maximum per industry
3. **Leverage Limits**: Maximum total leverage
4. **Correlation Monitoring**: Avoid hidden concentrations
5. **VaR Limits**: Maximum Value at Risk

### Strategy-Level Risk Management

**Drawdown Controls:**
- Reduce position sizes during drawdowns
- Stop trading if drawdown exceeds threshold
- Gradually increase sizing as equity recovers

**Volatility Adjustment:**
- Scale positions inversely with volatility
- Maintain consistent risk level

**Correlation Monitoring:**
- Ensure strategies are uncorrelated
- Avoid crowding in same trades

## Backtesting Best Practices

### Data Quality

- Adjust for corporate actions (splits, dividends)
- Survivor-ship bias avoidance
- Point-in-time data (no look-ahead bias)
- Accurate pricing (bid-ask spreads)

### Transaction Costs

Include realistic costs:
- Commissions
- Slippage
- Market impact
- Borrowing costs for shorts
- Financing costs for leverage

### Walk-Forward Analysis

1. **In-Sample Period**: Optimize parameters
2. **Out-of-Sample Period**: Test with optimized parameters
3. Roll forward and repeat
4. Aggregate out-of-sample results

**Benefits:**
- More realistic performance estimates
- Reduces overfitting
- Tests adaptability

### Statistical Significance

Test if results are statistically significant:
- **Monte Carlo Simulation**: Randomize trade order
- **Bootstrap Methods**: Resample returns
- **Permutation Tests**: Randomize entry signals

**T-Statistic for Sharpe Ratio:**
```
t = (Sharpe_Ratio × √N)
```
Where N = number of observations

Need t > 2 for 95% confidence

## Common Pitfalls

1. **Overfitting**: Too many parameters, perfect historical fit
2. **Look-Ahead Bias**: Using future information
3. **Survivorship Bias**: Only testing on companies that survived
4. **Data Snooping**: Testing many strategies, reporting only best
5. **Ignoring Costs**: Unrealistic profit estimates
6. **Regime Changes**: Strategy works in one period, fails in another
7. **Lack of Diversification**: All eggs in one strategy
8. **Execution Assumptions**: Can't actually trade at theoretical prices

## Conclusion

Successful quantitative trading requires:
1. **Robust Strategy Development**: Based on sound economic rationale
2. **Rigorous Backtesting**: With realistic assumptions
3. **Strict Risk Management**: Position sizing and portfolio controls
4. **Continuous Monitoring**: Adapt to changing markets
5. **Diversification**: Multiple uncorrelated strategies
6. **Technology**: Reliable execution systems

The edge in quantitative trading comes not from secret formulas, but from disciplined implementation, thorough research, and continuous improvement. Markets are competitive, and strategies that work today may not work tomorrow. Adaptability and risk management are key to long-term success.
