# TrendFlow: Profit-Aware Demand Forecasting for Fashion Retail

## Abstract (3-4 sentences)
[Your contribution in academic language]

## Key Features
- Time-series forecasting with seasonality detection
- Newsvendor Model optimization
- Interactive decision support dashboard
- 95% confidence interval demand projection

## Installation
[Step-by-step setup instructions]

## Quick Start
def calculate_optimal_order_quantity(
    demand_forecast: float,
    unit_cost: float,
    profit_margin: float,
    confidence_interval: tuple = None
) -> dict:
    """
    Calculate optimal order quantity using Newsvendor Model.
    
    The Newsvendor Model maximizes expected profit by balancing
    overstocking costs (c) against understocking costs (p).
    
    Formula: Q* = F^(-1)(p / (p + c))
    Where F^(-1) is the inverse CDF of demand distribution.
    
    Args:
        demand_forecast: Mean predicted demand from Prophet
        unit_cost: Cost per unit (overstocking penalty)
        profit_margin: Profit per unit sold (understocking penalty)
        confidence_interval: Optional (lower, upper) bounds from Prophet
        
    Returns:
        dict: {
            'optimal_quantity': float,
            'expected_profit': float,
            'service_level': float (critical ratio)
        }
        
    Example:
        >>> calculate_optimal_order_quantity(100, 20, 30)
        {'optimal_quantity': 120, 'expected_profit': 2400, 'service_level': 0.6}
        
    References:
        Schweitzer & Cachon (2000). "Decision Bias in the Newsvendor Problem"
    """
    critical_ratio = profit_margin / (profit_margin + unit_cost)
    
    # If confidence interval provided, use it; else use mean
    if confidence_interval:
        lower, upper = confidence_interval
        # Map critical ratio to demand distribution
        optimal_qty = lower + critical_ratio * (upper - lower)
    else:
        optimal_qty = demand_forecast * critical_ratio
        
    expected_profit = optimal_qty * profit_margin - unit_cost * max(0, optimal_qty - demand_forecast)
    
    return {
        'optimal_quantity': round(optimal_qty),
        'expected_profit': round(expected_profit, 2),
        'service_level': critical_ratio
    }

## Methodology
### Forecasting Component
- Facebook Prophet for time-series
- Handles holidays, seasonality, trends

### Optimization Component  
- Newsvendor Model: Q* = F^(-1)(p/(p+c))
- Where p = profit margin, c = cost
- Maximizes expected profit, not just accuracy

## Results
[Tables showing performance vs baselines]

## Dataset
- Source: Synthetic/Real (specify)
- Size: X samples, Y products, Z months
- Features: sales, social mentions, seasonality

## Citation
[BibTeX format]

## License
MIT License

## Contact
[Your email/LinkedIn]
