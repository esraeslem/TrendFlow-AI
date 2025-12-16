mkdir -p data/raw data/processed

cat > data/README.md << 'EOF'
# Fashion Sales Data

## Overview
Synthetic dataset simulating fashion retail sales for multiple product categories.

## Files
- `fashion_sales_data.csv` - Main synthetic dataset

## Schema
| Column | Type | Description |
|--------|------|-------------|
| Date | datetime | Sales date (daily frequency) |
| Product | string | Product category name |
| Sales | int | Units sold per day |

## Data Generation
Created using synthetic data generator with:
- **Seasonality**: Summer peak for dresses, winter for jackets
- **Trend**: 3% annual growth
- **Noise**: ±20% random variation

## Statistics
- **Date range**: 2024-01-01 to 2024-12-31
- **Products**: 4 categories (Floral Dress, Leather Jacket, Denim Jeans, Chunky Sneaker)
- **Total records**: ~1,460 (365 days × 4 products)

## Usage
```python
import pandas as pd
df = pd.read_csv('data/fashion_sales_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
```
EOF

git add data/README.md
git commit -m "docs: add data documentation"
