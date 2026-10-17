# Fashion Sales Data

`raw/fashion_sales_data.csv` is a **synthetic** dataset produced by
`python data_generator.py` (implementation: `src/data_processing.py`, seed 42).
`tests/test_data.py` checks that the committed file is exactly what the
generator produces.

## Schema

| Column  | Type     | Description                     |
|---------|----------|---------------------------------|
| Date    | date     | Daily, 2023-01-01 to 2024-12-30 |
| Product | string   | Floral Dress, Leather Jacket, Denim Jeans, Chunky Sneaker |
| Sales   | int      | Units sold that day             |

2,920 rows (730 days × 4 products).

## How demand is generated

| Component | Rule |
|-----------|------|
| Base level | 100 units/day |
| Floral Dress | +150 in April–August |
| Leather Jacket | +200 in October–February |
| Chunky Sneaker | +50 every day (higher baseline) |
| Denim Jeans | no seasonal effect |
| Trend | +15 units/day from 1 Jan 2024 (step, not gradual) |
| Noise | Normal(0, 20), truncated at 0, floored to an integer |

There is no weekly pattern, no holidays, no promotions and no stockouts, so
it is a clean test bed rather than a realistic one.
