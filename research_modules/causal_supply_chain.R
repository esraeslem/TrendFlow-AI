# Load necessary libraries
# install.packages("ggdag")
# install.packages("ggplot2")
library(ggdag)
library(ggplot2)

# Define the Causal Structure (Directed Acyclic Graph)
# We are modeling:
# 1. Influencer Posts (Trend) -> distinct from Organic Demand
# 2. Seasonality -> Confounder affecting both Supply & Demand
# 3. The Goal: Isolate "Overstock Waste" causes
fashion_dag <- dagify(
  Overstock_Waste ~ Production_Vol + True_Demand,
  Production_Vol ~ Forecast + Seasonality,
  True_Demand ~ Influencer_Buzz + Seasonality,
  Forecast ~ Influencer_Buzz, # The flaw: Forecasting based only on buzz leads to waste
  exposure = "Production_Vol",
  outcome = "Overstock_Waste",
  coords = list(
    x = c(Overstock_Waste = 2, Production_Vol = 1, True_Demand = 3, Forecast = 1, Influencer_Buzz = 2, Seasonality = 2),
    y = c(Overstock_Waste = 1, Production_Vol = 2, True_Demand = 2, Forecast = 3, Influencer_Buzz = 4, Seasonality = 3)
  )
)

# Plotting the Graph with Academic Styling
ggdag(fashion_dag, text = FALSE, use_labels = "name") +
  theme_dag_blank() +
  geom_dag_node(color = "#2c3e50", alpha = 0.8) +
  geom_dag_text(color = "white", size = 4) +
  geom_dag_edges(edge_color = "#95a5a6") +
  ggtitle("Figure 1: Causal DAG of Fashion Supply Chain Waste",
          subtitle = "Identifying Seasonality as a Confounder in Demand Forecasting") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, size = 10))

ggsave("causal_dag_output.png", width = 8, height = 6)
