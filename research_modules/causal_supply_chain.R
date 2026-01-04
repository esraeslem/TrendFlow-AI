# Ensure required packages are installed
required_pkgs <- c("ggdag", "ggplot2")
to_install <- required_pkgs[!(required_pkgs %in% installed.packages()[, "Package"])]
if (length(to_install) > 0) {
  install.packages(to_install, repos = "https://cloud.r-project.org")
}

library(ggdag)
library(ggplot2)

# Make working directory the repository root when run with `Rscript research_modules/causal_supply_chain.R`
args <- commandArgs(trailingOnly = FALSE)
file_arg <- args[grep("--file=", args)]
if (length(file_arg) > 0) {
  script_dir <- dirname(sub("--file=", "", file_arg))
  # go up one level so the repo root is the working directory
  setwd(normalizePath(file.path(script_dir, "..")))
}

# Print working directory for debugging when run non-interactively
message("Working directory: ", getwd())

# Define the Causal Structure (Directed Acyclic Graph)
fashion_dag <- dagify(
  Overstock_Waste ~ Production_Vol + True_Demand,
  Production_Vol ~ Forecast + Seasonality,
  True_Demand ~ Influencer_Buzz + Seasonality,
  Forecast ~ Influencer_Buzz,
  exposure = "Production_Vol",
  outcome = "Overstock_Waste",
  coords = list(
    x = c(Overstock_Waste = 2, Production_Vol = 1, True_Demand = 3, Forecast = 1, Influencer_Buzz = 2, Seasonality = 2),
    y = c(Overstock_Waste = 1, Production_Vol = 2, True_Demand = 2, Forecast = 3, Influencer_Buzz = 4, Seasonality = 3)
  )
)

# Plotting the Graph with Academic Styling
p <- ggdag(fashion_dag, text = FALSE, use_labels = "name") +
  theme_dag_blank() +
  geom_dag_node(color = "#2c3e50", alpha = 0.8) +
  geom_dag_text(color = "white", size = 4) +
  geom_dag_edges(edge_color = "#95a5a6") +
  ggtitle("Figure 1: Causal DAG of Fashion Supply Chain Waste",
          subtitle = "Identifying Seasonality as a Confounder in Demand Forecasting") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, size = 10))

print(p)

# Save the output
out_file <- file.path(getwd(), "causal_dag_output.png")
ggsave(out_file, plot = p, width = 8, height = 6)
message("Saved plot to: ", out_file)
