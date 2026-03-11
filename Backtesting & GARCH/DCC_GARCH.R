library(rugarch)
library(rmgarch)
library(readr)
library(dplyr)
library(xts)


# CONFIGURATION
portfolio_name <- "blueChip" # Options: crossAsset, blueChip
train_size     <- 4564       # Options: 5298, 4564
refit_step     <- 50         # Model refit frequency

# Output directory based on alpha levels
output_dir <- "~/Desktop/Masterarbeit Finanzmathe/Masterthesis - Backtesting/VaR_and_ES/var_97_5_es_97_5"
#output_dir <- "~/Desktop/Masterarbeit Finanzmathe/Masterthesis - Backtesting/VaR_and_ES/var_95_es_97_5"


alpha_var <- 0.025  # Significance level for Value at Risk
alpha_es  <- 0.025  # Significance level for Expected Shortfall

file_path <- sprintf("Desktop/Masterarbeit Finanzmathe/Masterthesis - Backtesting/Datasets/%s_portfolio.csv", portfolio_name)

cat(paste("Starte Analyse für:", portfolio_name, "\n"))
cat(paste("VaR Alpha:", alpha_var, "| ES Alpha:", alpha_es, "\n"))

# 1. LOAD DATA AND CREATE SYNTHETIC TIMELINE
raw_data <- read.csv(file_path, sep = ",", header = TRUE, stringsAsFactors = FALSE)
returns_mat <- as.matrix(raw_data)

# Create a synthetic daily sequence
# xts requires a timeline; using a fictional start date.. it doesnt matter where to start
n_rows <- nrow(returns_mat)
synthetic_dates <- seq(from = as.Date("2000-01-01"), by = "days", length.out = n_rows)

returns_xts <- xts(returns_mat, order.by = synthetic_dates)
returns_xts <- na.omit(returns_xts)

n_total <- nrow(returns_xts)
cat(paste("Gesamtanzahl Renditen:", n_total, "\n"))

if (n_total <= train_size) {
  stop(paste("Fehler: Zu wenige Daten (", n_total, ") für Training (", train_size, ")."))
}

n_forecast <- n_total - train_size
cat(paste("Training (In-Sample):", train_size, "\n"))
cat(paste("Forecast (Out-of-Sample):", n_forecast, "\n"))


# 2. MODEL SPECIFICATION
num_assets <- ncol(returns_xts)
weights    <- rep(1 / num_assets, num_assets)

uspec <- ugarchspec(
  mean.model = list(armaOrder = c(1, 0)),
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  distribution.model = "norm"
)

spec <- dccspec(
  uspec = multispec(replicate(num_assets, uspec)), 
  dccOrder = c(1, 1),
  distribution = "mvnorm"
)


# 3. ROLLING FORECAST

dcc_roll <- dccroll(
  spec,
  data = returns_xts, 
  n.ahead = 1,
  forecast.length = n_forecast, 
  n.start = train_size,          
  refit.every = refit_step,
  refit.window = "recursive",    
  solver = "gosolnp",
  fit.control = list(eval.se = TRUE)
)


# 4. EXTRACT RISK METRICS 
VaR_vec <- numeric(n_forecast)
ES_vec  <- numeric(n_forecast)
idx     <- 1

# Precompute Z-scores for efficiency
z_score_var <- qnorm(alpha_var) 
z_score_es  <- qnorm(alpha_es)  

# Analytical Expected Shortfall factor for normal distribution
# ES = sigma * (pdf(z_es) / alpha_es)
es_factor   <- dnorm(z_score_es) / alpha_es 

for (i in seq_along(dcc_roll@mforecast)) {
  fcast <- dcc_roll@mforecast[[i]]@mforecast
  if (length(fcast$H) == 0) next
  
  H_list  <- fcast$H
  mu_list <- fcast$mu
  
  if (is.array(mu_list) && length(dim(mu_list)) == 3) {
    mu_mat <- matrix(mu_list, nrow = dim(mu_list)[3], byrow = TRUE)
  } else {
    mu_mat <- matrix(mu_list, ncol = num_assets, byrow = TRUE)
  }
  
  for (t in seq_along(H_list)) {
    if (idx > n_forecast) break
    
    Cov_t <- as.matrix(H_list[[t]][, , 1])
    port_variance <- t(weights) %*% Cov_t %*% weights
    sigma_p       <- sqrt(as.numeric(port_variance))
    
    mu_t  <- mu_mat[t, ]
    ret_t <- sum(weights * mu_t)
    
    # --- CALCULATE VAR AND ES ---
    
    # 1. Calculate Value at Risk
    VaR_vec[idx] <- ret_t + z_score_var * sigma_p
    
    # 2. Calculate Expected Shortfall
    # ES is the expected return in the tail beyond alpha_es
    ES_vec[idx]  <- -(ret_t + es_factor * sigma_p)
    
    idx <- idx + 1
  }
}

# 5. EXPORT RESULTS TO CSV
# Prepare data for export
realized_xts_forecast <- tail(returns_xts, n_forecast)
plot_dates            <- index(realized_xts_forecast) 
realized_port_ret     <- as.numeric(realized_xts_forecast %*% weights)

#ensure vector lengths match
n_min <- min(length(plot_dates), length(VaR_vec))
if (length(plot_dates) != length(VaR_vec)) {
  plot_dates        <- head(plot_dates, n_min)
  realized_port_ret <- head(realized_port_ret, n_min)
  VaR_vec           <- head(VaR_vec, n_min)
  ES_vec            <- head(ES_vec, n_min)
}


#Save VaR and ES to CSV
# Create a dataframe with the calculated metrics
results_df <- data.frame(
  VaR = VaR_vec,
  ES  = ES_vec
)

csv_filename <- file.path(output_dir, paste0(portfolio_name, "_GARCH.csv"))
write.csv(results_df, csv_filename, row.names = FALSE)
cat("Fertig.\n")