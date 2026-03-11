library(quarks)
library(esback)
library(readr)
library(dplyr)
library(gridExtra) 

portfolio_name <- "crossAsset" # Options: crossAsset, blueChip
alpha_es_acerby <- 0.025
output_dir <- "~/Desktop/Masterarbeit Finanzmathe/Masterthesis - Backtesting/VaR_and_ES/var_97_5_es_97_5"
#output_dir <- "~/Desktop/Masterarbeit Finanzmathe/Masterthesis - Backtesting/VaR_and_ES/var_95_es_97_5"
price_path <- paste0("Desktop/Masterarbeit Finanzmathe/Masterthesis - Backtesting/Datasets/", portfolio_name, "_portfolio.csv")
raw_returns <- read.csv(price_path, sep = ",", header = TRUE)
return_mat <- as.matrix(raw_returns)

num_assets <- ncol(return_mat)
weights <- rep(1 / num_assets, num_assets)
portfolio_returns <- as.numeric(return_mat %*% weights)


results_df <- data.frame()

model_names <- c("historical", "variance_covariance", "monte_carlo","GARCH", "Diffusion")

for(model in model_names) {
  
  file_name <- paste0(portfolio_name, "_", model, ".csv")
  results_path <- file.path(output_dir, file_name)
  
  risk_data <- read_csv(results_path, show_col_types = FALSE)
  
  q_vec <- risk_data$VaR
  e_vec <- risk_data$ES
  
  n_forecast <- length(q_vec)
  r_vec <- tail(portfolio_returns, n_forecast)
  
  # Define time periods for backtesting
  periods <- list(
    "Jahr_1"    = 1:min(252, n_forecast),
    "Jahr_2"    = if(n_forecast > 252) 253:min(504, n_forecast) else integer(0),
    "Jahr_3"    = if(n_forecast > 504) 505:n_forecast else integer(0),
    "Insgesamt" = 1:n_forecast
  )
  
  for(p_name in names(periods)) {
    idx <- periods[[p_name]]
    if(length(idx) == 0) next
    
    r_sub <- r_vec[idx]
    q_sub <- q_vec[idx]
    e_sub <- e_vec[idx]
    
    # Initialize placeholders for the current iteration
    current_esback_p <- NA
    current_quarks_p <- NA
    current_exceedances <- NA # Placeholder for exceedances count
    
    # 1. Expected Shortfall Backtest (esback) 
    try({
      backtest_result <- esback::esr_backtest(r = r_sub, q = q_sub, e = e_sub, alpha = 0.025, version = 1)
      # Extract p-value and round to 4 decimals
      current_esback_p <- round(backtest_result$pvalue_twosided_asymptotic, 4)
    })
    
    
    # 2. VaR Coverage Test (quarks)
    loss_sub <- -r_sub          
    var_abs_sub <- abs(q_sub)      
    test_obj <- list(loss = loss_sub, VaR = var_abs_sub, p = 0.95)
    
    try({
      cvg_result <- quarks::cvgtest(obj = test_obj, conflvl = 0.95)
      current_quarks_p <- round(cvg_result$p.cc, 4)
    })
    
    # --- Append current results to dataframe ---
    results_df <- rbind(results_df, data.frame(
      Model = model,
      Period = p_name,
      N_Days = length(idx),
      Exceedances = current_exceedances,   
      ESBack_p_val = current_esback_p,    
      Quarks_p_cc = current_quarks_p,
      stringsAsFactors = FALSE
    ))
  }
}

pdf_output_path <- file.path(output_dir, paste0(portfolio_name, "_Test_Results.pdf"))
pdf(pdf_output_path, width = 11, height = 8.5) 
gridExtra::grid.table(results_df)
dev.off()

cat(pdf_output_path, "\n")