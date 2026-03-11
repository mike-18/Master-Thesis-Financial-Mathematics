library(quarks)
library(esback)
library(readr)
library(dplyr)
library(gridExtra) 

portfolio_name <- "blueChip" # Options: crossAsset, blueChip
alpha_es_acerby <- 0.025
output_dir <- "~/Desktop/Masterarbeit Finanzmathe/Masterthesis - Backtesting/VaR_and_ES/var_97_5_es_97_5"
price_path <- paste0("Desktop/Masterarbeit Finanzmathe/Masterthesis - Backtesting/Datasets/", portfolio_name, "_portfolio.csv")
raw_returns <- read.csv(price_path, sep = ",", header = TRUE)
return_mat <- as.matrix(raw_returns)

num_assets <- ncol(return_mat)
weights <- rep(1 / num_assets, num_assets)
portfolio_returns <- as.numeric(return_mat %*% weights)

# Custom function for Acerbi & Szekely Test 2
acerbi_szekely_test2 <- function(ret, var, es, alpha) {
  var_abs <- abs(var)
  es_abs  <- abs(es)
  T_len <- length(ret)
  is_breach <- ret < -var_abs
  term <- ifelse(is_breach, ret / es_abs, 0)
  term[is.nan(term) | is.infinite(term)] <- 0
  z2_score <- sum(term) / (T_len * alpha) + 1
  reject_h0 <- z2_score < -0.7
  return(list(
    Z2_Score = z2_score,
    Reject_H0_5pct = reject_h0,
    Breaches = sum(is_breach),
    Expected_Breaches = T_len * alpha,
    N = T_len
  ))
}

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
    current_as_z2 <- NA
    current_exceedances <- NA 
    
    # 1. Acerbi & Szekely Test 2
    try({
      as_result <- acerbi_szekely_test2(ret = r_sub, var = q_sub, es = e_sub, alpha = alpha_es_acerby)
      current_as_z2 <- round(as_result$Z2_Score, 4)
      current_exceedances <- as_result$Breaches 
    })
    
    #Append current results to dataframe 
    results_df <- rbind(results_df, data.frame(
      Model = model,
      Period = p_name,
      N_Days = length(idx),
      Exceedances = current_exceedances,   
      AS_Z2_Score = current_as_z2,
      stringsAsFactors = FALSE
    ))
  }
}

pdf_output_path <- file.path(output_dir, paste0(portfolio_name, "_Test_Results.pdf"))
pdf(pdf_output_path, width = 11, height = 8.5) 
gridExtra::grid.table(results_df)
dev.off()

cat(pdf_output_path, "\n")
