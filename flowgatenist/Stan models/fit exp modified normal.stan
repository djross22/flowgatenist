// 
//

data {
  int<lower=1> N;             //number of cells
  vector[N] signal;           //flowcytometry signal in fluorescence channel from blanks
  real min_scaled_beta;       // lower limit on scaled_beta parameter

}

transformed data {
  vector[N] scaled_signal;    //signal shifted and rescaled by sample standard deviation
  real signal_mean;
  real signal_sd;
  
  signal_mean = mean(signal);
  signal_sd = sd(signal);
  
  scaled_signal = (signal - signal_mean)/signal_sd;

}

parameters {
  real scaled_mu;                   // mean
  real<lower=0> scaled_sigma;       // standard deviation
  real<lower=min_scaled_beta> scaled_beta;        // inverse shape (scale of exponential dist.)
  

}

transformed parameters {
  real<lower=0> scaled_lambda;        // shape
  
  scaled_lambda = 1/scaled_beta;

}

model {
  scaled_mu ~ normal(0, 10);
  scaled_sigma ~ normal(0, 10);
  scaled_beta ~ normal(0, 10);

  scaled_signal ~ exp_mod_normal(scaled_mu, scaled_sigma, scaled_lambda);


}

generated quantities {
  real mu;      // mean
  real sigma;   // standard deviation
  real beta;    // inverse shape (scale of exponential dist.)
  real lamb;    // shape parameter of epxonential dist. - 
                //   normally would be 'lambda' but avoiding that becasue it is a Python keyword
  real post_pred_signal;  //posterior predictive signal
  
  mu = scaled_mu*signal_sd + signal_mean;
  sigma = scaled_sigma*signal_sd;
  beta = scaled_beta*signal_sd;
  lamb = scaled_lambda/signal_sd;
  
  post_pred_signal = exp_mod_normal_rng(mu, sigma, 1/beta);

}
