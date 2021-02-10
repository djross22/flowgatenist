// 
//

data {
  int<lower=1> N;             //number of cells
  vector[N] signal;             //flowcytometry signal in BL1A channel from blanks


}

transformed data {
  vector[N] scaled_signal; //signal rescaled by sample standard deviation
  real signal_SD;
  
  signal_SD = sd(signal);
  
  scaled_signal = signal/signal_SD;

}

parameters {
  real scaled_mu;                   // mean
  real<lower=0> scaled_sigma;       // standard deviation
  
}

transformed parameters {
}

model {
  scaled_mu ~ normal(0, 10);
  scaled_sigma ~ normal(0, 10);

  scaled_signal ~ normal(scaled_mu, scaled_sigma);


}

generated quantities {
  real mu;      // mean
  real sigma;   // standard deviation
  real beta;    // inverse shape (scale of exponential dist.)
  real post_pred_signal;  //posterior predictive signal
  
  mu = scaled_mu*signal_SD;
  sigma = scaled_sigma*signal_SD;
  
  post_pred_signal = normal_rng(mu, sigma);

}
