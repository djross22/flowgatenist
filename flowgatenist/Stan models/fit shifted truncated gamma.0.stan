// 
//

data {
  int<lower=1> N;                                             // number of cytometry events
  real lower_bound;                                           // lower bound for truncation of signal
  real upper_bound;                                           // upper bound for truncation of signal
  vector<lower=lower_bound, upper=upper_bound>[N] signal;     // flow cytometry signal
  
}

transformed data {
  vector[N] scaled_signal;    // signal rescaled by sample standard deviation
  real signal_SD;
  real scaled_lower;
  real scaled_upper;
  
  signal_SD = sd(signal);
  
  scaled_signal = signal/signal_SD;
  scaled_lower = lower_bound/signal_SD;
  scaled_upper = upper_bound/signal_SD;

}

parameters {
  real<upper=scaled_lower> scaled_shift;        // scaled shift of gamma distributed data
  real log_difference;                                // 1/2 difference between log alpha and 
                                                      //     log_scaled_beta for gamma distribution
  real log_average;                                   // average of log alpha and log_scaled_beta for gamma distribution
  
}

transformed parameters {
  real alpha;                  // shape of gamma distribution
  real scaled_beta;            // scaled rate of gamma distribution
  vector[N] shifted_signal;
  
  shifted_signal = scaled_signal - scaled_shift;
  
  alpha = exp(log_average + log_difference);
  scaled_beta = exp(log_average - log_difference);

}

model {
  scaled_shift ~ normal(0, 10);
  log_difference ~ normal(2, 30);
  log_average ~ normal(0, 30);
  
  for (n in 1:N) {
    shifted_signal[n] ~ gamma(alpha, scaled_beta) T[scaled_lower-scaled_shift,scaled_upper-scaled_shift];
  }

}

generated quantities {
  real beta;                     // rate of gamma distribution
  real beta_Mathematica;         // Mathematica defined beta (inverse of Stan defined beta)
  real gamma_mean;               // mean of gamma distributed signal
  real gamma_goem_mean;          // geometric mean of gamma distributed signal
  real shift;                    // shift of gamma distributed data
  
  shift = scaled_shift*signal_SD;
  
  beta = scaled_beta/signal_SD;
  
  beta_Mathematica = 1/beta;
  
  gamma_mean = alpha/beta;
  gamma_goem_mean = exp(digamma(alpha) - log(beta));

}
