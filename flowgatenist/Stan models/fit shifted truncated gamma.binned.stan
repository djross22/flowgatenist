// 
//

data {
  int<lower=1> N;                                             // number of cytometry events
  real lower_bound;                                           // lower bound for truncation of signal
  real upper_bound;                                           // upper bound for truncation of signal
  vector<lower=lower_bound, upper=upper_bound>[N] signal;     // mid-point flow cytometry signal for each bin
  int<lower=0> counts[N];                                     // number of events observed for each bin
  
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
  real<lower=0> alpha;                          // shape of gamma distribution
  real<lower=0> scaled_beta;                    // scaled rate of gamma distribution
  
}

transformed parameters {
  vector[N] shifted_signal;
  
  shifted_signal = scaled_signal - scaled_shift;

}

model {
  //scaled_shift ~ normal(0, 10);
  //alpha ~ normal(2, 30);
  //scaled_beta ~ normal(0, 30);
  
  for (n in 1:N) {
    //shifted_signal[n] ~ gamma(alpha, scaled_beta) T[scaled_lower-scaled_shift,scaled_upper-scaled_shift];
    //shifted_signal[n] ~ gamma(alpha, scaled_beta);
    target += counts[n]*gamma_lpdf(shifted_signal[n] | alpha, scaled_beta);
    
    if (shifted_signal[n] < scaled_lower-scaled_shift || shifted_signal[n] > scaled_upper-scaled_shift) {
      target += negative_infinity();
    } else {
      target += -1*counts[n]*log_diff_exp(gamma_lccdf(scaled_upper-scaled_shift | alpha, scaled_beta), gamma_lccdf(scaled_lower-scaled_shift | alpha, scaled_beta));
    }
    
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
