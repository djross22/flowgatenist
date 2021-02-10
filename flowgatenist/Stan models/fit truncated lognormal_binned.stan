// Fit to flow cytometry data with a truncated log-normal distribution.
//
// This version of the model takes binned data to speed up computation for datasets with many data point
//

data {
  int<lower=1> N;        // number of cytometer events
  vector[N] signal;      // flowcytometry signal for each event
  int<lower=0> counts[N];  // number of events observed for each bin
  
  real lower_trunc;      // lower truncation on the data

}

transformed data {
  
}

parameters {
  real lognorm_mu;                   // location parameter for log-normal distribution
  real<lower=0> lognorm_sig;         // scale parameter for log-normal distribution
  
}

transformed parameters {
}

model {
  lognorm_mu ~ normal(5, 30);
  lognorm_sig ~ normal(0, 30);
  
  for (i in 1:N) {
    //signal[i] ~ lognormal(lognorm_mu, lognorm_sig) T[lower_trunc, ];
    target += counts[i]*(lognormal_lpdf(signal[i] | lognorm_mu, lognorm_sig) - lognormal_lccdf(lower_trunc | lognorm_mu, lognorm_sig));
  }
  
}

generated quantities {

}
