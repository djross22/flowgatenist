// Fit to flow cytometry data with a mixture model of a exponential background +
//   a gamma distibution for the actual cell distribution + a tail distribution
//   to account for multi-plet events. All convoluted with a Normally distributed noise
//   The parameters for the background and noise are infered from a blank measurement fit 
//   with the Stan model: "fit exp modified normal.stan"
//

data {
  real signal;             // cytometry signal for a single event
  
  int<lower=3> N_tail;     // number of terms to keep in tail/multiplet distribution 
    
  real mu_mean;            // mean of location parameter for noise
  real log_sigma_mean;     // mean of log scale parameter for noise 
  real log_lamb_mean;      // mean 0f log inverse scale parameter for exponentially distributed background
                           //   normally would be 'lambda' but avoiding that becasue it is a Python keyword
  
  real alpha;              // shape parameter for gamma dist.
  real beta;               // rate (inverse scale) parameter of gamma dist.
  
  real tail_B;             // decay rate of tail distribution
  
  simplex[3] theta;        // fraction of event that are in each component of mixture distribution

}

transformed data {
  real mu;          // location parameter for noise
  real log_sigma;   // log of scale parameter for noise 
  real log_lamb;    // log of inverse scale parameter for exponentially distributed background
                    //   normally would be 'lambda' but avoiding that becasue it is a Python keyword
  
  real sigma;       // scale parameter for noise 
  real lamb;        // inverse scale parameter for exponentially distributed background
                    //   normally would be 'lambda' but avoiding that becasue it is a Python keyword
                    
  mu = mu_mean;
  log_sigma = log_sigma_mean;
  log_lamb = log_lamb_mean;
  
  sigma = exp(log_sigma);
  lamb = exp(log_lamb);
  
}

parameters {
  real<lower=0> gamma_signal;    // flow cytometry signal with noise subtracted.
  
}

transformed parameters {
  real noise_signal;    // noise part of scaled flow cytometry signal
  real log_like;        // log likelihood
  real gamma_log_like;        // log likelihood, gamma distribution part
  real back_log_like;        // log likelihood, background distribution part
  
  real log_norm_Z;           // normalization factor for tail distribution
  real ps[N_tail+1];         // N_tail+1 component mixture model
  real norm_x[N_tail-1];     // dummy variable for adding up norm_z
  real target_add;           // add up contributions to log posterior - to be added to target in model block
  
  for (k in 2:N_tail) {
    norm_x[k-1] = -k*tail_B;
  }
  log_norm_Z = log_sum_exp(norm_x);
  
  noise_signal = signal - gamma_signal;
  

  ps[1] = log(theta[2]) + gamma_lpdf(gamma_signal | alpha, beta); // singlet events
  
  for (k in 2:N_tail) {  // multi-plet/tail events
    ps[k] = log(theta[3]) - k*tail_B - log_norm_Z + gamma_lpdf(gamma_signal | k*alpha, beta);
  }
  
  ps[N_tail+1] = log(theta[1]) + exponential_lpdf(gamma_signal | lamb); // background/dark events
  
  log_like = log_sum_exp(ps);
  gamma_log_like = ps[1];
  back_log_like = ps[N_tail+1];
  
  target_add = log_sum_exp(ps);
    

}

model {

  target += normal_lpdf(noise_signal | mu, sigma);
  
  target += target_add;
  
}

generated quantities {
  

}
