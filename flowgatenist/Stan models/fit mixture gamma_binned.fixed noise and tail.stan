// Fit to flow cytometry data with a mixture model of a exponential background +
//   a gamma distibution for the actual cell distribution + a tail distribution
//   to account for multi-plet events. All convoluted with a Normally distributed noise
//   The parameters for the background and noise are infered from a blank measurement fit 
//   with the Stan model: "fit exp modified normal.stan"
//

data {
  int<lower=1> N;          // number of cytometer events
  vector[N] signal;        // mid-point flow cytometry signal for each bin
  int<lower=0> counts[N];  // number of events observed for each bin
  
  int<lower=3> N_tail;     // number of terms to keep in tail/multiplet distribution 
    
  real mu_mean;            // mean of location parameter for noise
  real log_sigma_mean;     // mean of log scale parameter for noise 
  real log_lamb_mean;      // mean of log inverse scale parameter for exponentially distributed background
                           //   normally would be 'lambda' but avoiding that becasue it is a Python keyword
  
  real tail_B;             // decay rate of tail distribution
  real tail_fraction;      // theta_tail/(theta_tail + theta1)
  
  real prior_dark_obs;      // parameters for setting prior on theta_back parameter
  real prior_bright_obs;     // parameters for setting prior on theta_back parameter

}

transformed data {
  real mu;          // location parameter for noise
  real log_sigma;   // log of scale parameter for noise 
  real log_lamb;    // log of inverse scale parameter for exponentially distributed background
                    //   normally would be 'lambda' but avoiding that becasue it is a Python keyword
  
  real sigma;       // scale parameter for noise 
  real lamb;        // inverse scale parameter for exponentially distributed background
                    //   normally would be 'lambda' but avoiding that becasue it is a Python keyword
                    
  real log_a_lower;
  real log_a_upper;
  real log_b_lower;
  real log_b_upper;
  
  mu = mu_mean;
  log_sigma = log_sigma_mean;
  log_lamb = log_lamb_mean;
  
  sigma = exp(log_sigma);
  lamb = exp(log_lamb);
  
  log_a_lower = 0;
  log_a_upper = 3;
  
  log_b_lower = -9;            // lower limit is for scale=10,000;
  log_b_upper = log_lamb+0.4;  // upper limit is inverse scale of exponential background dist.
  
}

parameters {
  real<lower=log_a_lower, upper=log_a_upper> log_alpha; // log of alpha parameter for gamma distribution
  real<lower=log_b_lower, upper=log_b_upper> log_beta;  // log of beta parameter for gamma distribution
  
  real<lower=0, upper=1> theta_back;  // fraction of events in background distribution
  
  vector<lower=0>[N] gamma_signal;    // flow cytometry signal with noise subtracted.
  

}

transformed parameters {
  vector[N] noise_signal;    // noise part of scaled flow cytometry signal
  
  real log_norm_Z;           // normalization factor for tail distribution
  real ps[N_tail+1];         // 16 component mixture model
  real norm_x[N_tail-1];     // dummy variable for adding up norm_z
  real target_add;           // add up contributions to log posterior - to be added to target in model block
  
  real alpha;                // shape parameter for gamma dist.
  real beta;                 // rate (inverse scale) parameter of gamma dist..
  
  real theta1;               // fraction of events that are singlet cell events
  real theta_tail;           // fraction of events from cell multiplets
  
  theta1 = (1 - theta_back)*(1 - tail_fraction);
  theta_tail = (1 - theta_back)*tail_fraction;
  
  for (k in 2:N_tail) {
    norm_x[k-1] = -k*tail_B;
  }
  log_norm_Z = log_sum_exp(norm_x);
  
  alpha = exp(log_alpha);
  beta = exp(log_beta);
  
  noise_signal = signal - gamma_signal;
  
  target_add = 0;
  for (i in 1:N) {
    ps[1] = log(theta1) + gamma_lpdf(gamma_signal[i] | alpha, beta); // singlet events
    for (k in 2:N_tail) {  // multiplet tail events
      ps[k] = log(theta_tail) - k*tail_B - log_norm_Z + gamma_lpdf(gamma_signal[i] | k*alpha, beta);
    }
    ps[N_tail+1] = log(theta_back) + exponential_lpdf(gamma_signal[i] | lamb); // spurious/background/dark events
    
    target_add += log_sum_exp(ps)*counts[i];
  }

}

model {
  #log_alpha ~ normal(1.5, 1.5);
  #log_beta ~ normal(-5.5, 3);
  
  theta_back ~ beta(prior_dark_obs, prior_bright_obs);
  
  //noise_signal ~ normal(mu, sigma);
  for (i in 1:N) {
    target += normal_lpdf(noise_signal[i] | mu, sigma)*counts[i];
  }
  
  target += target_add;
  
  
}

generated quantities {
  real log_difference;           // 1/2 difference between log alpha and log_scaled_beta for gamma distribution
  real log_average;              // average of log alpha and log_scaled_beta for gamma distribution
  real gamma_scale;              // scale parameter of gamma distribution
  real gamma_mean;               // mean of gamma distributed signal
  real gamma_goem_mean;          // geometric mean of gamma distributed signal
  real effective_tail_fraction;  // (number of cells corresponding to tail events)/(total number of events)
  real log_B;                    // log of decay parameter for gamma distribution multiplet tail
  
  log_difference = 0.5*(log_alpha - log_beta);
  log_average = 0.5*(log_alpha + log_beta);
  
  log_B = log(tail_B);
  
  gamma_scale = 1/beta;
  
  gamma_mean = alpha/beta;
  gamma_goem_mean = exp(digamma(alpha) - log(beta));
  
  effective_tail_fraction = theta_tail*(2*exp(tail_B) - 1)/(exp(tail_B) - 1);
  
}
