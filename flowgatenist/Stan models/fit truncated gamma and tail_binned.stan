// Fit to flow cytometry data with a mixture model of a gamma distibution 
// plus a tail distribution that is given by summed multiple draws from the main gamma
//
// This version of the model takes binned data to speed up computation for datasets with many data point
//

data {
  int<lower=1> N;          // number of cytometer events
  real lower_trunc;      // lower truncation on the data
  vector[N] signal;        // flow cytometry signal for each event, with the mean background subtracted
  int<lower=0> counts[N];  // number of events observed for each bin
  
  int<lower=3> N_tail;     // number of terms to keep in tail/multiplet distribution 

}

transformed data {
}

parameters {
  real log_difference;           // 1/2 difference between log alpha and log_scaled_beta for gamma distribution
  real log_average;              // average of log alpha and log_scaled_beta for gamma distribution
  
  real log_B;                    // log of decay parameter for gamma distribution multiplet tail
  
  real<lower=0, upper=1> theta_tail;    // fraction of event that are in each component of mixture distribution
  
}

transformed parameters {
  real log_norm_Z;           // normalization factor for tail distribution
  real ps[N_tail];           // N_tail component mixture model
  real norm_x[N_tail-1];     // dummy variable for adding up norm_z
  real target_add;           // add up contributions to log posterior - to be added to target in model block
  simplex[N_tail] theta;     // proportion of events in each gamma distribution,
  
  real alpha;                // shape parameter for gamma dist.
  real beta;                 // rate (inverse scale) parameter of gamma dist..
  
  real tail_B;               // decay rate of tail distribution
  
  real theta1;               // fraction of events that are singlet cell events
  
  theta1 = 1 - theta_tail;
  
  tail_B = exp(log_B);
  for (k in 2:N_tail) {
    norm_x[k-1] = -k*tail_B;
  }
  log_norm_Z = log_sum_exp(norm_x);
  
  theta[1] = theta1;
  for (k in 2:N_tail) {
    theta[k] = theta_tail*exp(-k*tail_B - log_norm_Z);
  }
  
  alpha = exp(log_average + log_difference);
  beta = exp(log_average - log_difference);
  
  target_add = 0;
  for (i in 1:N) {
    for (k in 1:N_tail) {  // multiplet tail events
      ps[k] = log(theta[k]) + gamma_lpdf(signal[i] | k*alpha, beta) - gamma_lccdf(lower_trunc | k*alpha, beta);
      
    }
    target_add += log_sum_exp(ps)*counts[i];
  }

}

model {
  log_difference ~ normal(3.45, 30);
  log_average ~ normal(-3, 30);
  log_B ~ normal(0, 3);
  
  target += target_add;
  
}

generated quantities {
  real beta_Mathematica;         // Mathematica defined beta (inverse of Stan defined beta)
  real gamma_mean;               // mean of gamma distributed signal
  real gamma_goem_mean;          // geometric mean of gamma distributed signal
  real effective_tail_fraction;  // (number of cells corresponding to tail events)/(total number of events)
  
  beta_Mathematica = 1/beta;
  
  gamma_mean = alpha/beta;
  gamma_goem_mean = exp(digamma(alpha) - log(beta));
  
  effective_tail_fraction = theta_tail*(2*exp(tail_B) - 1)/(exp(tail_B) - 1);
  
}
