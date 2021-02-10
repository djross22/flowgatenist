// Fit to flow cytometry data with a mixture model of a exponential background +
//   a gamma distibution for the actual cell distribution + a tail distribution
//   to account for multi-plet events. All convoluted with a Normally distributed noise
//   The parameters for the background and noise are infered from a blank measurement fit 
//   with the Stan model: "fit exp modified normal.stan"
//

data {
  int<lower=1> N;        // number of cytometer events
  vector[N] signal;      // flowcytometry signal for each event
  
  int<lower=3> N_tail;    // number of terms to keep in tail/multiplet distribution 
    
  real mu_mean;          // mean of location parameter for noise
  real log_sigma_mean;   // mean of log scale parameter for noise 
  real log_lamb_mean;    // mean 0f log inverse scale parameter for exponentially distributed background
                         //   normally would be 'lambda' but avoiding that becasue it is a Python keyword
  //real mu_stdv;
  //real log_sigma_stdv;
  //real log_lamb_stdv;
  
  real alpha;                // shape parameter for gamma dist.
  real beta;                 // rate (inverse scale) parameter of gamma dist..
  
  //mu = -824;
  //sigma = 990;
  //lamb = 1.0/932;

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
  real log_B;                    // log of decay parameter for gamma distribution multiplet tail
  
  simplex[3] theta;                 // fraction of event that are in each component of mixture distribution
  
  vector<lower=0>[N] gamma_signal;    // flow cytometry signal with noise subtracted.
  

}

transformed parameters {
  vector[N] noise_signal;    // noise part of scaled flow cytometry signal
  
  real log_norm_Z;           // normalization factor for tail distribution
  real ps[N_tail+1];         // 16 component mixture model
  real norm_x[N_tail-1];     // dummy variable for adding up norm_z
  real target_add;           // add up contributions to log posterior - to be added to target in model block
  
  real tail_B;               // decay rate of tail distribution
  
  real theta1;               // fraction of events that are singlet cell events
  real theta_tail;           // fraction of events from cell multiplets
  real theta_back;           // fraction of events from background
  
  theta_back = theta[1];
  theta1 = theta[2];
  theta_tail = theta[3];
  
  tail_B = exp(log_B);
  for (k in 2:N_tail) {
    norm_x[k-1] = -k*tail_B;
  }
  log_norm_Z = log_sum_exp(norm_x);
  
  noise_signal = signal - gamma_signal;
  
  target_add = 0;
  for (i in 1:N) {
    ps[1] = log(theta1) + gamma_lpdf(gamma_signal[i] | alpha, beta); // singlet events
    for (k in 2:N_tail) {  // multiplet tail events
      ps[k] = log(theta_tail) - k*tail_B - log_norm_Z + gamma_lpdf(gamma_signal[i] | k*alpha, beta);
    }
    ps[N_tail+1] = log(theta_back) + exponential_lpdf(gamma_signal[i] | lamb); // spurious/background/dark events
    
    target_add += log_sum_exp(ps);
  }

}

model {
  log_B ~ normal(0, 3);
  
  noise_signal ~ normal(mu, sigma);
  
  target += target_add;
  
  
}

generated quantities {
  real beta_Mathematica;         // Mathematica defined beta (inverse of Stan defined beta)
  real gamma_mean;               // mean of gamma distributed signal
  real gamma_goem_mean;          // geometric mean of gamma distributed signal
  real log_difference;           // 1/2 difference between log alpha and log_beta for gamma distribution
  real log_average;              // average of log alpha and log_beta for gamma distribution
  
  log_difference = 0.5*(log(alpha) - log(beta));
  log_average = 0.5*(log(alpha) + log(beta));
  
  beta_Mathematica = 1/beta;
  
  gamma_mean = alpha/beta;
  gamma_goem_mean = exp(digamma(alpha) - log(beta));
  

}
