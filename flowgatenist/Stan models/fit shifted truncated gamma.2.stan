// 
//

data {
  int<lower=1> N;                                             // number of cytometry events
  real lower_bound;                                           // lower bound for truncation of signal
  real upper_bound;                                           // upper bound for truncation of signal
  vector<lower=lower_bound, upper=upper_bound>[N] signal;     // flow cytometry signal
  
}

transformed data {
  real shift;
  
  shift = -250;

}

parameters {
  real<lower=0> alpha;                  // shape of gamma distribution
  real<lower=0> beta;                     // rate of gamma distribution
  
  //real<upper=lower_bound> shift;                    // shift of gamma distributed data

}

transformed parameters {
  vector[N] shifted_signal;
  
  shifted_signal = signal - shift;
  
}

model {
  //scaled_shift ~ normal(0, 10);
  //log_difference ~ normal(2, 30);
  //log_average ~ normal(0, 30);
  
  for (n in 1:N) {
    //shifted_signal[n] ~ gamma(alpha, beta) T[lower_bound,upper_bound];
    shifted_signal[n] ~ gamma(alpha, beta) T[lower_bound-shift,upper_bound-shift];
  }

}

generated quantities {
  real beta_Mathematica;         // Mathematica defined beta (inverse of Stan defined beta)
  real gamma_mean;               // mean of gamma distributed signal
  real gamma_goem_mean;          // geometric mean of gamma distributed signal

  beta_Mathematica = 1/beta;
  
  gamma_mean = alpha/beta;
  gamma_goem_mean = exp(digamma(alpha) - log(beta));

}
