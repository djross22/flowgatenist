data {
  int<lower=1> N;             //number of cytometer events
  vector[N] signal;           //flowcytometry signal
  
  real mu;      // mean
  real sigma;   // standard deviation
  real<lower=0> alpha;        // shape of gamma
  real<lower=0> beta;         // rate (inverse scale) of gamma

}

transformed data {

}

parameters {
  vector<lower=0>[N] gamma_signal;    //gamma distributed part of flow cytometry signal

}

transformed parameters {
  vector[N] noise_signal;             // normally distributed part of scaled flow cytometry signal
  
  noise_signal = signal - gamma_signal;

}

model {
  noise_signal ~ normal(mu, sigma);
  gamma_signal ~ gamma(alpha, beta);

}

generated quantities {
  vector[N] log_like_gamma;         // log likelihood
  vector[N] log_like_noise;         // log likelihood
  vector[N] log_like;         // log likelihood
  
  for (i in 1:N) {
    log_like_gamma[i] = gamma_lpdf(gamma_signal[i] | alpha, beta);
    log_like_noise[i] = normal_lpdf(noise_signal[i] | mu, sigma);
    log_like[i] = log_like_gamma[i] + log_like_noise[i];
  }

}
