data {
  int<lower=1> N;             //number of cytometer events
  vector[N] signal;           //flowcytometry signal
  real mu;                    // mean of noise distribution
  real<lower=0> sigma;        // standard deviation of noise distribution


}

transformed data {

}

parameters {
  real<lower=0, upper=20> alpha;        // shape of gamma
  real<lower=0, upper=1> beta;         // rate (inverse scale) of gamma
  
  vector<lower=0>[N] gamma_signal;    //exponentially distributed part of flow cytometry signal
  
}

transformed parameters {
  vector[N] noise_signal;             // normally distributed part of scaled flow cytometry signal (scaled)
  
  noise_signal = signal - gamma_signal;

}

model {
  noise_signal ~ normal(mu, sigma);
  gamma_signal ~ gamma(alpha, beta);

}

generated quantities {
  real post_pred_signal;  //posterior predictive signal
  real post_pred_gamma;  //posterior predictive signal
  real post_pred_noise;  //posterior predictive signal
  
  vector[N] log_like_gamma;         // log likelihood
  vector[N] log_like_noise;         // log likelihood
  vector[N] log_like;         // log likelihood
  
  for (i in 1:N) {
    log_like_gamma[i] = gamma_lpdf(gamma_signal[i] | alpha, beta);
    log_like_noise[i] = normal_lpdf(noise_signal[i] | mu, sigma);
    log_like[i] = log_like_gamma[i] + log_like_noise[i];
  }
  
  post_pred_gamma = gamma_rng(alpha, beta);
  post_pred_noise = normal_rng(mu, sigma);
  post_pred_signal = post_pred_gamma + post_pred_noise;

}
