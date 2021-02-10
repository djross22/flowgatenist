// 
//

data {
  int<lower=1> N;        // number of data points
  vector[N] x;           // x data to be fit
  vector[N] y;           // y data to be fit
  
  vector[4] input_coef;   // coeficients from polynomial fitting in numpy - for normalization of coef

}

transformed data {
  real x_min;
  real x_max;
  
  x_min = min(x);
  x_max = max(x);

}

parameters {
  vector[4] norm_coef;            // normalized polynimial coeficients; coef[1] is 3rd order coeficient
  real<lower=0> sigma;            // standard deviation of noise
  
}

transformed parameters {
  vector[4] coef;            // normalized polynimial coeficients; coef[1] is 3rd order coeficient
  
  coef = norm_coef .* input_coef;
}

model {
  for (i in 1:N) {
    y[i] ~ normal(coef[1]*x[i]^3 + coef[2]*x[i]^2 + coef[3]*x[i] + coef[4], sigma);
  }

}

generated quantities {
  vector[3] deriv_array;
  vector[2] roots;
  real fit_mode;
  
  for (i in 1:3) {
    deriv_array[i] = (4-i)*coef[i];
  }
  
  roots[1] = (-deriv_array[2] + sqrt(deriv_array[2]^2 - 4*deriv_array[1]*deriv_array[3]) ) / (2 * deriv_array[1]);
  roots[2] = (-deriv_array[2] - sqrt(deriv_array[2]^2 - 4*deriv_array[1]*deriv_array[3]) ) / (2 * deriv_array[1]);
  
  if ( roots[2]>x_min && roots[2]<x_max )
    fit_mode = roots[2];
  else
    fit_mode = roots[1];

}
