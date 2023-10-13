import numpy as np

def fourier_coeff(f_eval,  T):
    return np.fft.fft(f_eval, axis=0) * T / len(f_eval)

N = 9   # number of input functions n= 1, 2, ..., N
num_exp = 10 #number of times for each N (for avergaing error)


d = 2**13 # grid points

t_left = -.5
t_right = .5

T = t_right - t_left

t = np.linspace(t_left,t_right,d)
noise = .02

beta = 1.1 
gamma =  1.01
if gamma <=1:
    raise ValueError("gamma must be > 1")
eta = .25
alpha_sup = (2*beta -1 - gamma) / 2*(gamma + eta)
print("alpha_sup" , alpha_sup)
alpha = .9 * alpha_sup

if alpha <= 0 or alpha > 1:
    raise ValueError("alpha must be > 0 and < 1")

def sin_polin(t,alpha,M):
    return sum((2* np.sin(2*j*np.pi*t) / (j**alpha)) for j in range(1,M+2))

def fourier_expon(t,alpha,M):
    return sum((np.exp(1j*2*k*np.pi*t) / (k**alpha)) for k in range(1,M+2)) + sum((np.exp(1j*2*k*np.pi*t) / (k**alpha)) for k in range(-M,0))



#CHOOSE ground truth function and obs. noise         y_i = x_i * true_w + noise
sum_until = 30  # index M of truncation of the fourier series
true_w = sin_polin(t,beta,sum_until)                        

##fourier coefficients of w_true
true_w_coeff = fourier_coeff(true_w, T)  # normalized by the number of grid points

#for time-loc inputs
delta = 0.076

#for freq-loc inputs
L_max = sum_until


#To draw samples from the distribution rho_l = l^-beta

def power_law_samples(N, L, eta):
    """
    Generates N samples drawn from a power law probability distribution on the integers
    with exponent eta.
    
    Args:
    - N: an integer specifying the number of samples to generate
    -L: the samples are drawn in the interval [1,L+1]
    - eta: a positive float specifying the exponent of the power law
    
    Returns:
    - A numpy array of N integers drawn from the power law distribution
    """
    # Define the power law probability density function
    def p(x):
        return 1.0 / (x**eta)
    
    # Define the normalization constant
    Z = sum([p(l) for l in range(1, L+1)])
    
    # Generate the samples
    samples = []
    while len(samples) < N:
        # Choose a random integer in the range [1, L+1] with probability proportional to p(l)
        n = np.random.choice(range(1, L+1), p=[p(l)/Z for l in range(1, L+1)])
        samples.append(n)
    
    return np.array(samples)

def kernel_cosines(t,gamma,M):
    return 1 + sum((2* np.cos(2*k*np.pi*t) / (k**gamma)) for k in range(1,M+1)) 

M = sum_until

kerfun_eval = kernel_cosines(t,gamma, M)

kernel_coeff = fourier_coeff(kerfun_eval, T)
kernel_coeff = np.abs(kernel_coeff)   ## CHECK WHY YO NEED THIS

#prepare error arrays
error_sampmean = np.zeros(N)
error_sampstd = np.zeros(N)
error_logmean = np.zeros(N)
error_logstd = np.zeros(N)

#Inference and compute H errors
for n in range(1,N+1):
    
    error_of_exper = np.zeros(num_exp)
    
    for j in range(0,num_exp):
    

        #Localized in frequency
        #random vector for frequencies
        # L = power_law_samples(n, L_max, eta)

        # X =  np.array([ np.cos(l*t*2*np.pi) + 1j*np.sin(l*t*2*np.pi)  for l in L]).T

        
         #Localized in time
        R = np.random.normal(0,.25,n)

        X = np.array([np.where(((t-l) <= 2*delta) & ((t-l)>=0), 1, 0) for l in R]).T / (2*delta)
   
        # fourier coefficients of X, Y and true sol  
        X_fourier =  fourier_coeff(X,T)                                   

        #take the conjugate of the fourier coefficients of X
        X_fourier_conj = np.conj(X_fourier) 
        
        #Output data matrix 
        Y = np.zeros((d,n))   #noisy 
        for i in range(0,n):
            Y[:,i] = np.fft.ifft(d*(true_w_coeff)*(X_fourier[:,i])) +  noise  * np.random.normal(0,1,d) 

        Y_fourier = fourier_coeff(Y,T)                                    

        lamb =  1e-4 * n ** (-  1  / (2* alpha + 2))
        w_fourier = np.zeros(t.size, dtype=np.complex128)
        for l in range(t.size):
            eigenval = kernel_coeff[l] * (np.abs(X_fourier[l,:])**2).sum() / N
            term1 = kernel_coeff[l] / (eigenval + lamb)
            term2 = (X_fourier_conj[l,:] * Y_fourier[l,:]).sum() / N

            w_fourier[l] = term1 * term2

        w_diff_coeff = w_fourier - true_w_coeff
        error_h_squared = (np.abs(w_diff_coeff[:M])**2 / kernel_coeff[:M]).sum()

        error_of_exper[j] = error_h_squared 
    
    error_sampmean[n-1] = np.mean(error_of_exper)     #error_of_exp.sum() / num_exp
    error_sampstd[n-1] = np.std(error_of_exper)        #np.sqrt((np.square(error_of_exp - error_sampmean[n-1] )).sum() / (num_exp-1))
    error_logmean[n-1] = np.mean(np.log(error_of_exper))
    error_logstd[n-1] = np.std(np.log(error_of_exper))


np.save('results/rkhs_errormean_timeloc_eta='+str(eta)+'lam=1e-4timesfac_delta='+str(delta)+'sum_until='+str(sum_until)+'_beta='+str(beta)+'_gamma='+str(gamma)+'_N='+str(N)+'_d='+str(d)+'_noise='+str(noise)+'_num_exp='+str(num_exp)+'_alpha='+str(alpha), error_sampmean)
np.save('results/rkhs_errorstd_timeloc_eta='+str(eta)+'lam=1e-4timesfac_delta='+str(delta)+'sum_until='+str(sum_until)+'_beta='+str(beta)+'_gamma='+str(gamma)+'_N='+str(N)+'_d='+str(d)+'_noise='+str(noise)+'_num_exp='+str(num_exp)+'_alpha='+str(alpha), error_sampstd)
np.save('results/rkhs_errorlogmean_timeloc_eta='+str(eta)+'lam=1e-4timesfac_delta='+str(delta)+'sum_until='+str(sum_until)+'_beta='+str(beta)+'_gamma='+str(gamma)+'_N='+str(N)+'_d='+str(d)+'_noise='+str(noise)+'_num_exp='+str(num_exp)+'_alpha='+str(alpha), error_logmean)
np.save('results/rkhs_errorlogstd_timeloc_eta='+str(eta)+'lam=1e-4timesfac_delta='+str(delta)+'sum_until='+str(sum_until)+'_beta='+str(beta)+'_gamma='+str(gamma)+'_N='+str(N)+'_d='+str(d)+'_noise='+str(noise)+'_num_exp='+str(num_exp)+'_alpha='+str(alpha), error_logstd)

