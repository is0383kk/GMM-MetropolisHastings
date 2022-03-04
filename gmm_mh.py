import numpy as np
from scipy.stats import  wishart, dirichlet 
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score as ari


############################## Loading a Data Set ##############################
print("Loading a Data Set")
x_nd_1 = np.loadtxt("./data1.txt") # Observation1(Corresponds to x_1 in the graphical model)
z_truth_n = np.loadtxt("./true_label.txt") # True label (True z_n)
K = 3 # Number of clusters
D = len(x_nd_1) # Number of data
dim = len(x_nd_1[0]) # Number of dimention
print(f"Number of clusters: {K}"); print(f"Number of data: {len(x_nd_1)}"); 
iteration = 50 # Iteration of gibbssampling
ARI = np.zeros((iteration)) # ARI per iteration
count_accept = np.zeros((iteration)) # number of acceptation


############################## Initializing parameters ##############################
# Please refer to the graphical model in README.
print("Initializing parameters")

# Set hyperparameters
alpha_k = np.repeat(2.0, K) # Hyperparameters for \pi
beta = 1.0; 
m_d_1 = np.repeat(0.0, dim); # Hyperparameters for \mu^A, \mu^B
w_dd_1 = np.identity(dim) * 0.05; # Hyperparameters for \Lambda^A, \Lambda^B
nu = dim # Hyperparameters for \Lambda^A, \Lambda^B (nu > Number of dimention - 1)

# Initializing \pi
pi_k = dirichlet.rvs(alpha=alpha_k, size=1).flatten()
alpha_hat_k = np.zeros(K)

# Initializing z
z_nk_new = np.random.multinomial(n=1, pvals=pi_k, size=D) # Current iteration z
z_nk_old = np.random.multinomial(n=1, pvals=pi_k, size=D) # z before 1 iteration
_, z_n = np.where(z_nk_new == 1)
_, z_n = np.where(z_nk_old == 1)

# Initializing unsampled \mu, \Lambda
mu_kd_1 = np.empty((K, dim)); lambda_kdd_1 = np.empty((K, dim, dim))

# Initializing learning parameters
eta_nk = np.zeros((D, K))
tmp_eta_n = np.zeros((K, D))
beta_hat_k_1 = np.zeros(K);
m_hat_kd_1 = np.zeros((K, dim)); 
w_hat_kdd_1 = np.zeros((K, dim, dim)); 
nu_hat_k_1 = np.zeros(K); 
cat_liks_new = np.zeros(D)
cat_liks_old = np.zeros(D)



############################## Metropolis-Hastings algorithm ##############################
print("Metropolis-Hastings algorithm")
for i in range(iteration):
    print(f"----------------------Iteration : {i+1}------------------------")
    z_pred_n = [] # Labels estimated by the model
    count = 0

    z_nk = np.zeros((D, K));
    # Sampling the current iteration z 
    for k in range(K): 
        tmp_eta_n[k] = np.diag(-0.5 * (x_nd_1 - mu_kd_1[k]).dot(lambda_kdd_1[k]).dot((x_nd_1 - mu_kd_1[k]).T)).copy() 
        tmp_eta_n[k] += 0.5 * np.log(np.linalg.det(lambda_kdd_1[k]) + 1e-7)
        tmp_eta_n += np.log(pi_k[k] + 1e-7) 
        eta_nk[:, k] = np.exp(tmp_eta_n[k])
    eta_nk /= np.sum(eta_nk, axis=1, keepdims=True)

    for d in range(D):
        z_nk_new[d] = np.random.multinomial(n=1, pvals=eta_nk[d], size=1).flatten() # sampling z_nk_new

        cat_liks_new[d] = multivariate_normal.pdf(
                          x_nd_1[d], 
                          mean=mu_kd_1[np.argmax(z_nk_new[d])], 
                          cov=np.linalg.inv(lambda_kdd_1[np.argmax(z_nk_new[d])]),
                          )
        cat_liks_old[d] = multivariate_normal.pdf(
                          x_nd_1[d], 
                          mean=mu_kd_1[np.argmax(z_nk_old[d])], 
                          cov=np.linalg.inv(lambda_kdd_1[np.argmax(z_nk_old[d])]),
                          )
        judge_r = cat_liks_new[d] / cat_liks_old[d] 
        judge_r = min(1, judge_r) # acceptance rate
        rand_u = np.random.rand() # sampling random variable
        if judge_r >= rand_u: 
            z_nk[d] = z_nk_new[d]
            count = count + 1 # count accept
        else: 
            z_nk[d] = z_nk_old[d]
        z_pred_n.append(np.argmax(z_nk[d]))

    # Process on sampling \mu, \lambda using the updated z
    for k in range(K):
        # Calculate the parameters of the posterior distribution of \mu
        beta_hat_k_1[k] = np.sum(z_nk[:, k]) + beta; 
        m_hat_kd_1[k] = np.sum(z_nk[:, k] * x_nd_1.T, axis=1); 
        m_hat_kd_1[k] += beta * m_d_1; 
        m_hat_kd_1[k] /= beta_hat_k_1[k]; 

        
        # Calculate the parameters of the posterior distribution of \Lambda
        tmp_w_dd_1 = np.dot((z_nk[:, k] * x_nd_1.T), x_nd_1); 
        tmp_w_dd_1 += beta * np.dot(m_d_1.reshape(dim, 1), m_d_1.reshape(1, dim)); 
        tmp_w_dd_1 -= beta_hat_k_1[k] * np.dot(m_hat_kd_1[k].reshape(dim, 1), m_hat_kd_1[k].reshape(1, dim))
        tmp_w_dd_1 += np.linalg.inv(w_dd_1); 
        w_hat_kdd_1[k] = np.linalg.inv(tmp_w_dd_1); 
        nu_hat_k_1[k] = np.sum(z_nk[:, k]) + nu
        
        # Sampling \Lambda
        lambda_kdd_1[k] = wishart.rvs(size=1, df=nu_hat_k_1[k], scale=w_hat_kdd_1[k])
        
        # Sampling \mu
        mu_kd_1[k] = np.random.multivariate_normal(
            mean=m_hat_kd_1[k], cov=np.linalg.inv(beta_hat_k_1[k] * lambda_kdd_1[k]), size=1
        ).flatten()
    
    
    # Process on sampling \pi using the updated z
    # Calculate the parameters of the posterior distribution of \pi
    alpha_hat_k = np.sum(z_nk, axis=0) + alpha_k
    
    # Sampling \pi
    pi_k = dirichlet.rvs(size=1, alpha=alpha_hat_k).flatten()
    
    ARI[i] = np.round(ari(z_truth_n, z_pred_n), 3) # Calculate ARI
    count_accept[i] = count # Number of times accepted during current iteration
    print(f"ARI:{ARI[i]}, Accept_num:{count_accept[i]}")

    z_nk_old = z_nk_new


# ARI
plt.plot(range(0,iteration), ARI, marker="None")
plt.xlabel('iteration')
plt.ylabel('ARI')
plt.ylim(0,1)
#plt.savefig("./image/ari.png")
plt.show()
plt.close()

# number of acceptation 
plt.figure()
plt.ylim(0,D)
plt.plot(range(0,iteration), count_accept, marker="None", label="Accept_num")
plt.xlabel('iteration')
plt.ylabel('Number of acceptation')
plt.legend()
#plt.savefig('./image/accept.png')
plt.show()
plt.close()
