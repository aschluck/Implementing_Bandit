import numpy as np #dot and outer products
from scipy.optimize import minimize #MLE
from scipy.linalg import cho_factor, cho_solve#can solve matrix without explicity inverting it
from scipy.special import expit #sigmoid

from simulation_utils import get_feedback, create_env
import algos
from models import Driver

def dueling_bandit(T_total, t0=5,K=10,n=1e-6, eta=0.5):
    """
    initial parameters
    d: int
        Feature dimension on each arm
    S_stream: list of arrays
        Each elm is a candidate set of arm for that round
    T_total: int
        Total duels
    t0: int
        number of initial input
    theta_t: array-like, size d
        Feature weights. Used to simulate binary outcomes ##What do we do here
    n: float
        Used for numerical stability of V
    K: int
        Arm size    
    eta: float
        exploration parameter

___________________________________________________

    returns
    V:array, size (d,d)
        Final updated matrix
    Z: array, size(number of duals, d)
        All feature difference vectors observed
    O: array, size number of duals
        Binary outcomes observed
    theta_hat:array, size d
        Final MLE estimate of feature weights after T_total rounds
    """
    ###Line 2###
    #init:Select t0 pairs (xt yt) t in[t0],each drawn at random from St,
    #and observe the corresponding preference feedback ot t in [t0]
    Z=[] #stores z_tau=x_tau-y_tau
    O=[] #stores outcomes (binary, 1 if x_tau beats y_tau)

    """Using item_sampler to have a batch of trajectories"""
    task="Driver"
    simulation_object = create_env(task)
    data = np.load(f"ctrl_samples/{task.lower()}_items.npz")
    u_pool = data["u_set"]
    phi_pool = data["phi_raw"]    
    N_pool = u_pool.shape[0]
    ##########################################################################

    d = simulation_object.num_of_features
    #t0=5*d #higher t0 worked better with standford paper but this is a lot of initial
    
    for trials in range(t0):
        idx=np.random.choice(N_pool, size=2, replace=False)
        i,j=idx[0],idx[1]
        input_A,input_B= u_pool[i],u_pool[j]
        phi_A, phi_B, s = get_feedback(simulation_object, input_A, input_B, 'strict')
        Z.append(phi_A-phi_B)#stores z_tau=x_tau-y_tau where phi_a is from get_features
        O.append(s)
    
    Z=np.array(Z, dtype=float)
    if Z.ndim==1:
        Z=Z.reshape(1,-1)
    O=np.array(O, dtype=int)

    ###line 3###
    #Compute V_{t0+1} = sum of outer products of initial duels
    V=np.eye(d) * n #creates I vector times n for stability (confidence vector)
    for z in Z:
        V+=np.outer(z,z) ##outer product of each term

    ##for line 5
    def neg_log(theta, z, o):
        theta=np.ravel(theta)#flatten
        s= z @ theta
        p1=expit(s) ##PL model probability
        return -1*(np.sum(o*np.log(p1+1e-12)+(1-o)*np.log(1-p1+1e-12)))


    theta_hat=np.zeros(d, dtype=float)
    ###line 4 and 10
    #loop through all the remaining rounds
    for t in range(t0,T_total):
    #Line 5 (compute MLE)
        theta_hat=minimize(neg_log, theta_hat, args=(Z, O), method='L-BFGS-B').x

        #Line 6/7
        idx = np.random.choice(N_pool, size=K, replace=False)
        St = u_pool[idx]        # (K, feed_size)
        phis = phi_pool[idx]    # (K, d)

        cho=cho_factor(V)#this and cho_solve are an efficient way to compute V^-1z
        C=[]

        print("Duel")
        ##Line 6
        ##Checks if traj. i beats all others under estimate
        for i in range(K):
            winner=True
            for j in range(K):
                if i==j:##dueling self
                    continue
                z=(phis[i]-phis[j]).reshape(-1)
                
                mean=theta_hat.dot(z) #predicted preference difference
                solve=cho_solve(cho,z.reshape(-1,1))
                bonus=eta*np.sqrt((z.T @ solve).item()) #uncertainty bonus
                if mean+bonus<=0:
                    winner=False
                    break
            if winner:
                C.append(i)#flag

        #Line 7 find the most uncertain pair
        #choose pair with largest uncertainty
        if len(C)==0:##top two means
            scores=phis.dot(theta_hat)
            order = np.argsort(scores)
            i,j=order[-1],order[-2]
            xt,yt=St[i],St[j]
        elif len(C)==1:#used to find the most unceartain case compared to the "best"
            i = C[0]
            val=-np.inf
            j=None
            for k in range(K):
                if i==k:
                    continue
                z=(phis[i]-phis[k]).reshape(-1)
                solve=cho_solve(cho,z.reshape(-1,1))
                bonus=eta*np.sqrt((z.T @ solve).item()) 
                if bonus>val:
                    val=bonus
                    j=k
            xt,yt=St[i],St[j]
        else:
            val=-np.inf
            pair=None
            for a in range(len(C)):
                for b in range(a+1,len(C)):
                    i=C[a]
                    j=C[b]
                    z=(phis[i]-phis[j]).reshape(-1)
                    solve=cho_solve(cho,z.reshape(-1,1))
                    value=(z.T @ solve).item() #gets us ||x-y||_{V^{-1}}^2
                    if value>val:
                        val=value
                        pair=(a,b)
            temp1, temp2=pair
            xt,yt=St[C[temp1]],St[C[temp2]]

        #Line 8: duel
        phi_A, phi_B, s = get_feedback(simulation_object, xt, yt, 'strict')
        zt=phi_A - phi_B

        #Line 9:update
        Z=np.vstack([Z,zt])
        O=np.append(O,s)
        V+=np.outer(zt,zt)
    #end loop
    #check MLE again
    theta_hat=minimize(neg_log, theta_hat, args=(Z, O), method='L-BFGS-B').x
    print(theta_hat)
    print(V)
    print(Z)
    print(O)
    return theta_hat,V,Z,O

def sample_St(simulation_object, K):
    temp=[]
    for k in range(K):
        input_A, _, _ = algos.random(simulation_object)
        temp.append(input_A)
    return np.array(temp)    