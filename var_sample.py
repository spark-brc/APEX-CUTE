import math,random,parm

test_var=0
max_var_test=0
min_var_test=0
new_var=0
perturb_val = 0

def perturbation(test_var, max_var_test,min_var_test,perturb_val,jj):
    
    k6 = perturb_val

    #Standard Gaussian random number based on Numerical Recipies 'gasdev' and Marsaglia-Bray Algorithm
    k3 = 2.0
    while k3>=1 or k3==0:
        k7 = random.random()
        k8 = random.random()
        k1 = 2 * k7 - 1
        k2 = 2 * k8 -1
        k3 = k1 * k1 + k2 * k2

    k3 = (-2 * math.log(k3) / k3) ** 0.5
    k4 = random.random()
    if k4<0.5:
        k5 = k1 * k3
    else:
        k5 = k2 * k3
     
    new_var = test_var + k6 * (max_var_test - min_var_test) * k5

    #Check if new_var is an overshoot
    if new_var > max_var_test:
        new_var = max_var_test - (new_var - max_var_test)
        if new_var < min_var_test:
            new_var = max_var_test
    elif new_var <  min_var_test:
         new_var = min_var_test + (min_var_test - new_var)
         if new_var > max_var_test:
             new_var = min_var_test
    
    parm.cur_test_var[jj] = new_var 


    
 
 
