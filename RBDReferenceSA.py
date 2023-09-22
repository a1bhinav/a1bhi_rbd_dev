from re import A
import numpy as np
import copy

np.set_printoptions(precision=4, suppress=True, linewidth = 100)

#joint: a joint object (see API below)
#link: a link object (see API below)
#Xmat: a sympy transformation matrix with one free variable as defined by its joint
#Xmat_Func: a function that returns a numpy matrix when passed a value for the free variable
#Imat: a numpy 6x6 inertia matrix
#S: a numpy 6x1 motion subspace matrix

class RBDReference:
    def __init__(self, robotObj):
        self.robot = robotObj


    def crm(self, v): #returns a 6x6 matrix version of 1x6 vector v
                        #so that it can be used in computing the cross product (same as matlab)
        vcross = [[   0, -v[2],  v[1],    0.,    0.,    0.],
                  [v[2],    0., -v[0],    0.,    0.,    0.],
                  [-v[1], v[0],    0.,    0.,    0.,    0.],
                  [  0., -v[5],  v[4],    0., -v[2],  v[1]],
                  [v[5],    0., -v[3],  v[2],    0., -v[0]],
                  [-v[4], v[3],    0., -v[1],  v[0],    0.]]
        #Unlike matrix, asmatrix does not make a copy if the input is already a matrix or an ndarray.
        # Equivalent to matrix(data, copy=False).
        # https://numpy.org/doc/stable/reference/generated/numpy.asmatrix.html
        #                 x = np.array([[1, 2], [3, 4]])
        #                 m = np.asmatrix(x)
        #                 x[0, 0] = 5

        return np.asmatrix(vcross)

    def crf(self, v):
        # https://www.youtube.com/watch?v=pRZJdHnQYok&ab_channel=TheCompleteGuidetoEverything
        vcross = -self.crm(v).conj().T #negative complex conjugate transpose
        return vcross

    def icrf(self, v):
        res = [[0,  -v[2],  v[1],    0,  -v[5],  v[4]],
            [v[2],    0,  -v[0],  v[5],    0,  -v[3]],
            [-v[1],  v[0],    0,  -v[4],  v[3],    0],
            [    0,  -v[5],  v[4],    0,    0,    0],
            [ v[5],    0,  -v[3],    0,    0,    0],
            [-v[4],  v[3],    0,    0,    0,    0]]
        return -np.asmatrix(res)


    def idsva_series(self, q, qd, qdd, GRAVITY = -9.81):
        # allocate memory
        n = len(qd) # n = 7
        print(n)
        v = np.zeros((6,n))
        a = np.zeros((6,n))
        f = np.zeros((6,n))
        Xup0 =  [None] * n #list of transformation matrices in the world frame
        Xdown0 = [None] * n
        S = np.zeros((6,n))
        Sd = np.zeros((6,n))
        Sdd = np.zeros((6,n))
        Sj = np.zeros((6,n)) 
        IC = [None] * n 
        BC  = [None] * n 
        gravity_vec = np.zeros(6)
        gravity_vec[5] = -GRAVITY # a_base is gravity vec
        
        # forward pass
        for i in range(n):
            parent_i = self.robot.get_parent_id(i)
            print("This is parent of i")
            print(parent_i )
            Xmat = self.robot.get_Xmat_Func_by_id(i)(q[i])
          # compute X, v and a
            if parent_i == -1: # parent is base
                Xup0[i] = Xmat
                a[:,i] = Xmat @ gravity_vec
            else:
                Xup0[i] = Xmat @ Xup0[parent_i]
                v[:,i] = v[:,parent_i]
                a[:,i] = a[:,parent_i]

            Xdown0[i] = np.linalg.inv(Xup0[i])

            S[:,i] = self.robot.get_S_by_id(i)

            S[:,i] = Xdown0[i] @ S[:,i]
            print("This is S with the matrix multiplication")

            Sd[:,i] = self.crm(v[:,i]) @ S[:,i]
            Sdd[:,i]= self.crm(a[:,i])@ S[:,i]
            Sdd[:,i] = Sdd[:,i] + self.crm(v[:,i])@ Sd[:,i]
            Sj[:,i] = 2*Sd[:,i] + self.crm(S[:,i]*qd[i])@ S[:,i]

            v6x6 = self.crm(v[:,i])
            v[:,i] = v[:,i] + S[:,i]*qd[i]  # Line 3 in Algorithm
            a[:,i] = a[:,i] + np.array(v6x6 @ S[:,i]*qd[i]) # Line 4 in Algorithm

            if qdd is not None:
                a[:,i] += S[:,i]*qdd[i]

            # compute f, IC, BC
            Imat = self.robot.get_Imat_by_id(i)

            IC[i] = np.array(Xup0[i]).T  @ (Imat @ Xup0[i])
            f[:,i] = IC[i] @ a[:,i] + self.crf(v[:,i]) @ IC[i] @ v[:,i]
            f[:,i] = np.asarray(f[:,i]).flatten()
            BC[i] = (self.crf(v[:,i])@IC[i] + self.icrf( IC[i] @ v[:,i]) - IC[i] @ self.crm(v[:,i]))
        

        t1 = np.zeros((6,n))
        t2 = np.zeros((6,n))
        t3 = np.zeros((6,n))
        t4 = np.zeros((6,n))
        dtau_dq = np.zeros((n,n))
        dtau_dqd = np.zeros((n,n))


        #backward pass
        for i in range(n-1,-1,-1):
            t1[:,i] = IC[i] @ S[:,i]
            t2[:,i] = BC[i] @ S[:,i].T + IC[i] @ Sj[:,i]
            t3[:,i] = BC[i] @ Sd[:,i] + IC[i] @ Sdd[:,i] + self.icrf(f[:,i]) @ S[:,i]
            t4[:,i] = BC[i].T @ S[:,i]

            subtree_ids = self.robot.get_subtree_by_id(i) #list of all subtree ids (inclusive)



            dtau_dq[i, subtree_ids[1:]] = S[:,i] @ t3[:,subtree_ids[1:]]
            
            dtau_dq[subtree_ids[0:], i] = Sdd[:,i] @ t1[:,subtree_ids[0:]] + \
                                        Sd[:,i] @ t4[:,subtree_ids[0:]] 
            
            dtau_dqd[i, subtree_ids[1:]] = S[:,i] @ t2[:,subtree_ids[1:]]

            dtau_dqd[subtree_ids[0:], i] = Sj[:,i] @ t1[:,subtree_ids[0:]] + \
                                        S[:,i] @ t4[:,subtree_ids[0:]] 

            p = self.robot.get_parent_id(i)
            if p >= 0:
                IC[p] = IC[p] + IC[i]
                BC[p] = BC[p] + BC[i]
                f[:,p] = f[:,p] + f[:,i]

        return dtau_dq, dtau_dqd







