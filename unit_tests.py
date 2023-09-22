
import random
import numpy as np
from RBDReferenceSA import RBDReference as RBDReferenceSA, RBDReference
#from RBDReference.RBDReference import RBDReference
from URDFParser import URDFParser




def compare( s, q, qd, qdd): #input: RBDReference objects; random joint inputs

    #get outputs for IDSVA and rnea_grad
    #a = t.rnea_grad(q, qd, qdd)
    b1, b2 = s.idsva_series(q, qd, qdd) #(IDSVA)
    # b1 = dtau_dq, b2 = dtau_dqd
    print("The value of b1 is which is dtau_dq")
    print(b1)
    print("The value of b2 is which is dtau_dqq")
    print(b2)
    #a1, a2 = s.idsva_parallel_1(q, qd, qdd)

    #compare outputs
    #a = np.hstack((np.array(a1), np.array(a2)))
    b = np.hstack((np.array(b1), np.array(b2)))
    print("This is horizontally stacked array ")
    print(b)
    #c = a-b
    #res = np.all(c > -0.0001) and np.all(c < 0.0001)

    return q, qd, qdd



def gen_inputs(n): #randomly generate inputs for n joints
    q, qd, qdd = [], [], []

    for i in range(n):
        x1 = random.randint(-100,100)
        x2 = random.randint(-100,100)
        x3 = random.randint(-100,100)
        q.append(x1/100)
        qd.append(x2/100)
        qdd.append(x3/100)
    
    return q, qd, qdd


def test(filepath, num):

    #parse the urdf file
    parser = URDFParser()
    robot = parser.parse(filepath, alpha_tie_breaker = False)
    n = robot.get_num_joints()
    print("The number of joints of the robot is : ")
    print(n)
    #create RBDReference objects
    #t = RBDReference(robot)
    s = RBDReferenceSA(robot) 
    print("This is when the s=RBDReferenceSA(robot) is called")
    #run multiple comparisons
    for _ in range(num):
        q, qd, qdd = gen_inputs(n)
        print("We are printing the inputs which we will pass")
        print(qdd)
        r = compare( s, q, qd, qdd)
        print("Does everything starts after compare??")
        if r[0] == False:
            print("Failed when \nq = "+ str(r[1])+ "\nqd = "+
            str(r[2])+ "\nqdd = "+ str(r[3]))
            return

    print("Passed")
    return
    

def main():
    filepath1 = "/home/a1bhi/Some Fancy Name/RBDReference_dev/atlas.urdf"
    filepath2 = "iiwa.urdf"

    filepath = filepath2
    num = 1 #number of comparisions

    test(filepath, num)   



if __name__ == "__main__":
    main()







