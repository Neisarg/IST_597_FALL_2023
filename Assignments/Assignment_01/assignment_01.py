###########################################
# IST 597 : Foundations of Deep Learning
# 
#          Assignment 01 
#
# Instructor : Dr. C. Lee Giles
# TA : Neisarg Dave
###########################################

import numpy as np

# Note: Do not change the function names

# problem 3
def indexing():
    a = np.array([9, 8, 15, 17, 14, 2, 15, 7, 12, 1, 5, 1, 8, 16, 3, 15, 8, 15, 3, 5])
    # write your code here
    #-------------------
    
    res = None # update res with correct output

    return res


# probelem 4
class NNLayer:
    def __init__(self):
        self.x = None   # create array x
        self.W = None   # create matrix W 
 
    def x_transpose(self):
        # write your code here 
        #----------------------

        x_t = None # update this with correct output
        
        return x_t

    def matmul(self):
        # write your code here 
        #----------------------

        z = None # update this with correct output


        self.z = z
        return z

    def non_linearity(self):
        def softmax(x):
            # write your code here 
            #----------------------
            
            res = None #update this with correct output
            return res

        # compute sigmoid(z) here:
        #----------------------
        softmax_z = None #update this with correct output

        return softmax_z



if __name__ == "__main__":
    print("problem 3 : ", indexing())
    q4 = NNLayer()
    print("problem 4.1 : x^T = ", q4.x_transpose())
    print("problem 4.2 : z = Wx^T = ", q4.matmul())
    print("problem 4.3 : softmax(z) = ", q4.non_linearity())
    