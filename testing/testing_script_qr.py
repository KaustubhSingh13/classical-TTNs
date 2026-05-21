from testing import *
from qr_decomposition import qr_tensors
import numpy as np

#Testing for t.ndim = 2.
'''
ndim = 2
cap = 20

shape_list = []
for nrows in range(2,cap):
    for ncols in range(2,cap):   
        shape_list.append([nrows,ncols])


iso_success = []    # True iff Q was successfully isometrised wrt to the appropriate index.
tqr_success = []    # True iff T = QR wrt the appropriate index.
iso_possibility = []
# I will check for each index (hence these two will be nested lists)

for shape in shape_list:
    T = np.random.randn(*shape)
    for index in range(ndim):
        out1,out2, out3 = [],[],[]
        try:        
            q,r = qr_tensors(T,index)
            out1.append(isometry_check(q,index))
            out2.append(decomposition_check(T,q,r,index))
            out3.append(isometry_possible(q,index))
        except ValueError:
            out1.append('NA')
            out2.append('NA')
    iso_possibility.append(out3)
    iso_success.append(out1)
    tqr_success.append(out2)
''' 

def int_to_base(n,base):
    '''
    Takes an integer n, and gives its representation in the given base.
    if are in base 16, then 16 (base 10) -> (15,0) (base 16). 
    Basically here 15 is treated as an immutable character.
    '''
     
# Testing for t.ndim = 3
# Testing when isometrised with respect to the last index.
def complete_check(ndim, index, lo, hi):
    
    shape_list = []
    
    assert index<ndim, 'Index must be less than ndim'
    
    for i in range(lo,hi):  # Generalise these nested loops by changing the base.
        for j in range(lo,hi):
            for k in range(lo,hi):
                for l in range(lo,hi):
                    shape_list.append([i,j,k,l])

    # Each shape corresponds to an integer, the index of the shape_list. That is the common identifier. 

    iso_success = []    # True iff Q was successfully isometrised wrt to the appropriate index.
    tqr_success = []    # True iff T = QR wrt the appropriate index.
    iso_possibility = []

    for shape in shape_list:
        t = np.random.randn(*shape)
        iso_possibility.append(isometry_possible(t,index))
        try:
            q,r = qr_tensors(t,index)
            iso_success.append(isometry_check(q,index))
            tqr_success.append(decomposition_check(t,q,r,index))
        except:
            iso_success.append(False)
            tqr_success.append(False)
    print(f'iso_success == iso_possibility? : {iso_success == iso_possibility}\ntqr_success == iso_success? : {tqr_success == iso_success}')
    return(shape_list,iso_success,tqr_success,iso_possibility)
