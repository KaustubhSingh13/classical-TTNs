from TreeTensorNetworks1 import *
import contraction as ctr
from qr_decomposition1 import qr_tensors

def loc_list(ttn):
    '''
    Creates a list containing all the possible loc tuples that are possible
    '''
    loc_list = []
    for layer in range(ttn.layers):
        for pos in range(ttn.nroots*ttn.ncn**layer):
           loc_list.append(loc(ttn,[layer,pos])) 
    return loc_list

def shape_test(ttn):
    '''
    returns a list containing the shape of all the tensors in the ttn
    '''
    locs = loc_list(ttn)
    return [ttn.shape(l) for l in locs ]

def nbd_test(ttn):
    '''
    prints out the nbd tuple for each loc
    '''
    locs = loc_list(ttn)
    for l in locs:
        print(f'{l} --> {ttn.nbd(l)}')

def isometry_check(q,index):
    '''
    Checks if the tensor q has been isometrised with respect to the given index.
    '''
    lq = len(q.shape)
    assert lq > index, 'The index is out of range'
    #Include another assert statement for when it would be impossible to isometrise.

    q_shape = list(q.shape)
    contracting_indices = list(range(lq)); contracting_indices.pop(index) 

    ni = q_shape[index]
    
    qqt = np.tensordot(q,q.conj(), axes = [contracting_indices,contracting_indices])
    return np.allclose(qqt,np.eye(ni,ni))

def isometry_possible(q,index):
    isometric_index_dim = q.shape[index]
    contracted_index_dim = int(np.prod(q.shape)/isometric_index_dim)
    return contracted_index_dim >= isometric_index_dim # if the dimensionality of the vector space is less than the number of orthogonal vectors we propose to stuff in there then return False.

def decomposition_check(t,q,r,index):
    '''
    Checks if the T=QR decomposition has been performed correctly, with respect to the specified index
    '''
    t_shape = t.shape; q_shape = q.shape; r_shape = r.shape
    if t_shape != q_shape:
        return False, f'T.shape = {t_shape} but Q.shape = {q_shape}'

    if q_shape[index]!=r_shape[0]:
        return False, f'Q.shape = {q_shape} but R.shape = {r_shape}.\n The 0th index of R does not match with the {index}th index of Q.'

    if len(r_shape) != 2 or r_shape[0] != r_shape[1]:
        return False, f'R.shape = {r_shape}. \n This should be a square matrix.'

    qr = np.tensordot(q,r,axes = [index,0])
    ''' 
    The convention with tensordot is that it would place the free indices of r (which would be the 1st index) to the right of the free indices of q.
    This is why we would have to move the last index once more to the correct place in case we are not isometrising with respect to the last index.
    '''
    if index != len(t_shape)-1:     # by now t_shape is equal to q_shape.
        qr = np.moveaxis(qr,len(t_shape)-1,index)
    return np.allclose(t,qr)

def check_isometry_network(ttn,iso_loc,verbose=True):
    loc_list = ttn.loc_list
    for locc in loc_list:
        if locc != iso_loc:
            iso_index = ttn.iso_index(iso_loc,locc)
            if verbose:
                print(f'{str(locc)} iso_index : {iso_index} Isometrised? : {isometry_check(ttn.tensors[locc],iso_index)}')
            elif not isometry_check(ttn.tensors[locc], iso_index):
                print(f'{str(locc)} iso_index : {iso_index} is not isometrised')



def check_isometry_sweep(ttn, num_sweeps = 50):
    '''
    For each tensor in the network,
        sets it to be the iso loc
        then for every other tensor in the network
            changes the iso centre to be the other tensor through shift_iso_centre
            and then checks if the network has been isometrised properly through check_isometry_network
    '''
    ttn.randn_init()
    sweep = ttn.sweep
    ttn.set_iso(sweep[0])
    for i in range(num_sweeps):
        print('\nnew sweep')
        for j in range(1,len(sweep)):
            ttn.shift_isometrisation_centre(sweep[j-1],sweep[j])
            check_isometry_network(ttn,sweep[j],verbose=False)
            print(f'iso centre is now {sweep[j]}')
        ttn.shift_isometrisation_centre(sweep[-1],sweep[0])
        check_isometry_network(ttn, sweep[0],verbose=False)
        print(f'iso centre is now {sweep[0]}')
def coarse_isometry_check(ttn):
    '''
    For each loc in the network, 
        randomly initialises the network
        isometrieses the TN wrt the chosen loc
        checks if each tensor has been isometrised wrt the self.iso_index
        If any anomalies, tells you where the issue is.
    '''

    ll = ttn.loc_list
    for iso_l in ll:
        ttn.flush()
        ttn.randn_init()
        ttn.set_iso(iso_l)
        print(f'\niso loc is {iso_l}')
        for l in ll:
            if l != iso_l:
                print(f'\tloc is {l}')
                t = ttn.tensors[l]
                if not isometry_check(t,ttn.iso_index(iso_l,l)):
                    return f'when iso_loc = {iso_l}, the tensor at {l} was not isometrised wrt {ttn.iso_index(iso_l,l)}'
        
    print('VROK')

def iso_index_check(ttn):
    ll = ttn.loc_list
    for iso_l in ll:
        print('\n')
        print(f'Iso loc is {iso_l}')
        for l in ll:
            print(f'\tiso index for {l} is {ttn.iso_index(iso_l,l)}')
            print



def coarse_contraction_isometry_check(ttn):
    '''
    For each loc in the network, 
        randomly initialises the network
        isometrieses the TN wrt the chosen loc
        contacts the network around the chosen loc (effective network) and checks if each tensor in the effective network is isometrised
        If any anomalies, tells you where the issue is.
    '''

    ll = ttn.loc_list
    for iso_l in ll:
        ttn.flush()
        ttn.randn_init()
        ttn.set_iso(iso_l)
        print(f'\niso loc is {iso_l}')

        effn = ctr.effective_network(ttn,iso_l)
        for ii,ten in enumerate(effn):
            if not isometry_check(ten,0):
                    return f'when iso_loc = {iso_l}, the {i}th element of effn was not isometrised wrt {0}'
        
    print('VROK')
