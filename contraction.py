import numpy as np
from misc import *



def smallest_node_position(ttn,loc): #TEst this.
    '''
    to be used in effective_network to sort the neighbourhoods.
    returns the smallest position of all the leaf nodes that originates from loc.
    '''
    layer,pos = loc
    return pos*ttn.ncn**(ttn.layers - layer)

def stripped_nbd(ttn, loc:loc, prev_loc_list:list): # not tested.
    '''
    gives the ttn.nbd of loc, minus the nbd_element containing an element of prev_lov which is in the nbd of loc 

    (prev_loc_list must contain a loc instance which is in the neighbourhood of loc)
    
    in our usecases, prev_loc_list will contain only one such loc object.
    '''
    loc_neighbourhood = ttn.nbd(loc)
    nbd_locs = [nbd_elem[1] for nbd_elem in loc_neighbourhood] 

    for i,x in enumerate(nbd_locs):     # this double for loop can be eliminated if we use sets. we would have to make loc hashable.
        for y in prev_loc_list:
            if x==y:
                #prev_loc = x        # one such x is guaranteed to exist by construction, in our usecases.
                loc_neighbourhood.pop(i)
                loc_neighbourhood.sort(key=lambda nbd_elem: smallest_node_position(ttn,nbd_elem[1]))

                return loc_neighbourhood
   
    loc_neighbourhood.sort(key=lambda nbd_elem: smallest_node_position(ttn,nbd_elem[1]))
    return loc_neighbourhood        # I have put this extra one here in case prev_loc_list does not have an intersection with the nbd of the given element. That will practically never happen. The nested return statement would be get executed and be faster. 

def list_to_nested_list(lst):
    '''
        Takes a list and returns a list
        with all the elements of the previous list as singleton list elements
        needed to construct the stripped_neighbourhood_family from the stripped neighbourhood initally
    '''
    out = []
    for x in lst:
        out.append([x])
    return out


def int_to_alphabet(x):         # could use lambda functions in their place maybe
    '''
    0 -> a
    1 -> b
    ...
    25 -> z

    needed for einsum.
    '''
    assert type(x)==int, f'input needs to be an integer, you gave me {type(x)}'
    return chr(97+x)

def sum_str_list(lst:list) -> str:
    s = ''
    for x in lst:
        s += str(x)
    return s

def number_of_trailing_zeros(x:int,lst:list) -> int:
    '''
            assumes that all elements inside lst are int
            assumes that all elements other than 0 occur only once.

            returns the number of zeros after the element x before the next nonzero element or before lst terminates.
    '''
    # there can be a faster way to do this. I am going with the most straighforward way.
    s=0
    flag = x==0
    for y in lst:
        if not flag and y==x:
            flag = True
        elif flag and y==0:
            s+=1
        elif y!=x and flag:
            return s
    return s
 
def reorder_indices(ttn, tensor:np.ndarray, leaf_tensor_order:list) -> np.ndarray:  # not tested.
    '''
            Reorders the indices of tensor
            leaf_tensor_order contains loc objects in the order in which their indices appear in tensor.
            
            returns a ndarray in which the indices of tensor have been sorted wrt the position of the locs. (so that the tensor can be plugged into the hamiltonain to help evaluate the local effective hamiltonain.)
    '''
    pos_order = [locc[1] for locc in leaf_tensor_order]     #the positions 
    # I will be implementing a simple bubble sort algorithm. Obviously better algorithms exist. This function can be optimised further.

    for i in range(len(pos_order)-1):
        swapped = False
        for j in range(len(pos_order)-i-1):
            if pos_order[j] > pos_order[j+1]:

                # now performing the swaps in the tensor
                for k in range(ttn.ncn):			#Here
                    tensor = np.swapaxes(tensor, ttn.ncn*j+k+1, ttn.ncn*(j+1)+k+1)  # +1: axis 0 is bond dim
                pos_order[j], pos_order[j+1] = pos_order[j+1], pos_order[j]          # keep sort state consistent
                swapped = True
        
        if not swapped:
            break

    return tensor

def initial_reordering(ttn, contraction, stripped_neighbourhood, branch_loc):       # there should be a way of knowing when the indices are already in the desired order, so that we can skip this for those cases.

        # changing the index placement of contraction 
        if len(stripped_neighbourhood) == 0:
            return contraction      # dont need to reorder if in the leaf layer.

        final_index_placement = [nbd_elem[0][0] for nbd_elem in stripped_neighbourhood]
        missing_index = set(range(ttn.num_legs(branch_loc))) - set(final_index_placement)     #ttn.num_legs(branch_loc) = len(final_index_placement) + 1 because the index pointing towards the centre (loc) has been stripped.
        # the missing_index is the index pointing towards the centre.
        
        missing_index, = missing_index # unpacking the singleton set.
        final_index_placement.insert(0,missing_index)
        
        final_alphabetic_indices = sum_str_list([int_to_alphabet(i) for i in final_index_placement])
        init_alphabetic_indices = sum_str_list([int_to_alphabet(i) for i in range(len(final_alphabetic_indices))])

        return np.einsum(f'{init_alphabetic_indices}->{final_alphabetic_indices}',contraction)

def lengths_to_index_list(lengths:list) -> list:
    '''
        Takes in a list of integers which might have zeros
        replaces all slices separated by 0s by a running sum of the integerrs
        [0,0,3,4,0,0,2,3,0,5,0,0,0] -> [0,0,7,0,0,12,0,17,0,0,0]
    '''
    s = 0
    out = []
    in_nonzero_run = lengths[0]!=0
    for x in lengths:
        if x==0 and not in_nonzero_run:
            in_nonzero_run = False
            out.append(0)
        
        elif x==0 and in_nonzero_run:
            in_nonzero_run = False
            out += [s,0]
        else:
            in_nonzero_run = True
            s+=x
    if in_nonzero_run:
        out.append(s)
    return out
"""
def effective_network(ttn,loc):
    '''
    returns a list of ttn.num_legs(loc) tensors (if loc is not in the leaf layer) or 1 tensor (if loc is in leaf layer),
    formed by contracting the entire ttn around (and excluding) loc.

    '''
    loc_neighbourhood = ttn.nbd(loc)
    out = []

    
    for nbd_element in loc_neighbourhood:
        
        prev_loc_list = [loc]       # prev_loc_list needs to be refreshed every time we hit a new nbd_element.

        branch_loc = nbd_element[1]

        #print(f'branch loc is {branch_loc}')
        stripped_neighbourhood = stripped_nbd(ttn,branch_loc,prev_loc_list)     
        contraction = ttn.tensors[branch_loc]
        
        prev_loc_list.append(branch_loc)
        #print(f'shape before init reordering is {contraction.shape}')
        contraction = initial_reordering(ttn, contraction, stripped_neighbourhood,branch_loc)
        #print(f'shape after init reordering is {contraction.shape}')
        '''
        by now contraction should be the tensor at branch_loc with the index pointing towards the centre (loc)
        at axis 0, the first axis corresponding to the index which will be contracted with the first tensor in stripped_nbd
        second axis (index) corresponding to the index which will be contracted with the second tensor in stripped_nbd etc.

        '''
        skip_index = 0
        stripped_neighbourhood_family = list_to_nested_list(stripped_neighbourhood)      # stripped_neighbourhood will be the union of all the elements of stripped_neighbourhood_family.
        indices_list = []                    # indices_list will contain the info needed to know when to activate skip_index

        leaf_tensor_order = []     # this will contain the order in which we arrive at the leaf tensors. will be used for reordering the indices.

        while len(stripped_neighbourhood) > 0:
            #print("\nNEW WHILE LOOP ITERATION\n")

            #for testing; delete later
            delll = []
            for x in stripped_neighbourhood_family:
                delll_t = []
                for nbd_el in x:
                    delll_t.append(nbd_el[1])
                delll.append(delll_t)
            #print(f'locs in stripped nbd family {delll}')
            #for testing; delete later

            #creating the list needed for skip_index
            lengths = [len(x) for x in stripped_neighbourhood_family ]
            
            indices_list = lengths_to_index_list(lengths)
            #print(f'lengths is {lengths} and indicies list is {indices_list}')

            n = len(stripped_neighbourhood)
            
            new_stripped_neighbourhood_family = []
            new_stripped_neighbourhood = [] 
            new_in_leaf = []        # i think this line can be deleted.

            for i,nbd_element in enumerate(stripped_neighbourhood):
                #print('\n\tNEW FOR LOOP\n')
                if i in indices_list:
                    skip_index += ttn.ncn * number_of_trailing_zeros(i,indices_list)
                indices, child_loc = nbd_element
                #print(f'\tthe child loc is {child_loc}')
                child_tensor = ttn.tensors[child_loc]
                
                #print(f'\tshape of contraction is {contraction.shape}')
                #print(f'\tnow {child_loc} ka {indices[1]} leg with contraction ka {1+skip_index}  ')
                print(contraction.shape, child_tensor.shape, [1+skip_index,indices[1]])
                contraction = np.tensordot(contraction,child_tensor,axes=[1+skip_index,indices[1]] )  
                #print(f'shape of contraction post is {contraction.shape}')
                #print('\n')
                prev_loc_list.append(child_loc)         # you could change prev_loc_list to prev_loc_set. That data type would be more appropriate.
                
                # updating new_stripped_neighbourhood_family and new_stripped_neighbourhood
                temp = stripped_nbd(ttn,child_loc,prev_loc_list)
                new_stripped_neighbourhood_family.append(temp)
                new_stripped_neighbourhood += temp
                
                if nbd_element[1][0] == ttn.layers - 1:      # leaf layer is layers-1
                    leaf_tensor_order.append(nbd_element[1])	#Here

            stripped_neighbourhood = new_stripped_neighbourhood
            stripped_neighbourhood_family = new_stripped_neighbourhood_family

        contraction = reorder_indices(ttn,contraction,leaf_tensor_order)    
        out.append(contraction)

    return out
"""

def effective_network(ttn, loc):
    '''
    Returns a list of ttn.num_legs(loc) tensors formed by contracting the entire
    ttn around (and excluding) loc.

    axis_for[i] tracks the exact axis in `contraction` corresponding to
    stripped_neighbourhood[i], and is updated after every tensordot.

    Before contracting each child tensor, initial_reordering is applied so
    that its remaining axes are appended in the sorted order that matches the
    next iteration's stripped_neighbourhood — fixing the root cause of the
    shape mismatch caused by the old skip_index mechanism.
    '''
    loc_neighbourhood = ttn.nbd(loc)
    out = []

    for nbd_element in loc_neighbourhood:

        prev_loc_list = [loc]
        branch_loc = nbd_element[1]

        stripped_neighbourhood = stripped_nbd(ttn, branch_loc, prev_loc_list)
        contraction = ttn.tensors[branch_loc]
        prev_loc_list.append(branch_loc)

        contraction = initial_reordering(ttn, contraction, stripped_neighbourhood, branch_loc)

        # axis_for[i] = the current axis of `contraction` corresponding to
        # stripped_neighbourhood[i]. After initial_reordering this is simply 1,2,3,...
        axis_for = list(range(1, len(stripped_neighbourhood) + 1))

        leaf_tensor_order = []

        while len(stripped_neighbourhood) > 0:

            new_stripped_neighbourhood = []
            new_axis_for = []

            for i, nbd_elem in enumerate(stripped_neighbourhood):
                indices, child_loc = nbd_elem
                child_tensor = ttn.tensors[child_loc]

                # Compute child's stripped_nbd BEFORE appending child_loc to prev_loc_list.
                # This is used both to reorder child_tensor and as the next-level
                # neighbourhood contribution.
                child_stripped = stripped_nbd(ttn, child_loc, prev_loc_list)

                # Reorder child_tensor so that:
                #   axis 0  = leg pointing back toward our contraction (the "bond to contract")
                #   axes 1+ = remaining legs in sorted stripped_nbd order
                # This guarantees the appended axes are in the order the next
                # while-loop iteration expects — the core fix.
                child_tensor = initial_reordering(ttn, child_tensor, child_stripped, child_loc)

                target_axis = axis_for[i]

                # initial_reordering always puts the bond-to-parent at axis 0,
                # so we always contract along axis 0 of child_tensor.
                contraction = np.tensordot(contraction, child_tensor, axes=[target_axis, 0])

                # After removing axis target_axis, every axis > target_axis in the
                # remaining stripped_neighbourhood elements shifts left by 1.
                for j in range(i + 1, len(stripped_neighbourhood)):
                    if axis_for[j] > target_axis:
                        axis_for[j] -= 1

                # Do the same adjustment for new elements already registered
                # from earlier iterations of this for-loop.
                for k in range(len(new_axis_for)):
                    if new_axis_for[k] > target_axis:
                        new_axis_for[k] -= 1

                # The child's remaining axes (child_tensor.ndim - 1 of them) are
                # appended at the very end of the new contraction.
                n_new =  len(child_stripped)
                start_new = contraction.ndim - n_new
                for k in range(n_new):
                    new_axis_for.append(start_new + k)

                new_stripped_neighbourhood += child_stripped
                prev_loc_list.append(child_loc)

                if child_loc[0] == ttn.layers - 1:
                    leaf_tensor_order.append(child_loc)

            stripped_neighbourhood = new_stripped_neighbourhood
            axis_for = new_axis_for

        contraction = reorder_indices(ttn, contraction, leaf_tensor_order)
        out.append(contraction)

    return out
#Here
def effective_network_full(ttn, locc): 
    """
    Extends effective_network so that for a leaf node, the ncn physical legs
    are also represented as trivial identity environment tensors.
    Returns a list of env tensors, one per leg of locc (in nbd order, then physical legs).
    """
    env_tensors = effective_network(ttn, locc)

    if locc[0] == ttn.layers - 1:  # leaf node: physical legs have no sub-network below
        for _ in range(ttn.ncn):
            d = ttn.d
            # shape [d, d]: axis 0 acts as the "bond" (= d), axis 1 is the physical site
            env_tensors.append(np.eye(d))

    return env_tensors


def env_operator_sandwich(env_tensor, operators, local_axes):
    """
    env_tensor  : shape [bond_dim, d_0, d_1, ..., d_n]
    operators   : list of (d,d) arrays
    local_axes  : list of ints — which physical axes (0-indexed) each operator acts on
    Returns     : shape [bond_dim, bond_dim]
    """
    bond_dim   = env_tensor.shape[0]
    phys_shape = env_tensor.shape[1:]
    D_phys     = int(np.prod(phys_shape))

    full_op = np.array([[1.0]])
    for ax, d in enumerate(phys_shape):
        if ax in local_axes:
            op = operators[local_axes.index(ax)]
            full_op = np.kron(full_op, op)
        else:
            full_op = np.kron(full_op, np.eye(int(d)))

    env_flat = env_tensor.reshape(bond_dim, D_phys)
    return env_flat @ full_op @ env_flat.conj().T     # [bond_dim, bond_dim]


def build_H_eff(ttn, locc, env_tensors, hamiltonian):
    """
    Constructs H_eff at locc from the environment tensors returned by effective_network_full
    and the Hamiltonian terms.

    env_tensors : list of [bond_dim_k, phys_0, phys_1, ...] arrays, one per leg of locc
    hamiltonian : tfIsingH instance
    Returns     : H_eff as a (D, D) numpy array where D = product of all bond dims of locc
    """
    n_legs    = ttn.num_legs(locc)
    bond_dims = [ttn.bdim(locc, leg) for leg in range(n_legs)]
    D         = int(np.prod(bond_dims))
    H_eff     = np.zeros((D, D))

    nbd      = ttn.nbd(locc)
    site_map = {}   # global_physical_site_index -> (env_idx, local_axis)

    for env_idx, (_, branch_loc) in enumerate(nbd):
        if branch_loc[0] > locc[0]:        # child branch
            sites = sorted(ttn.leaf_indices(branch_loc))
        else:                              # parent branch: everything outside locc's subtree
            all_sites = set(range(ttn.nroots * ttn.ncn**ttn.layers))
            sites     = sorted(all_sites - set(ttn.leaf_indices(locc)))

        for local_axis, s in enumerate(sites):
            site_map[s] = (env_idx, local_axis)

    if locc[0] == ttn.layers - 1:
        base_pos = locc[1] * ttn.ncn      # first physical site of this leaf
        for j in range(ttn.ncn):
            global_site = base_pos + j
            env_idx     = len(nbd) + j    # appended after bond envs
            site_map[global_site] = (env_idx, 0)

    def contribution(env_matrices):
        """
        env_matrices : dict  env_idx -> (bond_dim, bond_dim) matrix M
        Missing legs get an identity of the appropriate bond_dim.
        Returns the full Kronecker product.
        """
        out = np.array([[1.0]])
        for l in range(n_legs):
            mat = env_matrices.get(l, np.eye(bond_dims[l]))
            out = np.kron(out, mat)
        return out
    #single site terms
    for (site, op) in hamiltonian.single_site_terms():
        if site not in site_map:
            continue
        env_idx, local_axis = site_map[site]
        M = env_operator_sandwich(env_tensors[env_idx], [op], [local_axis])
        H_eff += contribution({env_idx: M})

    # Two-site terms 
    for (op_i, op_j, i, j) in hamiltonian.two_site_terms():
        if i not in site_map or j not in site_map:
            continue
        env_i, local_i = site_map[i]
        env_j, local_j = site_map[j]

        if env_i == env_j:
            # Both sites live in the same env branch 
            M = env_operator_sandwich(env_tensors[env_i], [op_i, op_j],
                                      [local_i, local_j])
            H_eff += contribution({env_i: M})
        else:
            M_i = env_operator_sandwich(env_tensors[env_i], [op_i], [local_i])
            M_j = env_operator_sandwich(env_tensors[env_j], [op_j], [local_j])
            H_eff += contribution({env_i: M_i, env_j: M_j})

    return H_eff



def apply_H_eff(v, ttn, locc, env_tensors, hamiltonian):
    """
    Applies H_eff to vector v without forming the full matrix.
    Exploits the Kronecker structure: each term is a product of per-leg matrices.
    Each per-leg matrix is applied via tensordot on the reshaped vector.
    
    This avoids the O(D^2) memory cost of the dense matrix.
    """
    n_legs    = ttn.num_legs(locc)
    bond_dims = [ttn.bdim(locc, leg) for leg in range(n_legs)]

    # ── Build site_map (same as build_H_eff) ─────────────────────────────────
    nbd      = ttn.nbd(locc)
    site_map = {}

    for env_idx, (_, branch_loc) in enumerate(nbd):
        if branch_loc[0] > locc[0]:
            sites = sorted(ttn.leaf_indices(branch_loc))
        else:
            all_sites = set(range(ttn.nroots * ttn.ncn**ttn.layers))
            sites     = sorted(all_sites - set(ttn.leaf_indices(locc)))
        for local_axis, s in enumerate(sites):
            site_map[s] = (env_idx, local_axis)

    if locc[0] == ttn.layers - 1:
        base_pos = locc[1] * ttn.ncn
        for j in range(ttn.ncn):
            site_map[base_pos + j] = (len(nbd) + j, 0)

    # ── Helper: apply one term (dict of per-leg matrices) to v ───────────────
    def apply_term(v, leg_matrices):
        """
        leg_matrices: dict {leg_index -> (bond_dim, bond_dim) matrix}
        Applies the Kronecker product to v by acting on one leg at a time.
        """
        result = v.reshape(bond_dims)
        for leg in range(n_legs):
            mat = leg_matrices.get(leg, None)
            if mat is None:
                continue  # identity — skip
            # contract mat with result along axis `leg`
            result = np.tensordot(mat, result, axes=[[1], [leg]])
            # tensordot puts the new axis at position 0; move it back to `leg`
            result = np.moveaxis(result, 0, leg)
        return result.ravel()

    result = np.zeros_like(v)

    # ── Single-site terms ─────────────────────────────────────────────────────
    for (site, op) in hamiltonian.single_site_terms():
        if site not in site_map:
            continue
        env_idx, local_axis = site_map[site]
        M = env_operator_sandwich(env_tensors[env_idx], [op], [local_axis])
        result += apply_term(v, {env_idx: M})

    # ── Two-site terms ────────────────────────────────────────────────────────
    for (op_i, op_j, i, j) in hamiltonian.two_site_terms():
        if i not in site_map or j not in site_map:
            continue
        env_i, local_i = site_map[i]
        env_j, local_j = site_map[j]

        if env_i == env_j:
            M = env_operator_sandwich(env_tensors[env_i], [op_i, op_j],
                                      [local_i, local_j])
            result += apply_term(v, {env_i: M})
        else:
            M_i = env_operator_sandwich(env_tensors[env_i], [op_i], [local_i])
            M_j = env_operator_sandwich(env_tensors[env_j], [op_j], [local_j])
            result += apply_term(v, {env_i: M_i, env_j: M_j})

    return result



        


        
        


