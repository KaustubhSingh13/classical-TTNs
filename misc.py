

class tlist(list):
    #Code from user Raniz' answer on https://stackoverflow.com/questions/30341008/is-it-possible-to-index-nested-lists-using-tuples-in-python

    def __getitem__(self, index):
        if hasattr(index, "__iter__"):
            #index is list-like, traverse downwards
            item = self
            for i in index:
                item = item[i]
            return item
        # index is not list-like, let list.__getitem__ handle it
        return super().__getitem__(index)
    
    def __setitem__(self,index,value):
        # could generalise this if you want but no use case.
        self[index[0]][index[1]] = value


def parent_location(loc):
    #Generalise this for an arbitrary tree structure by also taking in the tree structure as an arugment.
    #(((),()),((),())) is an example of a binary tree structure of two layers.
    '''
    PARAMETERS :

    loc : The location tuple of a node in a binary tree.

    RETURNS :

    The location tuple of the parent node in the binary tree.
    '''
    assert loc[0]>0, 'A root node has no parent node.'
    return loc[0] - 1 , loc[1]//2
    

def adrs(loc):
    #Generalise this for an arbitrary tree structure by also taking in the tree structure as an arugment.
    #(((),()),((),())) is an example of a binary tree structure of two layers.
    '''
    PARAMETERS :

    loc : The location tuple of a node in a binary tree.

    RETURNS :

    The bitstring address.

    (0,0) --> ''
    
    (1,0) --> '0'
    (1,1) --> '1'
    
    (2,0) --> '00' 
    (2,1) --> '01'
    (2,2) --> '10'
    (2,3) --> '11'
    '''
    
    ad = ''
    s = loc[1]
    for i in range(loc[0]+1):
        w = s%self.ncn
        s = s//self.ncn
        ad += str(w)
    ad = ad[::-1]
    return ad

class loc(tuple):
    """
    A location tuple in a  binary tree,
    which validates that the location is within valid bounds.

    Arguments:
        ttn: An object representing the tree, expected to have a `.layers` attribute.
        iterable: A 2-tuple (layer, index) representing the node's location.
    """

    def __new__(cls, ttn, iterable=()):
        #You can try to generalise this by taking in the tree type as an arugment.
        if len(iterable) != 2:
            raise TypeError("loc requires a 2-element iterable: (layer, index)")

        layer, index = iterable

        # Validate tree bounds
        if not (-ttn.layers <= layer <= ttn.layers - 1):
            raise ValueError(f"Invalid layer index {layer} for a tree with {ttn.layers} layers")
 
        #how do I give these attributes to my objects?
        
        if layer < 0:
            layer = ttn.layers + layer

        max_index = ttn.nroots * ttn.ncn ** layer - 1                            #Fix this in case you decide to have a single root node.
        min_index = -ttn.nroots * ttn.ncn ** layer                                #Fix this in case you decide to have a single root node.

        if not (min_index <= index <= max_index):
            raise ValueError(f"Invalid position index {index} for layer {layer}")

        # Create and return the tuple
        obj = super().__new__(cls, iterable)
       
        obj.ttn = ttn
        obj.layers = ttn.layers                                    # Number of layers in the tree.
        obj.nroots = ttn.nroots                                    # Number of tensors on the top of the tree.
        obj.ncn = ttn.ncn                                          # Number of child nodes for each tensor. IS THIS REALLY NEEDED?

        return obj 
    
    @property
    def parent_location(self):
        #Generalise this for an arbitrary tree structure by also taking in the tree structure as an arugment.
        #(((),()),((),())) is an example of a binary tree structure of two layers.
        '''
        PARAMETERS :

        self : The location tuple of a node in a tree.

        RETURNS :

        The location tuple of the parent node in the binary tree.
        '''
        if self[0]==0:
            raise ValueError('A root node has no parent node.')
        return loc(self.ttn, (self[0] - 1 , self[1]//self.ncn))
    
    def num_nodes(self,layer):
        '''
        PARAMETERS :
        
        layer   :   The layer index ( = loc[0] ). The topmost layer has layer index = loc[0] = 0.
        
        RETURNS :
        
        The number of tensors in the given layer.
        '''
        if hasattr(layer, "__iter__"):
            #Incase a location tuple is passed instead.
            layer = layer[0]
    
        assert -self.layers <= layer <= self.layers -1, f'{layer} is not a valid layer index for a ttn with {self.layers} layers.'
        return( self.nroots*self.ncn**layer )
    
    @property
    def adrs(self):
        #Generalise this for an arbitrary tree structure by also taking in the tree structure as an arugment.
        #(((),()),((),())) is an example of a binary tree structure of two layers.
        
        #Fix this
        
        '''
        PARAMETERS :

        loc : The location tuple of a node in a tree.

        RETURNS :

        The bitstring address.
        
        For a binary tree with one root,
        
        (0,0) --> '0'
    
        (1,0) --> '00'
        (1,1) --> '01'
    
        (2,0) --> '000' 
        (2,1) --> '001'
        (2,2) --> '010'
        (2,3) --> '011'
        '''
        '''
        ad = ''
        s = self[1]
        for i in range(self[0]+1):
            w = s%self.ncn
            s = s//self.ncn
            ad += str(w)
        ad = ad[::-1]
        return ad
        '''
        if self[0] == 0:
            return str(self[1])
        else: #Something is wrong here.
            return self.parent_location.adrs + str(self[1]%self.ncn)

    def aloc(self,adr):
        '''
        Returns the loc tuple given the bitstring address.
        '''
        #Maybe create a class for the bitstring address?
        lyr = len(adr) - 1
        pos = 0
        for i in range(0,lyr+1):
            pos = pos**self.ncn + int(adr[i])
        return lyr,pos

    def dist(self,other): # Test this.
        ad1 = self.adrs
        ad2 = other.adrs

        dif = abs(self[0] - other[0]) #The difference in the layer number
        rdif = abs(int(ad1[0]) - int(ad2[0])) #The difference in the root number

        #Now we can calculate the distance as if they have the same layer number and are in the same root, and add rdif and dif later.

        ca = min(self[0],other[0])
        i = 0
        for l1,l2 in zip(ad1,ad2):
            if l1==l2:
                if i>0:
                    ca -= 1
                i+=1
            else:
                break
        return ca*2 +rdif +dif
    
    # Redefining equality so that a loc[1] index and the corresponding negative index are equal.
    def __eq__(self, other):  # this is breaking hashability. You do not need this redfinition of __eq__ .  Get rid of this entire class. 
        if isinstance(other,loc):
            if self[0] == other[0] or abs(self[0] - other[0]) == self.layers :
                if abs(self[1] - other[1]) in {0, self.nroots*self.ncn**self[0]}:
                    # abs() because reflexivity should hold in equality.
                    return True
            return False
        return NotImplemented

    
'''
class loc(tuple):
    #create a special tuple type that raises an error when the location is invalid (1,199) is an invalid location in a binary tree.
    #This should take the tree type as an argument. Default should be a binary tree.
    #have methods that would return the bitstring address given the location.
    
    def __new__(cls, ttn , iterable = ()):
        obj = super().__new__(cls,iterable)
        self.layers = ttn.layers
        assert -2**(self.iterable[0] + 1) <= self.iterable[1] <= 2**(self.iterable[0]+1) - 1, 'Invalid position index {} for a layer number {}'.format(self.iterable[1],self.iterable[0])                      #Change here incase you decide to have a binary tree with one root.
        assert -self.layers <= self.iterable[0] <= self.layers - 1,  'Invalid layer index {} for a tree with {} layers'.format(self.iterable[0],ttn.layers)
'''
'''
#debug this
class tlist(nlist):
    def __getitem__(self,loc):
        assert loc[1] <= 2**(loc[0]+1) - 1, 'The position index {} exceeds the greatest possible number of nodes {} for layer {}'.format(loc[1],2**(loc[0]+1),loc[0])

        return super().__getitem__(loc)


'''


