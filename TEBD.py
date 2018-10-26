# Declan Maguire - September 2018
# TEBD
import numpy as np
import scipy
import scipy.sparse as sparse
import scipy.sparse.linalg
import scipy.linalg
import math

class MPS:
	'''
	A finite matrix product state (MPS) with open bounds, in the Vidal format.

	--<Lj-1>--a--[Gj]--b--<Lj>--c--[Gj+1]--d--<Lj+1>--e--
	               |                  |                  
	               s                  t                  
	               |                  |                  
	
	An MPS in this format is represented as a chain of gamma tensors of rank
	3, and lambda tensors of rank 2 (represented by G and L in the comment
	diagrams here and below). At each lambda, the tensor network can be seen
	as the Schmidt Decompositon of the whole network into the part to the left
	of the lambda and the part to the right. As NumPy does not have a specific
	function to perform Schmidt Decomposition, this is performed by doing SVD
	decomposition on tensors reshaped into 2D arrays, and then reshaping them
	back into higher dimensional arrays.

	Gammas are implemented as 3D complex NumPy arrays, and lambdas as sparse
	NumPy matrices (distinct but related to matrices) in diagonal format with
	non-complex float entries.

	The corespondence between array dimensions and bonds for the gamma tensors
	is physical bond:left virtual bond:right virtual bond. For example, tensor
	Gj above has its dimensions ordered as Gj[s,a,b]. Lambdas are the same
	without the real bond, so for example Lj has Lj[b,c].

	Attributes:
		N (int): The number of particles.
		d (int): Dimension of hilbert space of individual particles.
		chi_max (int): Maximum virtual bond dimensions.
		gammas (list of complex 3D arrays): The gamma tensors as seen above,
			with the list length being N.
		lambdas (list of float diagonal sparse matrices): The lambda tensors as
			seen above, with the list length being N-1.
		max_gamma_dims (list of tuples of ints): List of length N, where each
			element is a tuple of 3 ints, giving the sizes of each dimension of
			the corresponding gamma tensor.

	Methods:
		__init__: See below. Gamma tensors of correct dimensions are randomly
			populated, and lambdas are all set to be identity matrices, then
			tensor network is orthonormalised.
		__len__: Returns N.
		__abs__: Returns magnitude of the wavefunction the MPS represents.
		__imul__(numeric): Scales the wavefunction, returns self.
		inner(MPS): Returns the inner product of two MPSs. In bra-ket notation,
			self.inner(bra) = <bra|self>.
		orthogonalise: Resets all tensors to be in Vidal form without
			normalising. Returns None.
		orthonormalise: As orthogonalise, but also normalises.
		EvolveBond(j, tensor=None, normalise=False): Acts on gamma tensor j
			and, if it exists, gamma tensor j+1, and all lambdas adjacent. All
			are contracted with each other, 'tensor' (if not None) is
			contracted with it, and then what results is decomposed back into
			the correct format. If normalise=True, then the lambda between the
			two gammas is also normalised. Returns None.
	'''

	def __init__(self, N, d, chi_max):
		'''
		Generates a randomly populated MPS in Vidal form.

		Gamma tensors are randomly populated, and lambdas set to identity, and
		then whole network is properly orthonormalised into Vidal form.

		Args:
		N (int): Number of particles simulated. Must be an integer or float
			equal to an integer, and be greater than 0.
		d (int): The dimensionality of the local Hilbert space of each
			particle. Must be an integer greater than 0.
		chi_max (int): The maximum number of components to be kept after
			Schmidt decomposition before truncation of remaining components.
			Must be an integer greater than 0.
		'''
		# Type checking, initialising N
		if not (type(N) == int): raise TypeError ('"N" must be type int')
		elif not N >= 1: 		 raise ValueError('"N" must be >= 1')
		self.N = N
		# Type checking, initialising d
		if not (type(d) == int): raise TypeError ('"d" must be type int')
		elif not d >= 1:		 raise ValueError('"d" must be >= 1')
		self.d = d
		# Type checking, initialising chi_max
		if not (type(chi_max) == int): raise TypeError ('"chi_max" must be an int')
		elif not chi_max >= 1: 		   raise ValueError('"chi_max" must be >= 1')
		self.chi_max = chi_max

		# Generating list of the maximal dimensions of the gamma tensors.
		# The bond dimensions are the minimum needed at each site for a complete
		# respresentation of an arbitrary state, except where this exceeds
		# chi_max in which case the dimension is limited to chi_max.
		self.max_gamma_dims = [(self.d,
							   min(self.chi_max,
							   	   self.d**j,
						  	       self.d**(self.N - j)
								   ),
							   min(self.chi_max,
							       self.d**(j+1),
							       self.d**(self.N - (j+1))
						    	   )
							    )
							   for j in range(self.N)
							   ]

		# Generating list of gamma tensors by generating randomly populated
		# arrays of correct dimension, and then complexifying.
		self.gammas = [np.random.rand(*self.max_gamma_dims[j]).astype('complex')
					   for j in range(self.N)
					   ]

		# Generating lambdas as list of diagonal sparse identity matrices of
		# correct dimension (numpy does not yet have sparse arrays).
		self.lambdas = [sparse.identity(self.max_gamma_dims[j][2])
						for j in range(self.N - 1)
						]

		# This normalises the state with appropriate unitarity of gamma tensors.
		self.orthonormalise()

	def __len__(self): # The length is the number of sites in the system
		return self.N

	def __abs__(self): # The norm of the wavefunction, cleaning up complex part
		return abs(self._norm())

	def __imul__(self, scalar): # For easily scaling the wavefunction
		self.gammas[-1] *= scalar
		return self

	def inner(self, bra):
		'''
		Calculates the inner product of bra and other, e.g. <bra|self>

		Here the ket is 'self'. Works by creating an "accumulator" tensor A
		by first contracting the first gammas of self and bra together along
		their joining physical bond 's' and their leftmost dangling virtual
		bonds (conceptually, contracting them with an imaginary identity
		tensor). A is then recursively contracted with the adjoining lambdas,
		then with their adjoining gamma tensors from self, then the adjoining
		gamma tensor from bra along their two shared indices. At the end of the
		two MPSs, the final open bonds are contracted with each other like the
		leftmost open bonds were contracted initially.

		|  +--[G0]--b-<L0>--d-[G1]--f--<L1>-  |  b--<L0>-d--[G1]-f-<L1>--  |
		|  |   |               |              |  |           | 	           |
		|  a   s               t              | [A]  	     t 	           |
		|  |   |               |              |  |           |             |
		|  +--[G'0]-c-<L'0>-e-[G'1]-g-<L'1>-  |  c-<L'0>-e-[G'1]-g-<L'1>-  |

		Args:
			bra (MPS): An MPS of the same length and local dimension as self

		'''
		# Type checking if self and bra are compatible
		if not ( (len(self) == len(bra)) or (self.d == bra.d) ):
			raise ValueError('Particle number and local dimensions must match')
		# Creating A from contraction of first gamma matrices of self and bra
		A = np.einsum('sab,sac -> bc',
					   self.gammas[0], # [G0]: sab
					   np.conj(bra.gammas[0]), # [G'0]: sad
					   optimize = True)
		# Iteratively contracting A with sites to its right
		for j in range(1, self.N):
			A = np.einsum('bc,bd,ce -> de',
						  A, # [A]: bc
						  self.lambdas[j-1].toarray(), # <L0>: bd
						  bra.lambdas[j-1].toarray(), # <L'0>: ce
						  optimize = True)
			A = np.einsum('de,tdf -> tef',
						  A, # [A]: de
						  self.gammas[j], # [G1]: tdf
						  optimize = True)
			A = np.einsum('tef,teg - > fg',
						  A, # [A]: tef
						  np.conj(bra.gammas[j]), # [G'1]: teg
						  optimize = True)
		# Contracting final dangling open bonds of A with each other
		inner_product = np.einsum('ff', A)
		# Voila
		return inner_product

	def _norm(self): # State norm, without complex component cleaned up
		return (self.inner(self))**(1/2)

	def orthogonalise(self):
		for j in range(self.N - 1):
			self.EvolveBond(j, tensor = None, normalise = False)

	def orthonormalise(self):
		for j in range(self.N - 1):
			self.EvolveBond(j, tensor = None, normalise = True)
		self.gammas[-1] *= (1/self._norm())

	def EvolveBond(self, j, tensor = None, normalise = False):

		#  -a-<Lj-1>-b-[Gj]-c-<Lj>-d-[Gj+1]-e-<Lj+1>-f- | -a-[Mj,j+1]-f-
		#               |              |                |      |  |
		#     (Step 1)  s              t                |      s  t
		#               |______________|                |     _|__|_
		#               [____tensor____]                |    [tensor]
		#               |              |                |      |  |
		#               u              v                |      u  v (Step 2)
		#_______________|______________|________________|______|__|__________
		#                   |
		# -a-[M'j,j+1]-f-   | -a-<Lj-1>-b-[G'j]-c-<L'j>-d-[G'j+1]-e-<Lj+1>-f-
		#      |   |        |               |                |
		#      u   v  (Step |   (Step 4)    u                v
		#      |   |    3)  |               |                |

		# Type & index checking j
		if not (type(j) == int): raise TypeError('Index must be integer')
		elif not (0 <= j < (self.N-1)): raise IndexError('Index "j" out of bounds')
		# Type and dimension checking tensor
		if not (type(tensor) == type(None)):
			if not (tensor.shape == 4*(self.d,)):
				raise TypeError('"tensor" argument must be a d*d*d*d array')

		# Checking if gamma tensors are at boundaries, contracting bounding
		# lambda tensors if they exist into new arrays "A" and "B"
		if (j == 0):
			LeftBounded = True
			A = self.gammas[0]
		else:
			LeftBounded = False
			A = np.einsum('ab,sbc -> sac', self.lambdas[j-1].toarray(), self.gammas[j], optimize = True)
		if ((j+1) == (self.N - 1)):
			RightBounded = True
			B = self.gammas[j+1]
		else:
			RightBounded = False
			B = np.einsum('tde,ef -> tdf', self.gammas[j+1], self.lambdas[j+1].toarray(), optimize = True)

		# Contracting remaining tensors into final rank-4 tensor "M"
		M = np.einsum('sac,cd -> sad', A, self.lambdas[j].toarray(), optimize = True)
		M = np.einsum('sad,tdf -> satf', M, B, optimize = True)
		if not (type(tensor) == type(None)):
			M = np.einsum('satf,sutv -> uavf', M, tensor, optimize = True) # M -> M'
		M_dims = M.shape

		# Reshaping new contracted tensor into matrix, performing SVD decomposition on it
		M = np.reshape(M, (M_dims[0]*M_dims[1], M_dims[2]*M_dims[3]) )
		U,D,Vt = np.linalg.svd(M, full_matrices = False)

		# Recording old norm of D for later rescaling
		D_old_norm = np.linalg.norm(D)
		InnerDim = min(self.max_gamma_dims[j][2], np.argmin(D))
		D = D[:InnerDim]
		D *= (D_old_norm/np.linalg.norm(D))

		# Reshaping U, Vt back into rank-3 tensors, truncating.
		self.gammas[j] =   np.reshape(U,  (M_dims[0], M_dims[1], -1) )
		self.gammas[j+1] = np.reshape(Vt, (M_dims[2], -1, M_dims[3]) )
		self.gammas[j] =   self.gammas[j][:, :, :InnerDim]
		self.gammas[j+1] = self.gammas[j+1][:, :InnerDim, :]

		# Checking if gamma tensors lie at boundaries, if not, contracting
		# inverses of outer lambdas into gammas (effectively factoring out
		# the original lambdas we contracted in).
		if not LeftBounded:
			LeftInvLambda = np.diag(np.reciprocal(np.diag(self.lambdas[j-1].toarray())))
			self.gammas[j] = np.einsum('ubc,ab -> uac',
									   self.gammas[j],
									   LeftInvLambda,
									   optimize = True
									   )
		if not RightBounded:
			RightInvLambda = np.diag(np.reciprocal(np.diag(self.lambdas[j+1].toarray())))
			self.gammas[j+1] = np.einsum('vde,ef -> vdf',
									     self.gammas[j+1],
									     RightInvLambda,
									     optimize = True
									     )

		# Normalising singular values if applicable
		if (normalise == True): D /= np.linalg.norm(D)

		# Replacing middle lambda
		self.lambdas[j] = sparse.diags(D)

class LocalMPO:
	'''
	A Matrix Product Operator (MPO) consisting of local bond operators.

	--<Lj-1>--a--[Gj]--b--<Lj>--c--[Gj+1]--d--<Lj+1>--e--
	               |                  |                  
	               s                  t                  
	               |__________________|                  
	               [____Operator[j]___]                  
	               |                  |                  
	               u                  v                  
	               |                  |                  

	At its core, a list of 4D numeric arrays which act pairs of adjacent gamma
	tensors in an MPS, as well as a single 2D array to represent an operator on
	the final gamma tensor in an MPS. If these arrays are all identical or
	repeat after some point, then only as many as is needed are stored.

	Each site operator S is has its dimensions ordered in its array as
	S[s,u,t,v] (see diagram).

	Attributes:
		N (int): Number of particles MPO acts on.
		d (int): Local dimension of the particles the MPO acts on.
		rep_len (int): How often the local operators repeat. Defaults to 1.
		Operators (list of 4D NumPy arrays): Operators on pairs of gammas j,
		j+1 in an MPS. List has N-1 elements.
		FinalSite (2D NumPy array): Operator that acts on the final gamma in an
		MPS.

	Methods:
		__init__(N, d, SiteOperators=None, BondOperators=None, rep_len=1):
			See below.
		__len__: Returns N.
		__imul__(numeric): Scales MPO. Returns self.
		apply(MPS): Applies MPO to MPS, transforming it. Returns None.
		sandwich(bra (MPS), ket(MPS): Returns <bra|self|ket>.
		ExpectationValue(state (MPS)): Returns <state|self|state>
	'''

	def __init__(self, N, d, SiteOperators = None, BondOperators = None, rep_len = 1):
		# A pointless amount of type/value checking
		if not (type(N) == int): raise TypeError('"N" must be of type int')
		elif (N < 1): raise ValueError('"N" must be greater than 0')
		self.N = N

		if not (type(d) == int): raise TypeError('"d" must be of type int')
		elif (d < 1): raise ValueError('"d" must be greater than 0')
		self.d = d

		if not (type(rep_len) == int): raise TypeError('"rep_len" must be of type int')
		elif (rep_len < 1): raise ValueError('"rep_len" must be greater than 0')
		self.rep_len = rep_len

		# Site operators type & value checks
		if not (type(SiteOperators) == list):
			if not (SiteOperators == None):
				raise TypeError('"SiteOperators" must be a list of arrays')
			else:
				SiteOperators = min(N, rep_len)*[np.zeros((d,d))]
				self.FinalSite = np.identity(d)
		else:
			self.FinalSite = SiteOperators[(self.N - 1) % self.rep_len]

		for operator in SiteOperators:
			if not operator.shape == 2*(self.d,):
				raise TypeError('All arrays in "SiteOperators" must have dimensions d*d')
		if len(SiteOperators) < min(self.rep_len, self.N):
			raise ValueError('"SiteOperators" must not have length less than rep_len and N')
		# Bond operators type & value checks
		if not (type(BondOperators) == list):
			if not (BondOperators == None):
				raise TypeError('"BondOperators" must be a list of arrays')
			else:
				BondOperators = min(N, rep_len)*[np.zeros((d,d,d,d))]
		for operator in BondOperators:
			if not operator.shape == 4*(self.d,):
				raise TypeError('All arrays in "BondOperators" must have dimensions d*d*d*d')
		if (len(BondOperators) < min(self.rep_len, self.N - 1)):
			raise ValueError('"SiteOperators" must not have length less than rep_len and N-1')

		self.Operators = [(BondOperators[j] + 
						   np.einsum('ab,cd -> abcd',
						   			 SiteOperators[j],
						   			 np.identity(self.d)
						   			 )
						   )
						  for j in range(min(self.N-1, self.rep_len))
						  ]

		if (self.N % 2 == 0):
			self.Even = True
		else:
			self.Even = False

	def __len__(self):
		return self.N

	def __imul__(self, scalar):
		self.FinalSite *= scalar
		return self

	def __getitem__(self, j):
		if not (type(j) == int): raise IndexError('Index must be an int value')
		if not (0 <= j <= self.N): raise IndexError('Index out of bounds')
		if (j < self.N):
			return self.Operators[j % self.rep_len]
		else:
			return self.FinalSite

	def __setitem__(self, j, NewOp):
		if not (type(j) == int): raise IndexError('Index must be an int value')
		if not (0 <=j <= self.N): raise IndexError('Index out of bounds')
		if (j == self.N):
			if not (NewOp.shape == 2*(self.d,)):
				raise ValueError('Final operator must be d*d array')
			else:
				self.FinalSite = NewOp
		else:
			if not (NewOp.shape == 4*(self.d,)):
				raise ValueError('Non-final operators must be d*d*d*d arrays')
			else:
				self.Operators[j % self.rep_len] = NewOp

	def apply(self, state):
		if not (len(self) == len(state)):
			raise TypeError('Operator and state must be of same length')
		elif not (self.d == state.d):
			raise TypeError('Operator and state must have same local dimension d')

		if not self.Even:
			state.gammas[-1] = np.einsum('sab,st -> tab',
										 state.gammas[-1],
										 self.FinalSite,
										 optimize = True
										 )
		for j in range(self.N - 1):
			if j%2 == 0:
				state.EvolveBond(j, tensor = self.Operators[j%self.rep_len])
		for j in range(self.N - 1):
			if j%2 == 1:
				state.EvolveBond(j, tensor = self.Operators[j % self.rep_len])
		if self.Even:
			state.gammas[-1] = np.einsum('sab,st -> tab',
										 state.gammas[-1],
										 self.FinalSite,
										 optimize = True
										 )

	def sandwich(self, bra, ket):
		if not (len(self) == len(bra) == len(ket)):
			raise TypeError('MPO, bra, and ket must all have same length')
		elif not (self.d == bra.d == ket.d):
			raise TypeError('MPO, bra, and ket must all have same local dimension d')
		A = ket.gammas[0]
		B = np.conj(bra.gammas[0])
		for j in range(self.N - 1):
			A = np.einsum('sab,bc -> sac', A, bra.lambdas[j].toarray(), optimize = True)
			B = np.einsum('uae,ef -> uaf', B, ket.lambdas[j].toarray(), optimize = True)
			print(j)
			if (j%2 == 0):
				A = np.einsum('sac,tcd -> stad', A, bra.gammas[j+1], optimize = True)
				A = np.einsum('stad,stuv -> uvad', A, self.Operators[j%self.rep_len], optimize = True)
				A = np.einsum('uvad,uaf -> vfd', A, B, optimize = True)
				B = np.conj(bra.gammas[j+1])
			else:
				B = np.einsum('uaf,vfg -> uvag', B, np.conj(bra.gammas[j+1]), optimize = True)
				B = np.einsum('uvag,stuv -> stag', B, self.Operators[j%self.rep_len], optimize = True)
				B = np.einsum('stag,sac -> tcg', B, A, optimize = True)
				A = ket.gammas[j+1]
		A = np.einsum('sab,st -> tab', A, self.FinalSite, optimize = True)
		return np.einsum('tab,tab', A, B, optimize = True)

	def ExpectationValue(self, state):
		if not (len(self) == len(state)):
			raise TypeError('MPO and state must have same length')
		elif not (self.d == state.d):
			raise TypeError('MPO and state must have same local dimension d')
		return self.sandwich(state, state)

class ExpMPO(LocalMPO):
	def __init__(self, myMPO, exponent):
		ops = myMPO.Operators
		LocalMPO.__init__(self, myMPO.N, myMPO.d, SiteOperators = None, BondOperators = ops, rep_len = myMPO.rep_len)
		self.FinalSite = myMPO.FinalSite
		for j in range(min(self.rep_len, self.N-1)):
			temp = self[j]
			tempshape = temp.shape
			temp = np.reshape(temp, (tempshape[0]*tempshape[1], tempshape[2]*tempshape[3]))
			if j%2 == 0:
				temp = scipy.linalg.expm((exponent/2)*temp)
			else:
				temp = scipy.linalg.expm(exponent*temp)
			temp = np.reshape(temp, tempshape)
			self[j] = temp
		if self.Even:
			self.FinalSite = scipy.linalg.expm((exponent/2)*self.FinalSite)
		else:
			self.FinalSite = scipy.linalg.expm(exponent*self.FinalSite)

	def apply(self, state):
		if not (self.N==state.N): raise TypeError('N values do not match')
		elif not (self.d==state.d): raise TypeError('d values do not match')
		for j in range(self.N-1):
			if j%2==0:
				state.EvolveBond(j, tensor = self[j], normalise = False)
		if self.Even:
			state.gammas[-1] = np.einsum('sab,st -> tab',
										 state.gammas[-1],
										 self.FinalSite,
										 optimize = True
										 )
		for j in range(self.N-1):
			if j%2==1:
				state.EvolveBond(j, tensor = self[j], normalise = False)
		if not self.Even:
			state.gammas[-1] = np.einsum('sab,st -> tab',
										 state.gammas[-1],
										 self.FinalSite,
										 optimize = True
										 )
		for j in range(self.N-1):
			if j%2==0:
				state.EvolveBond(j, tensor = self[j], normalise = False)
		if self.Even:
			state.gammas[-1] = np.einsum('sab,st -> tab',
										 state.gammas[-1],
										 self.FinalSite,
										 optimize = True
										 )


N_test = 50
d_test = 2
chi_test = 16

print('Making MPS')
MPS_test = MPS(N_test, d_test, chi_test)
print('MPS made')
print('Orthonormalising')
MPS_test.orthonormalise()
print('Orthonormalisations finished')
print('State norm before tests = ', abs(MPS_test))

print('Making operators and MPO')
Ident_Site_Op = np.zeros((d_test, d_test))
Ident_Bond_Op = np.einsum('ab,cd -> abcd', Ident_Site_Op, Ident_Site_Op)
MPO_test = LocalMPO(N_test, d_test, SiteOperators = None, BondOperators = [Ident_Bond_Op])
print('Operators and MPO made')

print('Attempting to exponentiate MPO')
newMPO = ExpMPO(MPO_test, -1*(-1)**0.5)

print('Old state norm  = ', abs(MPS_test))
print('Acting expmpo on state')
newMPO.apply(MPS_test)
print('New state norm  = ', abs(MPS_test))