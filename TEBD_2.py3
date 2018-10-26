# Declan Maguire - October 2018
# TEBD

# THIS CODE IS INCOMPLETE

import numpy as np
import scipy
import scipy.sparse as sparse
import scipy.sparse.linalg
import scipy.linalg
import math
from cmath import log

class MPS:
	'''
	A finite matrix product state (MPS) with open bounds, in Vidal form.

	--<Lj-1>--a--[Gj]--b--<Lj>--c--[Gj+1]--d--<Lj+1>--e--
	               |                  |                  
	               s                  t                  
	               |                  |                  
	
	An MPS represented by a chain of rank 3 tensors (called gamma tensors here,
	represented by G in diagrams) and rank 2 tensors (called lambda tensors,
	represented by L in diagrams). 

	Gammas are implemented as 3D complex NumPy arrays, and lambdas as SciPy
	diagonal sparse matrices

	The bonds of the gamma tensors are ordered in their arrays as left virtual
	bond, physical bond, right virtual bond (e.g. for Gj above, it would be
	Gj[a,s,b]). Lambda tensors have theirs ordered left virtual, right virtual
	(Lj[b,c]).

	Attributes:
		n (int): The number of particles.
		d (int): Dimension of hilbert space of individual particles.
		chi (int): Maximum virtual bond dimensions.
		gammas (list of complex 3D arrays): The gamma tensors as seen above,
			with the list length being n.
		lambdas (list of float diagonal sparse matrices): The lambda tensors as
			seen above, with the list length being n-1.
		gamma_dims_max (list of tuples of ints): List of length N, where each
			element is a tuple of 3 ints, giving the maximum allowed sizes of
			each dimension of the corresponding gamma tensor.

	Methods:
		__init__: See below. Gamma tensors of correct dimensions are randomly
			populated, and lambdas are all set to be identity matrices, then
			tensor network is orthonormalised.
		__len__: Returns n.
		__abs__: Returns magnitude of the wavefunction the MPS represents.
		__imul__(numeric): Scales the wavefunction, returns self.
		__idiv__(scalar): Scales wavefunction by 1/scalar, returns self.
		inner(MPS): Returns the inner product of two MPSs. In bra-ket notation,
			self.inner(bra) = <bra|self>.
	'''
	def __init__(self, n, d, chi):
		if not type(n)==int:   raise TypeError('"n" must have type int')
		if not type(d)==int:   raise TypeError('"d" must have type int')
		if not type(chi)==int: raise TypeError('"chi" must have type int')
		if not n>0:   raise ValueError('"n" must be greater than 0')
		if not d>0:   raise ValueError('"d" must be greater than 0')
		if not chi>0: raise ValueError('"chi" must be greater than 0')

		self.n = n
		self.d = d
		self.chi = chi

		self.gammas = [np.random.rand(*self.gamma_dims_max(j)).astype('complex')
					   for j in range(n)
					   ]

		self.lambdas = [sparse.identity(self.gamma_dims_max(j)[2])
						for j in range(n -1)
						]

		self.orthonormalise()
		self.orthonormalise()

	def __len__(self):
		return self.n

	def __imul__(self, scalar): # Implements scaling with "myMPS *= 5.2" syntax
		if self.n == 1:
			self.gammas[-1] *= scalar
			return self
		else:
			self.lambdas[-1] *= abs(scalar)
			if not scalar == 0:
				self.gammas[-1] *= (scalar/abs(scalar))
			return self

	def __itruediv__(self, scalar):
		if scalar == 0: raise ValueError('Cannot divide by 0')
		self *= (1/scalar)
		return self

	def __abs__(self):
		return abs(self._norm())

	def _norm(self): # Gives wavefunction norm without making real
		return self.inner(self)**(1/2)

	def gamma_dims_max(self, j):
		return (
				min(self.chi, self.d**j, self.d**(self.n-j)),
				self.d,
				min(self.chi, self.d**(j+1), self.d**(self.n-j-1))
				)

	def get_pair_tensor(self, j):
		if not type(j)==int:    raise TypeError('"j" must be an integer')
		if not 0<=j<(self.n-1): raise IndexError('"j" out of bounds')

		if j==0: left_bound_lambda = np.array([[1]])
		else: left_bound_lambda = self.lambdas[j-1].toarray()
		if (j+1)==(self.n-1): right_bound_lambda = np.array([[1]])
		else: right_bound_lambda = self.lambdas[j+1].toarray()

		pair_tensor = np.einsum('ab,bsc,cd,dte,ef -> aste',
								left_bound_lambda,
								self.gammas[j],
								self.lambdas[j].toarray(),
								self.gammas[j+1],
								right_bound_lambda,
								optimize = True)
		return pair_tensor

	def _set_pair_from_tensor(self, j, tensor):
		'''
		Creates self.gammas[j], self.gammas[j+1], self.lambdas[j] from tensor.

		Treats 'tensor' as though it were the contraction of self.lambdas[j-1],
		self.gammas[j], self.lambdas[j], self.gammas[j+1], self.lambdas[j+1],
		and regenerates them.
		'''

		# Index typechecking
		if not type(j)==int:    raise TypeError('"j" must be an integer')
		if not 0<=j<(self.n-1): raise IndexError('"j" out of bounds')
		# Tensor typechecking/dimension recording
		pair_dims = (self.gammas[j].shape[0],
					 self.d,
					 self.d,
					 self.gammas[j+1].shape[2])
		if not tensor.shape == pair_dims:
			raise ValueError('"tensor" has wrong dimensions')

		# This is terrible so I made a function for it. Inverses of diagonal
		# matrices are just what you get when you reciprocate the diagonals
		def _inv(sparse_dia_matrix):
			return np.diag(np.reciprocal(np.diag(sparse_dia_matrix.toarray())))

		# Creating left and right inverses of left and right lambda tensors
		# If tensors = 0, they are left as-is.
		# If left and right tensors, don't exist, identity 1x1 tensors are used
		# Left lambdas
		if j==0:
			left_inv_lambda = np.array([[1]])
		elif sparse.linalg.norm(self.lambdas[j-1]) == 0:
			left_inv_lambda = self.lambdas[j-1].toarray()
		else:
			left_inv_lambda = _inv(self.lambdas[j-1])
		# Right lambdas
		if (j+1)==(self.n - 1):
			right_inv_lambda = np.array([[1]])
		elif sparse.linalg.norm(self.lambdas[j+1]) == 0:
			right_inv_lambda = self.lambdas[j+1].toarray()
		else:
			right_inv_lambda = _inv(self.lambdas[j+1])

		# Reshaping tensor into matrix so we can SVD it, then SVDing it
		M = np.reshape(tensor, (pair_dims[0] * pair_dims[1],
								pair_dims[2] * pair_dims[3]))
		U,D,Vt = np.linalg.svd(M, full_matrices = False)

		# Truncating U,d,Vt, and rescaling D to preserve norm
		old_D_norm = np.linalg.norm(D)
		# 'argmin' here to help truncate any extra 0s.
		inner_dim = min(self.gamma_dims_max(j)[2], np.argmin(D))
		D = D[:(inner_dim)]
		if not old_D_norm==0:
			D *= (old_D_norm/np.linalg.norm(D))
		U = U[:,:(inner_dim)]
		Vt = Vt[:(inner_dim),:]

		# Reshaping matrices back into tensors, remaking gammas/lambdas
		self.gammas[j] = np.reshape(U, (pair_dims[0],
										self.d,
										inner_dim) )
		self.gammas[j+1] = np.reshape(Vt, (inner_dim,
										  self.d,
										  pair_dims[3]) )
		self.gammas[j] = np.einsum('ab,bsd -> asd',
								   left_inv_lambda,
								   self.gammas[j],
								   optimize = True)
		self.gammas[j+1] = np.einsum('dte,ef -> dtf',
									 self.gammas[j+1],
									 right_inv_lambda,
									 optimize =  True)
		self.lambdas[j] = sparse.diags(D)

	def _orthogonalise_pair(self, j):
		self._set_pair_from_tensor(j, self.get_pair_tensor(j))

	def _orthonormalise_pair(self, j):
		self._orthogonalise_pair(j)
		lambda_norm = sparse.linalg.norm(self.lambdas[j])
		if lambda_norm == 0:
			raise ValueError('Pair unnormalisable: norm=0 at j={}'.format(j))
		else:
			self.lambdas[j] /= lambda_norm

	def orthogonalise(self):
		for j in range(self.n - 1):
			self._orthogonalise_pair(j)

	def orthonormalise(self):
		for j in range(self.n - 1):
			self._orthonormalise_pair(j)
		self /= self._norm()

	def update_pair(self, j, operator):
		#  -a-<Lj-1>-b-[Gj]-c-<Lj>-d-[Gj+1]-e-<Lj+1>-f- | -a-[Mj,j+1]-f-
		#               |              |                |      |  |
		#     (Step 1)  s              t                |      s  t
		#               |______________|                |    __|__|__
		#               [___operator___]                |   [operator]
		#               |              |                |      |  |
		#               u              v                |      u  v (Step 2)
		#_______________|______________|________________|______|__|__________
		#                   |
		# -a-[M'j,j+1]-f-   | -a-<Lj-1>-b-[G'j]-c-<L'j>-d-[G'j+1]-e-<Lj+1>-f-
		#      |   |        |               |                |
		#      u   v  (Step |   (Step 4)    u                v
		#      |   |    3)  |               |                |

		# Operator typechecking
		if not operator.shape == 4*(self.d,):
			raise ValueError('"operator" has wrong dimensions')
		M = self.get_pair_tensor(j)
		M_new = np.einsum('astb,stuv -> auvb',
						  M, # astb
						  operator, # stuv
						  optimize = True)
		self._set_pair_from_tensor(j, M_new)

	def update_site(self, j, operator, strict_update = True):
		# Index typechecking
		if not 0<=j<(self.n): raise IndexError('"j" out of bounds')
		# Operator typechecking
		if not operator.shape == 2*(self.d,):
			raise ValueError('"operator" has wrong dimensions')
		# Acting operator on gamma
		self.gammas[j] = np.einsum('asb,su -> aub',
								   self.gammas[j], # asb
								   operator, # su
								   optimize = True)
		# Re-orthogonalising pairs on each side
		if strict_update:
			if not j==0:
				self._orthogonalise_pair(j-1)
			if not j==(self.n - 1):
				self._orthogonalise_pair(j)

	def inner(self, bra):
		'''
		Calculates the inner product of bra and other, e.g. <bra|self>

		Contracts the first gammas of self and bra into a new tensor 'A', then
		iteratively contracts the adjoining lambdas and gammas with themselves
		and into A until the end, when the final dangling bonds are contracted.

		|  +--[G0]--b-<L0>--d-[G1]--f--<L1>-  |  b--<Lj-1>-d--[Gj]-f-<Lj>--  |
		|  |   |               |              |  |             | 	         |
		|  a   s               t              | [A]  	       t 	         |
		|  |   |               |              |  |             |             |
		|  +--[G'0]-c-<L'0>-e-[G'1]-g-<L'1>-  |  c-<L'j-1>-e-[G'j]-g-<L'j>-  |

		Args:
			bra (MPS): An MPS of the same length and local dimension as self.

		'''
		# Type checking if self and bra are compatible
		if not isinstance(bra, MPS): raise ValueError('"bra" must be an MPS')
		if not (len(self)==len(bra) and self.d==bra.d):
			raise ValueError('Particle number and local dimensions must match')

		# Creating A from contraction of first gamma matrices of self and bra
		A = np.einsum('asb,asc -> bc',
					   self.gammas[0], # [G0]: asb
					   np.conj(bra.gammas[0]), # [G'0]: asd
					   optimize = True)

		# Iteratively contracting A with sites to its right
		for j in np.arange(1, self.n):
			A = np.einsum('dtf,bd,bc,ce,etg -> fg',
						  self.gammas[j], # [Gj]: dtf
						  self.lambdas[j-1].toarray(), # <L0>: bd
						  A, # [A]: bc
						  bra.lambdas[j-1].toarray(), # <L'0>: ce
						  np.conj(bra.gammas[j]), # [G'j]: etg
						  optimize = True)

		# Contracting final dangling open bonds of A with each other, returning
		return np.einsum('ff', A)

class Local_MPO:
	def __init__(self, raw_operator_list, n, d):
		if not (type(n) == int): raise TypeError('"n" must be of type int')
		elif (n < 1): raise ValueError('"n" must be greater than 0')
		if not (type(d) == int): raise TypeError('"d" must be of type int')
		elif (d < 1): raise ValueError('"d" must be greater than 0')
		self.n = n
		self.d = d
		self._rep_len = min(len(raw_operator_list), n - 1)
		self._operator_list = [None for _ in range(self._rep_len)]
		self._operator_is_big = np.array([False for _ in range(self._rep_len)])
		self._final_operator = None
		for j in np.arange(min(len(raw_operator_list), n)):
			self[j] = raw_operator_list[j]

	def __len__(self):
		return self.n

	def __getitem__(self, j):
		if j == (self.n - 1) or j == -1: return self._final_operator
		elif 0<=j<(self.n - 1): return self._operator_list[j % self._rep_len]
		else: raise IndexError('Index out of bounds')

	def __setitem__(self, j, raw_op): # This was a motherfucker to implement
		# If raw_op is not None, it is assumed its first entry is a single
		# site operator and the second a dual site operator.
		#
		# This whole terrible nest of logic makes it so that if raw op is
		# None or (None, None), or [None, None] etc, the operator j will
		# be set to the identity, if either are not None but the other is
		# the operator will be set to that, and if both aren't None then
		# the first will be turned into a pair operator and the two added.

		if not 0<=j<self.n: raise IndexError('Index out of bounds')

		if raw_op is None: new_op = np.identity(self.d)
		elif raw_op[0] is None: new_op = np.identity(self.d)
		elif not raw_op[0].shape==2*(self.d,):
			raise ValueError('raw_op[0] has wrong dimensions')
		else: new_op = raw_op[0]

		self._operator_is_big[j % self._rep_len] = False

		if (j % self._rep_len)==0: self._final_operator = new_op

		if not raw_op==None:
			if not raw_op[1] is None:
				if not raw_op[1].shape == 4*(self.d,):
					raise TypeError('raw_op[1] has wrong dimensions')
				elif raw_op[0] is not None:
					new_op = raw_op[1] + self._dualise_site_operator(new_op)
				else:
					new_op = raw_op[1]
				self._operator_is_big[j % self._rep_len] = True

		self._operator_list[j % self._rep_len] = new_op

	def _dualise_site_operator(self, op):
		return self.join_site_operators(op, np.identity(self.d))

	def is_big(self, j):
		if not 0<=j<self.n: raise IndexError('Index out of bounds')

		if j == (self.n - 1): return False
		else: return self._operator_is_big[j % self._rep_len]


	def join_site_operators(self, op_1, op_2):
		if not op_1.shape==2*(self.d,) and op_2.shape==2*(self.d,):
			raise ValueError('Operators have incorrect dimensions')
		return np.einsum('su,tv -> stuv', op_1, op_2, optimize = True)

	def evolve(self, j, state):
		if not self.d == state.d:
			raise ValueError('Local dimensions must match')
		if self.is_big(j): state.update_pair(j, operator = self[j])
		elif not self.is_big(j): state.update_site(j, self[j], strict_update = False)
		else: raise ValueError('Operator not initialised')

	def act_on(self, state):
		raise NotImplementedError('Must define subclass to implement method')

	def sandwich(self, ket, bra):
		raise NotImplementedError('Must define subclass to implement method')

	def expectation_value(self, state):
		return self.sandwich(state, state)


class Regular_MPO(Local_MPO):

	def exponentiate(self, x):
		return Exp_MPO(self, x)

	def act_on(self, state):
		if not len(self)==len(state):
			raise ValueError('MPO and MPS lengths must match')
		for j in range(len(self)):
			if j%2 == 0:
				self.evolve(j, state)
		for j in range(len(self)):
			if j%2 == 1:
				self.evolve(j, state)

	def sandwich(self, bra, ket):
		#        ## FOR EVEN j ##        ..|        ## FOR ODD j ##         ..|
		#         b     c       d        ..|         b     c       d        ..|
		# +---{A}--<Lj>--[Gj+1]--<Lj+1>--..| +---{A}--<Lj>--[Gj+1]--<Lj+1>--..|
		# |   s|____________|t           ..| |    |            |____________..|
		# |    [__self[j]___]            ..| |   s|            [___self[j+1]..|
		# |a   |           v|____________..| |a   |____________|t           ..|
		# |    |u           [___self[j+1]..| |    [__self[j]___]            ..|
		# |    |  e     f   |   g        ..| |   u|  e     f   |v  g        ..|
		# +---{B}--<L'j>-[G'j+1]-<L'j+1>-..| +---{B}--<L'j>-[G'j+1]-<L'j+1>-..|

		# Quick type checks
		if not (len(self) == len(bra) == len(ket)):
			raise TypeError('MPO, bra, and ket must all have same length')
		elif not (self.d == bra.d == ket.d):
			raise TypeError('MPO, bra, ket, must share same local dimension d')

		A = ket.gammas[0]
		B = np.conj(bra.gammas[0])

		for j in np.arange(self.n - 1): # Keep contracting in until we are at end
			A = np.einsum('asb,bc -> asc',
						  A, # {A}: asb
						  ket.lambdas[j].toarray(), # <Lj>: bc
						  optimize = True)
			B = np.einsum('aue,ef -> auf',
						  B, # {B}: aue
						  bra.lambdas[j].toarray(), # <L'j>: ef
						  optimize = True)

			if (j%2 == 0): # For contracting even bonds
				A = np.einsum('asc,ctd -> astd',
							  A, # {A}: asc
							  ket.gammas[j+1], # [Gj+1]: ctd
							  optimize = True)
				if self.is_big(j): # Contract full two site operator in
					A = np.einsum('astd,stuv -> auvd',
								  A, # {A}: astd
								  self[j], # self[j]: stuv
								  optimize = True)
				else: # Contract small one site operator in
					A = np.einsum('astd,su -> autd',
								  A, # {A}: astd
								  self[j], # self[j]: su
								  optimize = True)
				A = np.einsum('auvd,auf -> fvd',
							  A, # {A}: auvd
							  B, # {B}: auf
							  optimize = True)
				B = np.conj(bra.gammas[j+1])

			else: # For contracting odd bonds
				B = np.einsum('auf,fvg -> auvg',
							  B, # {B}: auf
							  np.conj(bra.gammas[j+1]), # [G'j+1]: fvg
							  optimize = True)
				if self.is_big(j):
					B = np.einsum('auvg,stuv -> astg',
								  B, # {B}: auvg
								  self[j], # self[j]: stuv
								  optimize = True)
				else:
					B = np.einsum('auvg,su -> asvg',
								  B, # {B}: auvg
								  self[j], # self[j]: su
								  optimize = True)
				B = np.einsum('astg,asc -> ctg',
							  B, # {B}: astg
							  A, # {A}: asc
							  optimize = True)
				A = ket.gammas[j+1]
		# Contracting in final (small) operator, A, and B.
		return np.einsum('asb,su,aub',
					  	 A, # {A}: asb
					  	 self[-1], # self[-1]: su
					  	 B, # {B}: aub
					  	 optimize = True)

class Exp_MPO(Local_MPO):
	def __init__(self, old_MPO, x):
		self.n = old_MPO.n
		self.d = old_MPO.d
		self.x = x
		self._rep_len = old_MPO._rep_len
		self._operator_list = [self._exponential(x, j, old_MPO._operator_list[j])
							   for j in np.arange(len(old_MPO._operator_list))]
		self._operator_is_big = old_MPO._operator_is_big.copy()
		self._final_operator = self._exponential(x, self.n - 1, old_MPO._final_operator)

	def __imul__(self, scalar):
		if self.n%2 == 1:
			self._final_operator *= scalar**(1/2)
		else:
			self._final_operator *= scalar
		return self

	def __itruediv__(self, scalar):
		if scalar == 0:
			raise ValueError('Cannot divide by 0')
		self *= (1/scalar)
		return self

	def _exponential(self, x, j, operator):
		if j%2 == 0:
			s = 2
		else:
			s = 1
		if operator.shape == 2*(self.d,):
			return scipy.linalg.expm((x/s)*operator)
		elif operator.shape == 4*(self.d,):
			new_op = np.reshape(operator, 2*(self.d**2,))
			new_op = scipy.linalg.expm((x/s)*new_op)
			return np.reshape(new_op, 4*(self.d,))
		else:
			raise ValueError('Operator has wrong dimensions')

	def act_on(self, state):
		if not len(self)==len(state):
			raise ValueError('MPO and MPS lengths must match')
		for j in range(len(self)):
			if j%2 == 0: self.evolve(j, state)
		for j in range(len(self)):
			if j%2 == 1: self.evolve(j, state)
		for j in range(len(self)):
			if j%2 == 0: self.evolve(j, state)

	def sandwich(self, ket, bra):
		if not self.n==len(ket)==len(bra):
			raise ValueError('Lengths do not match')
		elif not self.d==ket.d==bra.d:
			raise ValueError('Local dimensions do not match')

		is_even = True
		A = np.einsum('su,tv->stuv', np.identity(1), np.identity(self.d))
		for j in range(self.n - 1):
			if is_even:
				B = np.einsum('asb,bc,ctd->astd',
							  ket.gammas[j],
							  ket.lambdas[j].toarray(),
							  ket.gammas[j+1],
							  optimize = True)
				C = np.einsum('axe,ef,fyg->axyg',
							  np.conj(bra.gammas[j]),
							  bra.lambdas[j].toarray(),
							  np.conj(bra.gammas[j+1]),
							  optimize = True)
				if self.is_big(j):
					B = np.einsum('astd,stuv->auvd',
								  B,
								  self[j],
								  optimize = True)
					C = np.einsum('axyg,uwxy->auwg',
								  C,
								  self[j],
								  optimize = True)
				else:
					B = np.einsum('astd,su->autd',
								  B,
								  self[j],
								  optimize = True)
					C = np.einsum('axyg,ux->auyg',
								  C,
								  self[j],
								  optimize = True)
				A = np.einsum('auvd,auhz,hzwg->dvgw',
							  B,
							  A,
							  C,
							  optimize = True)
				is_even = False
			else:
				if self.is_big(j):
					A = np.einsum('stuv,twvx,sa,ub->awbx',
								  A,
								  self[j],
								  ket.lambdas[j].toarray(),
								  bra.lambdas[j].toarray(),
								  optimize = True)
				else:
					A = np.einsum('stuv,tv,sa,ub,wx->awbx',
								  A,
								  self[j],
								  ket.lambdas[j].toarray(),
								  bra.lambdas[j].toarray(),
								  np.identity(self.d),
								  optimize = True)
				is_even = True

		if is_even:
			return np.einsum('stuv,at,sab,vc,ucb',
							 A,
							 self[-1],
							 ket.gammas[-1],
							 self[-1],
							 np.conj(bra.gammas[-1]))
		else:
			return np.einsum('stsu,tu', A, self[-1])

	def log_expectation_value(self, state):
		return (log(self.expectation_value(state))/self.x)

my_n = 20
my_d = 2
my_chi = 8
J = 1
sig = 0.5*np.array([[1,0],[0,-1]])
op_pair = [(-0*sig, -J*np.einsum('su,tv->stuv', sig, sig))]
my_MPS = MPS(my_n, my_d, my_chi)
H = Regular_MPO(op_pair, my_n, my_d)

for j in range(3,80):
	Ui = Exp_MPO(H, -1/j)
	print('j = ', j, 'and current energy expectation value per particle is ', Ui.log_expectation_value(my_MPS)/my_n)
	for _ in range(10):
		Ui.act_on(my_MPS)
		my_MPS.orthonormalise()
print('j = ', j, 'and current energy expectation value per particle is ', Ui.log_expectation_value(my_MPS)/my_n)