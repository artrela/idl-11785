import numpy as np


class CTC(object):

	def __init__(self, BLANK=0):
		"""
		
		Initialize instance variables

		Argument(s)
		-----------
		
		BLANK (int, optional): blank label index. Default 0.

		"""

		# No need to modify
		self.BLANK = BLANK


	def extend_target_with_blank(self, target):
		"""Extend target sequence with blank.

		Input
		-----
		target: (np.array, dim = (target_len,))
				target output containing indexes of target phonemes
		ex: [1,4,4,7]

		Return
		------
		extSymbols: (np.array, dim = (2 * target_len + 1,))
					extended target sequence with blanks
		ex: [0,1,0,4,0,4,0,7,0]

		skipConnect: (np.array, dim = (2 * target_len + 1,))
					skip connections
		ex: [0,0,0,1,0,0,0,1,0]
		"""

		extended_symbols = [self.BLANK]
		for symbol in target:
			extended_symbols.append(symbol)
			extended_symbols.append(self.BLANK)

		N = len(extended_symbols)

		# -------------------------------------------->
		skip_connect = [self.BLANK for _ in extended_symbols]
		for i in range(1, len(target)):
			if target[i] != target[i-1]:
				skip_connect[i*2 + 1] = 1
		
		# <---------------------------------------------

		extended_symbols = np.array(extended_symbols).reshape((N,))
		skip_connect = np.array(skip_connect).reshape((N,))

		return extended_symbols, skip_connect


	def get_forward_probs(self, logits, extended_symbols, skip_connect):
		"""Compute forward probabilities.

		Input
		-----
		logits: (np.array, dim = (input_len, len(Symbols)))
				predict (log) probabilities

				To get a certain symbol i's logit as a certain time stamp t:
				p(t,s(i)) = logits[t, qextSymbols[i]]

		extSymbols: (np.array, dim = (2 * target_len + 1,))
					extended label sequence with blanks

		skipConnect: (np.array, dim = (2 * target_len + 1,))
					skip connections

		Return
		------
		alpha: (np.array, dim = (input_len, 2 * target_len + 1))
				forward probabilities

		"""

		S, T = len(extended_symbols), len(logits)
		alpha = np.zeros(shape=(T, S))

		# -------------------------------------------->
		# TODO: Intialize alpha[0][0]
		# TODO: Intialize alpha[0][1]
		# TODO: Compute all values for alpha[t][sym] where 1 <= t < T and 1 <= sym < S (assuming zero-indexing)
		# IMP: Remember to check for skipConnect when calculating alpha #!!!!!!!!!! go back and check this
		# <---------------------------------------------

		alpha[0, 0] = logits[0, extended_symbols[0]]
		alpha[0, 1] = logits[0, extended_symbols[1]]

		for t in range(1, T):
			alpha[t, 0] = alpha[t-1, 0]*logits[t, extended_symbols[0]]
   
			for s in range(1, S):
				alpha[t, s] = alpha[t-1, s] + alpha[t-1, s-1]
	
				# if s > 1 and skip_connect[s] and extended_symbols[s] != extended_symbols[s-2]:
				if skip_connect[s]:
					alpha[t, s] += alpha[t-1, s-2]
	 
				alpha[t, s] *= logits[t, extended_symbols[s]]
 
		return alpha
		# raise NotImplementedError


	def get_backward_probs(self, logits, extended_symbols, skip_connect):
		"""Compute backward probabilities.

		Input
		-----
		logits: (np.array, dim = (input_len, len(symbols)))
				predict (log) probabilities

				To get a certain symbol i's logit as a certain time stamp t:
				p(t,s(i)) = logits[t,extSymbols[i]]

		extSymbols: (np.array, dim = (2 * target_len + 1,))
					extended label sequence with blanks

		skipConnect: (np.array, dim = (2 * target_len + 1,))
					skip connections

		Return
		------
		beta: (np.array, dim = (input_len, 2 * target_len + 1))
				backward probabilities
		
		"""

		S, T = len(extended_symbols), len(logits)
		beta = np.zeros(shape=(T, S))

		# -------------------------------------------->
		S -= 1
		T -= 1
  
		beta[T, S] = logits[T, extended_symbols[S]]
		beta[T, S-1] = logits[T, extended_symbols[S-1]]

		for t in range(T-1, -1, -1):
			beta[t, S] = beta[t+1, S] * logits[t, extended_symbols[S]]
   
			for s in range(S-1, -1, -1):
				beta[t, s] = beta[t+1, s] + beta[t+1, s+1]
	
				if s <= S-2 and skip_connect[s+2]:
					beta[t, s] += beta[t+1, s+2]
	 
				beta[t, s] *= logits[t, extended_symbols[s]]
	
		
		for t in range(T, -1, -1):
			for s in range(S, -1, -1):
				beta[t, s] /= logits[t, extended_symbols[s]]
  
  
		# <--------------------------------------------

		return beta
		# raise NotImplementedError
		

	def get_posterior_probs(self, alpha, beta):
		"""Compute posterior probabilities.

		Input
		-----
		alpha: (np.array, dim = (input_len, 2 * target_len + 1))
				forward probability

		beta: (np.array, dim = (input_len, 2 * target_len + 1))
				backward probability

		Return
		------
		gamma: (np.array, dim = (input_len, 2 * target_len + 1))
				posterior probability

		"""

		[T, S] = alpha.shape
		gamma = np.zeros(shape=(T, S))
		sumgamma = np.zeros((T,))

		# -------------------------------------------->
		
		for t in range(T):
	  
			for s in range(S):
				gamma[t, s] = alpha[t, s] * beta[t, s]
				sumgamma[t] += gamma[t, s]

			for s in range(S):
				gamma[t, s] /= sumgamma[t]
  
		# <---------------------------------------------

		return gamma 
		# raise NotImplementedError


class CTCLoss(object):

	def __init__(self, BLANK=0):
		"""

		Initialize instance variables

		Argument(s)
		-----------
		BLANK (int, optional): blank label index. Default 0.
		
		"""
		# -------------------------------------------->
		# No need to modify
		super(CTCLoss, self).__init__()

		self.BLANK = BLANK
		self.gammas = []
		self.ctc = CTC()
		# <---------------------------------------------

	def __call__(self, logits, target, input_lengths, target_lengths):

		# No need to modify
		return self.forward(logits, target, input_lengths, target_lengths)

	def forward(self, logits, target, input_lengths, target_lengths):
		"""CTC loss forward

		Computes the CTC Loss by calculating forward, backward, and
		posterior proabilites, and then calculating the avg. loss between
		targets and predicted log probabilities

		Input
		-----
		logits [np.array, dim=(seq_length, batch_size, len(symbols)]:
			log probabilities (output sequence) from the RNN/GRU

		target [np.array, dim=(batch_size, padded_target_len)]:
			target sequences

		input_lengths [np.array, dim=(batch_size,)]:
			lengths of the inputs

		target_lengths [np.array, dim=(batch_size,)]:
			lengths of the target

		Returns
		-------
		loss [float]:
			avg. divergence between the posterior probability and the target

		"""

		# No need to modify
		self.logits = logits
		self.target = target
		self.input_lengths = input_lengths
		self.target_lengths = target_lengths

		#####  IMP:
		#####  Output losses should be the mean loss over the batch

		# No need to modify
		B, _ = target.shape
		total_loss = np.zeros(B)
		self.extended_symbols = []
  
		for batch_itr in range(B):
			# -------------------------------------------->
			# Computing CTC Loss for single batch
			# Process:
			#	 Truncate the target to target length
			target_trunc = target[batch_itr, :target_lengths[batch_itr]]
			#	 Truncate the logits to input length
			logits_trunc = logits[:input_lengths[batch_itr], batch_itr, :]
			#	 Extend target sequence with blank
			target_trunc, skip = self.ctc.extend_target_with_blank(target_trunc)
			#	 Compute forward probabilities
			alpha = self.ctc.get_forward_probs(logits_trunc, target_trunc, skip)
			#	 Compute backward probabilities
			beta = self.ctc.get_backward_probs(logits_trunc, target_trunc, skip)
			#	 Compute posteriors using total probability function
			gamma = self.ctc.get_posterior_probs(alpha, beta)
			#	 Compute expected divergence for each batch and store it in totalLoss
			#	 Take an average over all batches and return final result
			self.gammas.append(gamma)
			self.extended_symbols.append(target_trunc)
			#####  IMP:
			#####  Output losses should be the mean loss over the batch
			for t in range(len(target_trunc)):
				for s in range(len(logits_trunc)):
					total_loss[batch_itr] -= gamma[s][t] * np.log(logits_trunc[s][target_trunc[t]]) 
			# <---------------------------------------------
   
			# <---------------------------------------------

		total_loss = np.mean(total_loss)
		
		return total_loss
		# raise NotImplementedError
		

	def backward(self):
		"""
		
		CTC loss backard

		Calculate the gradients w.r.t the parameters and return the derivative 
		w.r.t the inputs, xt and ht, to the cell.

		Input
		-----
		logits [np.array, dim=(seqlength, batch_size, len(Symbols)]:
			log probabilities (output sequence) from the RNN/GRU

		target [np.array, dim=(batch_size, padded_target_len)]:
			target sequences

		input_lengths [np.array, dim=(batch_size,)]:
			lengths of the inputs

		target_lengths [np.array, dim=(batch_size,)]:
			lengths of the target

		Returns
		-------
		dY [np.array, dim=(seq_length, batch_size, len(extended_symbols))]:
			derivative of divergence w.r.t the input symbols at each time

		"""

		# No need to modify
		T, B, C = self.logits.shape	
		dY = np.full_like(self.logits, 0)
  
		# we have a dY that finds the divergence 

		for batch_itr in range(B):
			# -------------------------------------------->
			# Computing CTC Derivative for single batch
			# Process:
			#	 Truncate the target to target length
			#	 Truncate the logits to input length
			#	 Extend target sequence with blank
			#	 Compute derivative of divergence and store them in dY
			# <---------------------------------------------

			# -------------------------------------------->

			target_trunc = self.target[batch_itr, :self.target_lengths[batch_itr]]
			#	 Truncate the logits to input length
			logits_trunc = self.logits[:self.input_lengths[batch_itr], batch_itr, :]
			#	 Extend target sequence with blank
			target_trunc, skip = self.ctc.extend_target_with_blank(target_trunc)
     
			for t in range(self.input_lengths[batch_itr]):
				for c in range(len(target_trunc)):
					# t = T, c = C 
					dY[t, batch_itr, target_trunc[c]] -= self.gammas[batch_itr][t][c] / logits_trunc[t][target_trunc[c]] 
			# <---------------------------------------------

		return dY
		# raise NotImplementedError
