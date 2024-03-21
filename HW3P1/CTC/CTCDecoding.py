import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = ["-"] + symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1
        
        S, T, B = y_probs.shape

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)

        #return decoded_path, path_prob
        # raise NotImplementedError
        # for b in range(B):
            
        max_idxs = np.argmax(y_probs[:, :, 0], axis=0)
        
        for i, idx in enumerate(max_idxs):
            
            if max_idxs[i] != max_idxs[i-1]:
                decoded_path.append(idx)
                
            path_prob *= y_probs[:, :, 0][idx, i]
        
        decoded_path = ''.join([self.symbol_set[idx] for idx in decoded_path[:-1]])
        # decoded_path.append(decoded_path)
        
        return decoded_path, path_prob


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = ["-"] + symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        T = y_probs.shape[1]
        bestPath, FinalPathScore = {"-": 1.0}, 0
        tempPaths = {}
        
        for t in range(T):
            # clip beam within width
            currProb = y_probs[:, t, 0]
            bestPath = dict(sorted(bestPath.items(), key=lambda prob: prob[1])[-self.beam_width:])

            for path, score in bestPath.items():
                
                for idx, prob in enumerate(currProb):
                    
                    sym = self.symbol_set[idx]
                    new_prob = prob * score
                    new_path = path 
                    
                    if path[-1] == "-": 
                        new_path = path[:-1] + sym
                    elif sym != path[-1] and not (t==T-1 and sym=='-'):
                        new_path += sym
                    
                    if new_path in tempPaths:
                        tempPaths[new_path] += new_prob
                    else:
                        tempPaths[new_path] = new_prob
                    
            bestPath = tempPaths
            tempPaths = {}
        
        best = max(bestPath, key=bestPath.get)
        prunedPaths = { path[:-1] if path[-1] == "-" else path: prob for path, prob in bestPath.items() }
        
        return best, prunedPaths
