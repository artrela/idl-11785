# DO NOT EDIT this file. It is set up in such a way that if you make any edits,
# the test cases may change resulting in a broken local autograder.

# Imports
import sys
import json

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from test_mc import MCQTest
from test_rnn import RNNTest
from test_gru import GRUTest
from test_ctc import CTCTest
from test_ctc_decoding import SearchTest
from test import Test

mcq_test = MCQTest()
rnn_test = RNNTest()
gru_test = GRUTest()
ctc_test = CTCTest()
search_test = SearchTest()

print("Autograder - v1.1")


test_list = {
    'mcq' : [
        {'title': 'Section 1a - Multiple Choice Questions',
        'autolab': 'MCQ',
        'test': mcq_test.run_test,
        'score': 5
        }
    ],
    'rnn': [
        {'title': 'Section 2.1 - RNN Forward',
        'autolab': 'RNN_Forward',
        'test': rnn_test.test_rnncell_forward,
        'score': 5
        },
        {'title': 'Section 2.2 - RNN Backward',
        'autolab': 'RNN_Backward',
        'test': rnn_test.test_rnncell_backward,
        'score': 5
        },
        {'title': 'Section 2.3 - RNN Classifier',
        'autolab': 'RNN_Classifier',
        'test': rnn_test.test_rnn_classifier,
        'score': 10
        }
    ],
    'gru': [
        {'title': 'Section 3.1 - GRU Forward',
        'autolab': 'GRU_Forward',
        'test': gru_test.test_gru_forward,
        'score': 5
        },
        {'title': 'Section 3.2 - GRU Backward',
        'autolab': 'GRU_Backward',
        'test': gru_test.test_gru_backward,
        'score': 15
        },
        {'title': 'Section 3.3 - GRU Inference',
        'autolab': 'GRU_Inference',
        'test': gru_test.test_gru_inference,
        'score': 10
        }
    ],
    'ctc': [
        {'title': 'Section 4 - Extend Sequence with Blank',
        'autolab': 'Ext_Seq',
        'test': ctc_test.test_ctc_extend_seq,
        'score': 5
        },
        {'title': 'Section 4 - Posterior Probability',
        'autolab': 'Post_Prob',
        'test': ctc_test.test_ctc_posterior_prob,
        'score': 5
        },
        {'title': 'Section 4.1 - CTC Forward',
        'autolab': 'CTC_Forward',
        'test': ctc_test.test_ctc_forward,
        'score': 10
        },
        {'title': 'Section 4.2 - CTC Backward',
        'autolab': 'CTC_Backward',
        'test': ctc_test.test_ctc_backward,
        'score': 5
        }
    ],
    'search': [
        {'title': 'Section 5.1 - Greedy Search',
        'autolab': 'Greedy',
        'test': search_test.test_greedy_search,
        'score': 5
        },
        {'title': 'Section 5.2 - Beam Search',
        'autolab': 'Beam',
        'test': search_test.test_beam_search,
        'score': 15
        }
    ]
}


############################################################################################
########################## Test Cases - DO NOT EDIT ########################################
############################################################################################

if __name__ == "__main__":
    # # DO NOT EDIT
    if len(sys.argv) == 1:
        # run all tests
        tests = [test for sublist in test_list.values() for test in sublist]
        pass
    elif len(sys.argv) == 2:
        # run only tests for specified section
        test_type = sys.argv[1]
        if test_type in test_list:
            tests = test_list[test_type]
        else:
            sys.exit('Invalid test type option provided.\nEnter one of [mcq, rnn, gru, ctc, search.\nOr leave empty to run all tests.]')

    test = Test()
    for testcase in tests:
        test.run_tests(testcase['title'], testcase['test'], testcase['score'])

    # printing score summary
    print(f' {"_"*42}{"_"*16}')
    print(f'|{"TASK":<42}|{"" :<5}{"SCORE":<10}|')
    print(f'|{"_"*42}|{"_"*15}|')
    for title, score in test.scores.items():
        print(f'|{title:<42}|{"" :<5}{score:<10}|')
        print(f'|{"_"*42}|{"_"*15}|')
    print(f'|{"TOTAL SCORE":<42}|{"" :<5}{test.get_test_scores():<10}|')
    print(f'|{"_"*42}|{"_"*15}|')

    print("\n")
    print(json.dumps({"scores": test.scores}))
