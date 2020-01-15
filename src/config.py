from easydict import EasyDict as edict

# C.PRIVATE determines which variations of
# problems 1, 2 and 3 should be solved.
C = edict()
C.PRIVATE = edict()

C.PRIVATE.B1 = 57
C.PRIVATE.G1 = 2017

C.PRIVATE.B2 = 310
C.PRIVATE.G2 = 2017

_digit_sum = lambda num : sum(list(map(int, list(str(num)))))
C.PRIVATE.B1_DS = _digit_sum(C.PRIVATE.B1)
C.PRIVATE.B2_DS = _digit_sum(C.PRIVATE.B2)

C.PRIVATE.P = (C.PRIVATE.B1 + C.PRIVATE.B2) % 3 + 1
C.PRIVATE.Q = (C.PRIVATE.B1_DS + C.PRIVATE.B2_DS) % 8 + 1


# Problem 1
C.PROBLEM1 = edict()

C.PROBLEM1.A = (C.PRIVATE.B1 + C.PRIVATE.G1) % 7 + 1
C.PROBLEM1.B = (C.PRIVATE.B2 + C.PRIVATE.G2) % 4 + 3
C.PROBLEM1.f1 = 5 * (C.PRIVATE.B1_DS % 4 + 1)
C.PROBLEM1.f2 = 5 * (C.PRIVATE.B2_DS % 4 + 1)
C.PROBLEM1.COARSE_EPOCHS = 2000
C.PROBLEM1.FINE_EPOCHS = 1000
C.PROBLEM1.BATCH_SIZE = 120


# Problem 2
C.PROBLEM2 = edict()

C.PROBLEM2.DATASET_PATH = '../data2/dataset{}.mat'.format(C.PRIVATE.P)
C.PROBLEM2.UNDERFIT = True
C.PROBLEM2.OPTIMAL = True
C.PROBLEM2.OVERFIT = True
C.PROBLEM2.EPOCHS = 1000
C.PROBLEM2.BATCH_SIZE = 120


# Problem 3
C.PROBLEM3 = edict()

C.PROBLEM3.TEST_DATASET_PATH = '../data3/PenDigits/pendigits.tes'
C.PROBLEM3.TRAIN_DATASET_PATH = '../data3/PenDigits/pendigits.tra'

