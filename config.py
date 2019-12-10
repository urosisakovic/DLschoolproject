from easydict import EasyDict as edict

#TODO: All caps
C = edict()
C.PRIVATE = edict()

C.PRIVATE.B1 = 57
C.PRIVATE.G1 = 2017

C.PRIVATE.B2 = 57
C.PRIVATE.G2 = 2017

_digit_sum = lambda num : sum(list(map(int, list(str(num)))))
C.PRIVATE.B1_digit_sum = _digit_sum(C.PRIVATE.B1)
C.PRIVATE.B2_digit_sum = _digit_sum(C.PRIVATE.B2)

C.A = (C.PRIVATE.B1 + C.PRIVATE.G1) % 7 + 1
C.B = (C.PRIVATE.B2 + C.PRIVATE.G2) % 4 + 3
C.f1 = 5 * (C.PRIVATE.B1_digit_sum % 4 + 1)
C.f2 = 5 * (C.PRIVATE.B2_digit_sum % 4 + 1)

# Problem1 hyperparameters:
C.PROBLEM1 = edict()
C.PROBLEM1.interval_start = 0
C.PROBLEM1.interval_end = 100
C.PROBLEM1.steps = 1000