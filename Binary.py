#Binary counting
def solution(N):
    bn = str(bin(N))[2:]
    bn_group = False
    bn_highest = 0
    bn_zero_counter = 0
    for char in br :
        if char == '1' :
            if bn_highest < bn_zero_counter :
                bn_highest = bn_zero_counter
            bn_group = True
            bn_zero_counter = 0
        elif bn_group :
            bn_zero_counter +=1
    return bn_highest
    pass
