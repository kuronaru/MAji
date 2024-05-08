from src.utils.hand import Hand

DATAPATH = "../data/data.txt"
with open(DATAPATH, 'r') as f:
    data = f.read().splitlines()
    for hand_str in data:
        hand = Hand(hand_str)
        hand.print_hand("code")
        hand.cal_waiting_num_normal()
        hand.cal_waiting_num_7pairs()
        hand.cal_waiting_num_13orphans()
    f.close()