import logging

tile_symbol = [['🀇', '🀈', '🀉', '🀊', '🀋', '🀌', '🀍', '🀎', '🀏'],
               ['🀙', '🀚', '🀛', '🀜', '🀝', '🀞', '🀟', '🀠', '🀡'],
               ['🀐', '🀑', '🀒', '🀓', '🀔', '🀕', '🀖', '🀖', '🀘'],
               ['🀀', '🀁', '🀂', '🀃', '🀆', '🀅', '🀄']]


def dbglvl_config(level):
    logging.basicConfig(level=level)


def convert(tile_str):
    """
    Convert the abbr code of a hand to tile lists.

    E.g. "123m234567p555s11z" to [123][234567][555][11]
    :param tile_str: string of tiles
    :return: list of mpsz
    """
    man = []
    pin = []
    so = []
    zu = []
    if (tile_str.find('m') != -1):
        tile_str = tile_str.split('m', 1)
        man_str = tile_str[0]
        tile_str = tile_str[1]
        man = list(man_str)
    if (tile_str.find('p') != -1):
        tile_str = tile_str.split('p', 1)
        pin_str = tile_str[0]
        tile_str = tile_str[1]
        pin = list(pin_str)
    if (tile_str.find('s') != -1):
        tile_str = tile_str.split('s', 1)
        so_str = tile_str[0]
        tile_str = tile_str[1]
        so = list(so_str)
    if (tile_str.find('z') != -1):
        tile_str = tile_str.split('z', 1)
        zu_str = tile_str[0]
        tile_str = tile_str[1]
        zu = list(zu_str)
    return man, pin, so, zu


class Hand:
    """

    """
    def __init__(self, tile_str) -> None:
        self.dbgf = logging.getLogger("class_Hand")
        dbglvl_config(logging.INFO)
        self.__man = []
        self.__pin = []
        self.__so = []
        self.__zu = []
        self.__man, self.__pin, self.__so, self.__zu = convert(tile_str)
        self.__check_tiles()

    def __check_tiles(self):
        tile_sum = len(self.__man) + len(self.__pin) + len(self.__so) + len(self.__zu)
        if (tile_sum != 13):
            self.dbgf.error("Hand is not 14 tiles")
            return False
        return True

    # calculate waiting tiles(向听数)
    # 一般型
    def cal_waiting_num_normal(self):
        hand = self.__get_hand()
        mianzi_sum, pair_sum, dazi_sum = self.hand_segment(hand)

        waiting_num_normal = 0
        # 搭子未溢出
        if (mianzi_sum + dazi_sum + pair_sum <= 5):
            waiting_num_normal = 8 - 2 * mianzi_sum - pair_sum - dazi_sum
        # 搭子溢出
        if (mianzi_sum + dazi_sum + pair_sum > 5):
            # 有雀头
            if (pair_sum > 0):
                waiting_num_normal = 3 - mianzi_sum
            # 无雀头
            if (pair_sum == 0):
                waiting_num_normal = 4 - mianzi_sum
        self.dbgf.info("一般型:%d向听" % waiting_num_normal)
        return waiting_num_normal

    # 七对子型
    def cal_waiting_num_7pairs(self):
        hand = self.__get_hand()
        pair_sum = 0
        for tile_list in hand:
            if (tile_list):
                # kangzi can be used as 1 pair
                kangzi_cnt, tile_list = self.__find_kangzi(tile_list)
                pair_cnt, tile_list = self.__find_pair(tile_list)
                pair_sum += kangzi_cnt + pair_cnt
        self.dbgf.debug("pair:%d" % pair_sum)

        waiting_num_7pairs = 6 - pair_sum
        self.dbgf.info("七对子:%d向听" % waiting_num_7pairs)
        return waiting_num_7pairs

    # 国士无双型
    def cal_waiting_num_13orphans(self):
        orphan_num = self.__find_orphan()
        self.dbgf.debug("orphan:%d" % orphan_num)
        waiting_num_13orphans = 13 - orphan_num
        self.dbgf.info("国士无双:%d向听" % waiting_num_13orphans)
        return waiting_num_13orphans

    # calculate income tiles(进张)
    def cal_income_tiles(self):
        return []

    # split hand to mianzi and dazi
    # return mianzi_sum, pair_sum, dazi_sum of the best segment
    def hand_segment(self, hand):
        candidate_max = 0
        best_segment = 0, 0, 0

        # use single tiles for shunzi first
        self.dbgf.debug("use single tiles for shunzi first")
        temp_hand = hand.copy()
        mianzi_sum = 0
        pair_sum = 0
        dazi_sum = 0
        for tile_list in [temp_hand[0], temp_hand[1], temp_hand[2]]:
            if (tile_list):
                tile_list_rmv_dup = [x for x in tile_list if tile_list.count(x) == 1]
                self.dbgf.debug("remove duplicate %s" % tile_list_rmv_dup)
                shunzi_cnt, tile_list_remain = self.__find_shunzi(tile_list_rmv_dup)
                tile_list_dup = [x for x in tile_list if tile_list.count(x) > 1]
                tile_list = sorted(tile_list_remain + tile_list_dup)
                self.dbgf.debug("hand remove shunzi %s" % tile_list)
                # kangzi equals single tile, should be regarded as kezi + single
                # kangzi_cnt, tile_list = self.__find_kangzi(tile_list)
                kezi_cnt, tile_list = self.__find_kezi(tile_list)
                pair_cnt, dazi_cnt = self.__find_candidate(tile_list)
                mianzi_sum += shunzi_cnt + kezi_cnt
                pair_sum += pair_cnt
                dazi_sum += dazi_cnt
        if (temp_hand[3]):
            pair_cnt, tile_list = self.__find_pair(temp_hand[3])
            pair_sum += pair_cnt
        self.dbgf.debug("mianzi:%d, pair:%d, dazi:%d" % (mianzi_sum, pair_sum, dazi_sum))
        if (mianzi_sum + pair_sum + dazi_sum > candidate_max):
            candidate_max = mianzi_sum + pair_sum + dazi_sum
            best_segment = mianzi_sum, pair_sum, dazi_sum

        # use all duplicated tiles
        self.dbgf.debug("use all of duplicate tiles")
        temp_hand = hand.copy()
        mianzi_sum = 0
        pair_sum = 0
        dazi_sum = 0
        for tile_list in [temp_hand[0], temp_hand[1], temp_hand[2]]:
            if (tile_list):
                shunzi_cnt, tile_list = self.__find_shunzi(tile_list)
                kezi_cnt, tile_list = self.__find_kezi(tile_list)
                pair_cnt, dazi_cnt = self.__find_candidate(tile_list)
                mianzi_sum += shunzi_cnt + kezi_cnt
                pair_sum += pair_cnt
                dazi_sum += dazi_cnt
        if (hand[3]):
            pair_cnt, tile_list = self.__find_pair(hand[3])
            pair_sum += pair_cnt
        self.dbgf.debug("mianzi:%d, pair:%d, dazi:%d" % (mianzi_sum, pair_sum, dazi_sum))
        if (mianzi_sum + pair_sum + dazi_sum > candidate_max):
            candidate_max = mianzi_sum + pair_sum + dazi_sum
            best_segment = mianzi_sum, pair_sum, dazi_sum

        return best_segment

    # find shunzi in tile list 345
    # return shunzi_cnt and shunzi-removed tile_list
    def __find_shunzi(self, tile_list):
        shunzi_cnt = 0
        i = 0
        while (i < len(tile_list)):
            tile = tile_list[i]
            shunzi = [tile, str(int(tile) + 1), str(int(tile) + 2)]
            while (set(shunzi).issubset(set(tile_list))):
                self.dbgf.debug("found shunzi %s" % shunzi)
                shunzi_cnt += 1
                for j in shunzi:
                    tile_list.remove(j)
                self.dbgf.debug("hand after remove %s" % tile_list)
                if (i < len(tile_list)):
                    tile = tile_list[i]
                    shunzi = [tile, str(int(tile) + 1), str(int(tile) + 2)]
            i += 1
        return shunzi_cnt, tile_list

    # find kezi in tile list 333
    # return kezi_cnt and kezi-removed tile_list
    def __find_kezi(self, tile_list):
        kezi_cnt = 0
        i = 0
        while (i < len(tile_list)):
            tile = tile_list[i]
            while (tile_list.count(tile) >= 3):
                self.dbgf.debug("found kezi %s" % ([tile, tile, tile]))
                kezi_cnt += 1
                for j in range(3):
                    tile_list.remove(tile)
                self.dbgf.debug("hand after remove %s" % tile_list)
                if (i < len(tile_list)):
                    tile = tile_list[i]
            i += 1
        return kezi_cnt, tile_list

    # find kangzi in tile list 9999
    # return kangzi_cnt and kangzi-removed tile_list
    def __find_kangzi(self, tile_list):
        kangzi_cnt = 0
        i = 0
        while (i < len(tile_list)):
            tile = tile_list[i]
            while (tile_list.count(tile) == 4):
                self.dbgf.debug("found kangzi %s" % ([tile, tile, tile, tile]))
                kangzi_cnt += 1
                for j in range(4):
                    tile_list.remove(tile)
                self.dbgf.debug("hand after remove %s" % tile_list)
                if (i < len(tile_list)):
                    tile = tile_list[i]
            i += 1
        return kangzi_cnt, tile_list

    # find pair in tile list 77
    # return pair_cnt and pair-removed tile_list
    def __find_pair(self, tile_list):
        pair_cnt = 0
        i = 0
        while (i < len(tile_list)):
            tile = tile_list[i]
            while (tile_list.count(tile) >= 2):
                self.dbgf.debug("found pair %s" % ([tile, tile]))
                pair_cnt += 1
                for j in range(2):
                    tile_list.remove(tile)
                self.dbgf.debug("hand after remove %s" % tile_list)
                if (i < len(tile_list)):
                    tile = tile_list[i]
            i += 1
        return pair_cnt, tile_list

    # find dazi in tile list 45, 46
    # return dazi_cnt and dazi-removed tile_list
    def __find_dazi(self, tile_list):
        dazi_cnt = 0
        i = 0
        while (i < len(tile_list)):
            tile = tile_list[i]
            _dazi_ = [tile, str(int(tile) + 1)]  # 两面搭
            da_zi = [tile, str(int(tile) + 2)]  # 嵌张搭
            while (set(_dazi_).issubset(set(tile_list)) or
                   set(da_zi).issubset(set(tile_list))):
                if (set(_dazi_).issubset(set(tile_list))):
                    self.dbgf.debug("found _dazi_ %s" % _dazi_)
                    dazi_cnt += 1
                    for j in _dazi_:
                        tile_list.remove(j)
                    self.dbgf.debug("hand after remove %s" % tile_list)
                if (set(da_zi).issubset(set(tile_list))):
                    self.dbgf.debug("found da_zi %s" % da_zi)
                    dazi_cnt += 1
                    for j in da_zi:
                        tile_list.remove(j)
                    self.dbgf.debug("hand after remove %s" % tile_list)
                if (i < len(tile_list)):
                    tile = tile_list[i]
                    _dazi_ = [tile, str(int(tile) + 1)]
                    da_zi = [tile, str(int(tile) + 2)]
            i += 1
        return dazi_cnt, tile_list

    # find best segment of pairs and dazi
    # ex: 2335
    # return pair_cnt and dazi_cnt
    def __find_candidate(self, tile_list):
        self.dbgf.debug("calculate pair first")
        pair_first_temp_list = tile_list.copy()
        pair_first_pair_cnt, pair_first_temp_list = self.__find_pair(pair_first_temp_list)
        pair_first_dazi_cnt, pair_first_temp_list = self.__find_dazi(pair_first_temp_list)
        pair_first_cnt = pair_first_pair_cnt + pair_first_dazi_cnt

        self.dbgf.debug("calculate dazi first")
        dazi_first_temp_list = tile_list.copy()
        dazi_first_dazi_cnt, dazi_first_temp_list = self.__find_dazi(dazi_first_temp_list)
        dazi_first_pair_cnt, dazi_first_temp_list = self.__find_pair(dazi_first_temp_list)
        dazi_first_cnt = dazi_first_pair_cnt + dazi_first_dazi_cnt

        if (pair_first_cnt > dazi_first_cnt):
            self.dbgf.debug("pair first, found pair:%d, dazi:%d" % (pair_first_pair_cnt, pair_first_dazi_cnt))
            return pair_first_pair_cnt, pair_first_dazi_cnt
        else:
            self.dbgf.debug("dazi first, found pair:%d, dazi:%d" % (dazi_first_pair_cnt, dazi_first_dazi_cnt))
            return dazi_first_pair_cnt, dazi_first_dazi_cnt

    # find orphan in hand 19m19p19s1234567z
    def __find_orphan(self):
        hand = self.__get_hand()
        orphan_cnt = 0
        # 老头牌
        for i in range(3):
            orphan_cnt += hand[i].count('1') + hand[i].count('9')
        # 字牌
        orphan_cnt += len(hand[3])
        return orphan_cnt

    # return hand in list form
    def __get_hand(self):
        return [self.__man.copy(), self.__pin.copy(), self.__so.copy(), self.__zu.copy()]

    # print hand in 3 formats: "symbol", "code", "list"
    def print_hand(self, format):
        match format:
            case "symbol":
                symbol_str = ""
                hand = self.__get_hand()
                for i in range(4):
                    for tile in hand[i]:
                        symbol_str += tile_symbol[i][int(tile) - 1] + ' '
                self.dbgf.info(symbol_str)
            case "code":
                abbr_str = ""
                hand = self.__get_hand()
                mpsz = ['m', 'p', 's', 'z']
                for i in range(4):
                    if (hand[i]):
                        abbr_str += "".join(hand[i])
                        abbr_str += mpsz[i]
                self.dbgf.info(abbr_str)
            case "list":
                hand = self.__get_hand()
                self.dbgf.info(hand)
