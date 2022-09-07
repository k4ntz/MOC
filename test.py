class RectBB():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.is_used = False

    def touches(self, other):
        return (self.x <= other.x <= self.x_max or other.x <= self.x <= other.x_max) \
               and \
               (self.y <= other.y <= self.y_max or other.y <= self.y <= other.y_max)

    def merge(self, other):
        return RectBB(min(self.x, other.x), min(self.y, other.y),
                      max(self.x_max, other.x_max) - min(self.x, other.x),
                      max(self.y_max, other.y_max) - min(self.y, other.y))

    @property
    def x_max(self):
        return self.x + self.w

    @property
    def y_max(self):
        return self.y + self.h

    def __iter__(self):
        return iter((self.x, self.y, self.w, self.h))

    def __repr__(self):
        return f"RectBB{self.x, self.y, self.w, self.h}"


bbs = [RectBB(67, 202, 13, 6), RectBB(67, 202, 13, 6), RectBB(98, 185, 6, 14), RectBB(101, 159, 3, 6),
       RectBB(101, 159, 3, 6), RectBB(18, 90, 8, 14), RectBB(17, 88, 7, 2), RectBB(105, 67, 9, 16),
       RectBB(106, 68, 8, 15), RectBB(105, 67, 3, 3), RectBB(73, 67, 9, 16), RectBB(74, 68, 8, 15),
       RectBB(73, 67, 3, 3), RectBB(41, 67, 9, 16), RectBB(42, 68, 8, 15), RectBB(41, 67, 3, 3),
       RectBB(58, 48, 8, 14), RectBB(42, 47, 9, 15), RectBB(57, 46, 7, 2), RectBB(73, 23, 6, 4),
       RectBB(81, 22, 1, 1), RectBB(89, 20, 7, 7), RectBB(79, 20, 2, 2), RectBB(83, 19, 3, 1),
       RectBB(81, 14, 7, 5), RectBB(70, 14, 4, 3), RectBB(83, 5, 1, 1), RectBB(88, 4, 3, 3),
       RectBB(88, 4, 3, 3), RectBB(80, 2, 6, 6)]


def cmerge(bb, bb_other):
    if (bb is not bb_other) and bb.touches(bb_other):
        bb = bb.merge(bb_other)
        bb_other.is_used = True
    return bb

cur_bb = bbs[0]

for bb in bbs:
    cur_bb = cmerge(cur_bb, bb)





