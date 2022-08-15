import torch

rbox = torch.zeros(3, 21, 21)
rbox[0, :2, :] = 1
rbox[0, -2:, :] = 1
rbox[0, :, :2] = 1
rbox[0, :, -2:] = 1
rbox = rbox.view(1, 3, 21, 21)

gbox = torch.zeros(3, 21, 21)
gbox[1, :2, :] = 1
gbox[1, -2:, :] = 1
gbox[1, :, :2] = 1
gbox[1, :, -2:] = 1
gbox = gbox.view(1, 3, 21, 21)

blbox = torch.zeros(3, 21, 21)
blbox[2, :2, :] = 1
blbox[2, -2:, :] = 1
blbox[2, :, :2] = 1
blbox[2, :, -2:] = 1
blbox = blbox.view(1, 3, 21, 21)

drbox = torch.zeros(3, 21, 21)
drbox[0, :2, :] = .6
drbox[0, -2:, :] = .6
drbox[0, :, :2] = .6
drbox[0, :, -2:] = .6
drbox = drbox.view(1, 3, 21, 21)

dgbox = torch.zeros(3, 21, 21)
dgbox[1, :2, :] = .6
dgbox[1, -2:, :] = .6
dgbox[1, :, :2] = .6
dgbox[1, :, -2:] = .6
dgbox = dgbox.view(1, 3, 21, 21)

dblbox = torch.zeros(3, 21, 21)
dblbox[2, :2, :] = .6
dblbox[2, -2:, :] = .6
dblbox[2, :, :2] = .6
dblbox[2, :, -2:] = .6
dblbox = blbox.view(1, 3, 21, 21)

ybox = torch.zeros(3, 21, 21)
ybox[0, :2, :] = 1
ybox[0, -2:, :] = 1
ybox[0, :, :2] = 1
ybox[0, :, -2:] = 1
ybox[1, :2, :] = 1
ybox[1, -2:, :] = 1
ybox[1, :, :2] = 1
ybox[1, :, -2:] = 1
ybox = ybox.view(1, 3, 21, 21)

abox = torch.zeros(3, 21, 21)
abox[1, :2, :] = 1
abox[1, -2:, :] = 1
abox[1, :, :2] = 1
abox[1, :, -2:] = 1
abox[2, :2, :] = 1
abox[2, -2:, :] = 1
abox[2, :, :2] = 1
abox[2, :, -2:] = 1
abox = abox.view(1, 3, 21, 21)

pbox = torch.zeros(3, 21, 21)
pbox[0, :2, :] = 1
pbox[0, -2:, :] = 1
pbox[0, :, :2] = 1
pbox[0, :, -2:] = 1
pbox[2, :2, :] = 1
pbox[2, -2:, :] = 1
pbox[2, :, :2] = 1
pbox[2, :, -2:] = 1
pbox = pbox.view(1, 3, 21, 21)

dybox = torch.zeros(3, 21, 21)
dybox[0, :2, :] = .6
dybox[0, -2:, :] = .6
dybox[0, :, :2] = .6
dybox[0, :, -2:] = .6
dybox[1, :2, :] = .6
dybox[1, -2:, :] = .6
dybox[1, :, :2] = .6
dybox[1, :, -2:] = .6
dybox = dybox.view(1, 3, 21, 21)

dabox = torch.zeros(3, 21, 21)
dabox[1, :2, :] = .6
dabox[1, -2:, :] = .6
dabox[1, :, :2] = .6
dabox[1, :, -2:] = .6
dabox[2, :2, :] = .6
dabox[2, -2:, :] = .6
dabox[2, :, :2] = .6
dabox[2, :, -2:] = .6
dabox = dabox.view(1, 3, 21, 21)

dpbox = torch.zeros(3, 21, 21)
dpbox[0, :2, :] = .6
dpbox[0, -2:, :] = .6
dpbox[0, :, :2] = .6
dpbox[0, :, -2:] = .6
dpbox[2, :2, :] = .6
dpbox[2, -2:, :] = .6
dpbox[2, :, :2] = .6
dpbox[2, :, -2:] = .6
dpbox = dpbox.view(1, 3, 21, 21)

boxes = torch.cat((rbox, gbox, blbox, ybox, abox, pbox, drbox, dgbox, dblbox,
                   dybox, dabox, dpbox))
