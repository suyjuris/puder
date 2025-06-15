
import datetime
import os
import random

import torch
from PIL import Image
import hsluv

def _palette(n):
    pp = getattr(_palette, 'cache', None)
    if pp is not None and pp.shape[0] >= n:
        return pp[:n]

    p = torch.zeros(n, 3, dtype=torch.uint8)
    have = 1
    if pp is not None:
        p[:pp.shape[0]] = pp
        have = pp.shape[0]
        
    for i in range(have,n):
        hue = 0.61803398875*i % 1.0
        sat = 0.4+0.6*(0.4142135623730951*i % 1.0)
        lig = 0.3+0.4*(0.7320508075688772*i % 1.0)
        r,g,b = hsluv.hsluv_to_rgb([hue*360, sat*100, lig*100])
        p[i] = torch.tensor([r*255, g*255, b*255], dtype=p.dtype)

    _palette.cache = p
    return p

def _make_rect(count, size_x, size_y, target_x=16, target_y=9):
    import math
    x = ((size_y / size_x * count) * target_x / target_y)**0.5
    y = count / x

    x0, y1 = math.ceil(x), math.ceil(y)
    y0, x1 = math.ceil(count / x0), math.ceil(count / y1)
    t = target_x / target_y
    if abs(x0/y0-t) < abs(x1/y1-t):
        return x0, y0
    else:
        return x1, y1

def _img_from_array(arr, mode):
    s = arr.shape
    if mode == 'L':
        col = 204
        ss = []
    else:
        col = 246, 240, 228
        assert s[-1] == 3
        s = s[:-1]
        ss = [3]
    
    if len(s) == 1:
        x, y = _make_rect(s[0], 1, 1)
        img = Image.new(mode, (x, y), col)

        rest = s[0] % x
        img0 = Image.fromarray(arr[:s[0]-rest].reshape(-1, x, *ss))
        img.paste(img0)
        if rest:
            img1 = Image.fromarray(arr[-rest:].reshape(1, rest, *ss))
            img.paste(img1, (0, y-1))
        return img
    elif len(s) == 2:
        return Image.fromarray(arr, mode)
    elif len(s) % 2 == 0:
        pad = len(s)
        x, y = s[1], s[0]
        img0 = _img_from_array(arr[0,0,...], mode)
        img = Image.new(mode, (x * img0.width + (x-1)*pad, y * img0.height + (y-1)*pad), col)
        img.paste(img0)
        for yi in range(y):
            for xi in range(x):
                if xi == yi == 0: continue
                img_i = _img_from_array(arr[yi,xi,...], mode)
                img.paste(img_i, (xi*(pad + img0.width), yi*(pad + img0.height)))
        return img
    else:
        pad = len(s)
        img0 = _img_from_array(arr[0,...], mode)
        x, y = _make_rect(s[0], img0.width, img0.height)
        img = Image.new(mode, (x * img0.width + (x-1)*pad, y * img0.height + (y-1)*pad), col)
        img.paste(img0)
        for i in range(1, s[0]):
            img_i = _img_from_array(arr[i,...], mode)
            xi, yi = i % x, i // x
            img.paste(img_i, (xi*(pad + img0.width), yi*(pad + img0.height)))
        return img

def imgdump(data, name=None, force_scalar=False, scale=None):
    it = imgdump.counter
    imgdump.counter += 1

    if force_scalar and not torch.is_floating_point(data):
        data = data.to(dtype=torch.float32)
    
    if torch.is_floating_point(data):
        mask_zero = None
        mask_one = None
        vmin = 0
        vmax = 1

        if scale is None:
            dmin = data.min().item()
            if dmin == -float('inf'):
                mask_zero = data == dmin
                dmin = data[~mask_zero].min().item()
                vmin = 0.1

            dmax = data.max().item()
            if dmax == float('inf'):
                mask_one = data == dmax
                dmax = data[~mask_one].max().item()
                vmax = 0.9
        else:
            dmin, dmax = scale

        if dmin == dmax:
            src = torch.full_like(data, 127, dtype=torch.uint8, device='cpu')
        else:
            t = (data - dmin) / (dmax - dmin)
            t = t * (vmax - vmin) + vmin
            t.clamp_(0.0, 1.0)
            t *= 255
            src = t.to('cpu', dtype=torch.uint8)

        if mask_zero is not None: src[mask_zero] = 0
        if mask_one  is not None: src[mask_one ] = 255
        mode = 'L'
    elif data.dtype == torch.bool:
        src = t.cpu()
        mode = 'L'
    else:
        m = data.max().item()
        if data.min().item() < 0 or m > 65535:
            t = data - data.min()
            m = data.max().item()
            if m > 0: t = t*255 / m
            src = t.to('cpu', dtype=torch.uint8)
            mode = 'L'
        else:
            p = _palette(m+1)
            src = p[data.to('cpu', torch.long)]
            mode = 'RGB'
            
    arr = src.numpy()
    img = _img_from_array(arr, mode)

    path = 'out_images/' + imgdump.start_time
    os.makedirs(path, exist_ok=True)
    nname = '_' + name if name is not None else ''
    fname = f'{path}/{it:03d}{nname}_{",".join(map(str, data.shape))}.png'
    img.save(fname)

imgdump.counter = 0
imgdump.start_time = datetime.datetime.now().isoformat().replace('T', '_').replace(':', '-')[:19]

class Dummy:
    @staticmethod
    def imgdump(*x, **y): pass

if __name__ == '__main__':
    #imgdump(torch.randn(17), 'a')
    #imgdump(torch.randn(5, 17), 'b')
    #imgdump(torch.randn(13, 5, 17), 'c')
    #imgdump(torch.randn(7, 3, 5, 17), 'd')
    imgdump(torch.arange(17, dtype=torch.int))
