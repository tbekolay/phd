import os

import svgutils.transform as sg


class RectElement(sg.FigureElement):
    def __init__(self, x, y):
        s = 18
        rect = sg.etree.Element(sg.SVG+"rect",
                                {"x": str(x), "y": str(y - s),
                                 "width": str(s), "height": str(s),
                                 "style": "fill:white;"})
        sg.FigureElement.__init__(self, rect)


def el(char, path, x, y, scale=1, offset=(4, 24)):
    toret = []
    if char is not None:
        toret.append(RectElement(x + offset[0], y + offset[1]))
        toret.append(sg.TextElement(x + offset[0],
                                    y + offset[1],
                                    char,
                                    size=24,
                                    weight='bold',
                                    font='Arial'))
    if path is not None and path.endswith(".svg"):
        svg = sg.fromfile(path)
        svg = svg.getroot()
        svg.moveto(str(x), str(y), scale)
        toret = [svg] + toret
    return toret

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def svgpath(name, pdir="plots", subdir="results"):
    return os.path.join(root, pdir, subdir, '%s.svg' % name)


def svgfig(w, h):
    w = str(w)
    h = str(h)
    return sg.SVGFigure(w, h)


def savefig(fig, name, subdir="results"):
    path = svgpath(name, pdir="figures", subdir=subdir)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    fig.save(path)

in2px = 72


# #####################################
# Model 1: Neural cepstral coefficients
# #####################################

def temp_scaling():
    w = 6 * in2px
    h = 4 * in2px

    fig = svgfig(w * 2, h * 2)
    fig.append(el("A", svgpath('ncc-mfcc'), 0, 0))
    fig.append(el("B", svgpath('ncc-mfcc-long'), w, 0))
    fig.append(el("C", svgpath('ncc-ncc'), 0, h))
    fig.append(el("D", svgpath('ncc-ncc-short'), w, h))
    savefig(fig, 'temp-scaling')


def filter(filt):
    w = 4 * in2px
    h = 8 * in2px

    fig = svgfig(w * 3, h)
    fig.append(el("A", svgpath('%s-noise' % filt, subdir='methods'), 0, 0))
    fig.append(el("B", svgpath('%s-ramp' % filt, subdir='methods'), w, 0))
    fig.append(
        el("C", svgpath('%s-speech' % filt, subdir='methods'), w * 2, 0))
    savefig(fig, filt, subdir='methods')


def ncc_zscore():
    w = 5 * in2px
    h = 3.5 * in2px

    fig = svgfig(w * 2, h)
    fig.append(el("A", svgpath('ncc-zscore-acc-v'), 0, 0))
    fig.append(el("B", svgpath('ncc-zscore-acc-b'), w, 0))
    savefig(fig, 'ncc-zscore')


def ncc_phones():
    w = 5 * in2px
    h = 3.5 * in2px

    fig = svgfig(w * 2, h)
    fig.append(el("A", svgpath('ncc-phones-acc-b'), 0, 0))
    fig.append(el("B", svgpath('ncc-phones-racc-b'), w, 0))
    savefig(fig, 'ncc-phones')


# ############################
# Model 2: Syllable production
# ############################



# #############################
# Model 3: Syllable recognition
# #############################
