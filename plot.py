import itertools
import numpy as np
import math
from scipy.special import legendre

class TDFDependent:
    def __init__(self, parts, coefs):
        self.parts = parts
        self.coefs = coefs
        self.dims = coefs.shape

    def __call__(self, xs):
        basis = []
        for n, dim in enumerate(self.dims):
            basis.append([1] * dim)
            for i in range(dim):
                basis[-1][i] = self.parts[n][i](xs[n])

        ret = 0
        for t in itertools.product(*map(range, self.dims)):
            p = self.coefs[t]
            for n, i in enumerate(t):
                p *= basis[n][i]
            ret += p
        return ret

    def linspace(self, n, o, default):
        basis = []
        for m, dim in enumerate(self.dims):
            basis.append([1] * dim)
            if m != o:
                for i in range(dim):
                    basis[-1][i] = self.parts[m][i](default[m])
            else:
                for i in range(dim):
                    basis[-1][i] = self.parts[m][i].linspace(n)[1]

        ret = 0
        for t in itertools.product(*map(range, self.dims)):
            p = self.coefs[t]
            for m, i in enumerate(t):
                p *= basis[m][i]
            ret += p
        return ret


class GraphDrawer:
    def __init__(self, data, fDegree = None, epsilon=0.01):
        self.data = data
        if fDegree == None: fDegree = (1, ) * (self.data[0].size - 1)
        if np.prod(fDegree) != self.data[0].size: raise Exception("data is not matched with fDegree")
        self.fDegree = fDegree
        self.nDims = len(fDegree)
        self.nTopics = len(self.data)
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                       (200, 200, 0), (0, 200, 200), (200, 0, 200),
                       (200, 200, 200), (130, 130, 130), (60, 60, 60),
                       (255, 100, 100), (100, 255, 100), (100, 100, 255),
                       (240, 240, 80), (80, 240, 240), (240, 80, 240),
                       (255, 255, 255), (180, 0, 0), (0, 180, 0),
                       (0, 0, 180), (100, 100, 100)]
        self.cacheTopicDenseMap = None
        self.cacheAxisDefault = None
        self.grayTextureMap = None
        self.epsilon = epsilon
        self.tdf = []
        for d in self.data:
            basisParts = []
            for i, f in enumerate(fDegree):
                basisParts.append([])
                for dim in range(f):
                    basisParts[-1].append(np.polynomial.Legendre([0]*dim + [1], [0, 1]))
            coefs = d.reshape(fDegree[::-1]).transpose()
            self.tdf.append(TDFDependent(basisParts, coefs))

    def getLinspaces(self, n, o, x):
        return np.array(list(map(lambda tdf : tdf.linspace(n, o, x), self.tdf)))

    def prepareTopicDesneMap(self, width, height, axis, default = None):
        if not default: default = [0] * self.nDims
        if (not self.cacheTopicDenseMap is None) and self.cacheTopicDenseMap.shape == (width, height, self.nTopics) and self.cacheAxisDefault == (axis, default): return self.cacheTopicDenseMap
        self.cacheTopicDenseMap = np.zeros((width, height, self.nTopics))
        self.cacheAxisDefault = axis, default
        x = list(default)
        for j in range(height):
            x[axis[1]] = 1 - j / (height - 1)
            p = np.exp(self.getLinspaces(width, axis[0], x)) + self.epsilon
            p /= np.sum(p, axis=0)
            x[axis[0]] = np.linspace(0, 1, width)
            for i in range(width):
                self.cacheTopicDenseMap[i, j] = p[:, i]
        return self.cacheTopicDenseMap

    def findEdge(self, ms):
        edgeMap = np.zeros(ms.shape)
        for (i, j), v in np.ndenumerate(ms):
            i0 = max(i - 1, 0)
            j0 = max(j - 1, 0)
            i1 = min(i + 1, ms.shape[0] - 1)
            j1 = min(j + 1, ms.shape[1] - 1)
            n1 = [ms[i0, j], ms[i, j0], ms[i, j1], ms[i1, j]].count(.0)
            n2 = [ms[i0, j0], ms[i0, j1], ms[i1, j0], ms[i1, j1]].count(.0)
            if v: n1, n2 = 4 - n1, 4 - n2
            if n1 + n2 > 6: continue
            #edgeMap[i, j] = min(((4 - n1) + (4 - n2) * 0.5) / 2.5, 1)
            edgeMap[i, j] = 1
        return edgeMap

    def realToColorRange(self, r):
        r *= 6
        if r < 1: return 0, 0, int(r*255)
        if r < 2: return 0, int((r - 1)*255), 255
        if r < 3: return 0, 255, int((3-r)*255)
        if r < 4: return int((r-3)*255), 255, 0
        if r < 5: return 255, int((5-r)*255), 0
        if r < 6: return 255, int((r-5)*255), int((r-5)*255)
        return 255, 255, 255

    def prepareGrayTexture(self, scale):
        nPixel = 16 * scale
        if not self.grayTextureMap is None and self.grayTextureMap[0].shape == (nPixel, nPixel, 8): return self.grayTextureMap
        self.grayTextureMap = np.zeros((nPixel, nPixel, 8))

        for i in range(int(nPixel/2) - scale, int(nPixel/2) + scale):
            for j in range(0, scale):
                self.grayTextureMap[i, j, (0, 6)] += 1
                self.grayTextureMap[i, -j - 1, (0, 6)] += 1
                self.grayTextureMap[j, i, (0, 6)] += 1
                self.grayTextureMap[-j - 1, i, (0, 6)] += 1

        for i in range(nPixel):
            self.grayTextureMap[i, i, (1, 4, 5, 6)] += 1
        for j in range(1, scale):
            for i in range(nPixel):
                self.grayTextureMap[i - j, i, (1, 4, 5, 6)] += 1
                self.grayTextureMap[i, i - j, (1, 4, 5, 6)] += 1
        j = scale
        for i in range(nPixel):
            self.grayTextureMap[i - j, i, (1, 4, 5, 6)] += 0.5
            self.grayTextureMap[i, i - j, (1, 4, 5, 6)] += 0.5

        for j in range(0, scale):
            for i in range(nPixel):
                self.grayTextureMap[i, int(nPixel / 2)  - 1 - j, (2, 5, 6)] += 1
                self.grayTextureMap[i, int(nPixel / 2) + j, (2, 5, 6)] += 1

        for i in range(nPixel):
            self.grayTextureMap[i, nPixel - 1 - i, (3, 4, 5, 6)] += 1
        for j in range(1, scale):
            for i in range(nPixel):
                self.grayTextureMap[i - j, nPixel - 1 - i, (3, 4, 5, 6)] += 1
                self.grayTextureMap[i, nPixel - 1 - i - j, (3, 4, 5, 6)] += 1
        j = scale
        for i in range(nPixel):
            self.grayTextureMap[i - j, nPixel - 1 - i, (3, 4, 5, 6)] += 0.5
            self.grayTextureMap[i, nPixel - 1 - i - j, (3, 4, 5, 6)] += 0.5

        self.grayTextureMap[:, :, 7] += 1

        np.clip(self.grayTextureMap, 0, 1, out=self.grayTextureMap)
        return self.grayTextureMap

    def realToGrayTexture(self, r, x, y):
        r *= 8
        x %= self.grayTextureMap.shape[0]
        y %= self.grayTextureMap.shape[1]
        if r < 1: return self.grayTextureMap[x, y, 0] * r
        if r < 2: return self.grayTextureMap[x, y, 0] + (self.grayTextureMap[x, y, 1] - self.grayTextureMap[x, y, 0]) * (r - 1)
        if r < 3: return self.grayTextureMap[x, y, 1] + (self.grayTextureMap[x, y, 2] - self.grayTextureMap[x, y, 1]) * (r - 2)
        if r < 4: return self.grayTextureMap[x, y, 2] + (self.grayTextureMap[x, y, 3] - self.grayTextureMap[x, y, 2]) * (r - 3)
        if r < 5: return self.grayTextureMap[x, y, 3] + (self.grayTextureMap[x, y, 4] - self.grayTextureMap[x, y, 3]) * (r - 4)
        if r < 6: return self.grayTextureMap[x, y, 4] + (self.grayTextureMap[x, y, 5] - self.grayTextureMap[x, y, 4]) * (r - 5)
        if r < 7: return self.grayTextureMap[x, y, 5] + (self.grayTextureMap[x, y, 6] - self.grayTextureMap[x, y, 5]) * (r - 6)
        if r < 8: return self.grayTextureMap[x, y, 6] + (self.grayTextureMap[x, y, 7] - self.grayTextureMap[x, y, 6]) * (r - 7)
        return self.grayTextureMap[x, y, 7]

    def drawTopicContour(self, targetTopic, step, width, height, axis, default = None, gray = True):
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(img)
        ms = self.prepareTopicDesneMap(width, height, axis, default)[:,:,targetTopic]
        if gray: self.prepareGrayTexture(1)

        mbm = np.zeros((width, height))
        edgeMap = np.zeros((width, height))
        t = [(v, (i, j)) for (i, j), v in np.ndenumerate(ms)]
        t.sort()
        n = 0
        for cStep in range(1, int(1/step) + 1):
            if n >= len(t): break
            for v, (i, j) in t[n:]:
                if v >= cStep * step: break
                mbm[i, j] = 1
                n += 1
            if cStep % 2:
                em = self.findEdge(mbm)
                if width > 1280:
                    edgeMap += np.clip((em + np.lib.pad(em[1:, :], ((0, 1), (0, 0)), 'constant') + np.lib.pad(em[:, :-1], ((0, 0), (1, 0)), 'constant')), 0, 1) * 0.7
                elif width > 768:
                    edgeMap += np.clip((em + np.lib.pad(em[1:, :], ((0, 1), (0, 0)), 'constant')), 0, 1) * 0.7
                else:
                    edgeMap += em * 0.7
            elif cStep % 10:
                em = self.findEdge(mbm)
                if width > 1280:
                    edgeMap += np.clip((em + np.lib.pad(em[1:, :], ((0, 1), (0, 0)), 'constant') + np.lib.pad(em[:, :-1], ((0, 0), (1, 0)), 'constant')), 0, 1)
                elif width > 768:
                    edgeMap += np.clip((em + np.lib.pad(em[1:, :], ((0, 1), (0, 0)), 'constant')), 0, 1)
                else:
                    edgeMap += em
            else:
                em = self.findEdge(mbm)
                if width > 1280:
                    edgeMap += np.clip((em + np.lib.pad(em[1:, :], ((0, 1), (0, 0)), 'constant') +
                                        np.lib.pad(em[2:, :], ((0, 2), (0, 0)), 'constant') +
                                        np.lib.pad(em[3:, :], ((0, 3), (0, 0)), 'constant') +
                                        np.lib.pad(em[:, :-1], ((0, 0), (1, 0)), 'constant') +
                                        np.lib.pad(em[:, :-2], ((0, 0), (2, 0)), 'constant')), 0, 1)
                elif width > 768:
                    edgeMap += np.clip((em + np.lib.pad(em[1:, :], ((0, 1), (0, 0)), 'constant') +
                                        np.lib.pad(em[:, :-1], ((0, 0), (1, 0)), 'constant')), 0, 1)
                else:
                    edgeMap += np.clip((em + np.lib.pad(em[1:, :], ((0, 1), (0, 0)), 'constant')), 0, 1)
        for i, j in itertools.product(range(width), range(height)):
            cStep = int(ms[i, j] / step)
            v = 1-math.exp(-2 * ms[i, j] / step / 4)
            if gray:
                p = (self.realToGrayTexture(v, i, j) * 0.4, ) * 3
            else:
                p = (1-i/255. for i in self.realToColorRange(v))
            p = tuple(edgeMap[i, j] or pn for pn in p)
            draw.point((i, j), tuple(int(255*(1 - pn)) for pn in p))
        return img

    @staticmethod
    def hue(h):
        h %= 1
        h *= 6
        if h < 1: return (255, int(255 * h), 0)
        if h < 2: return (int(255 * (2 - h)), 255, 0)
        if h < 3: return (0, 255, int(255 * (h - 2)))
        if h < 4: return (0, int(255 * (4 - h)), 255)
        if h < 5: return (int(255 * (h - 4)), 0, 255)
        return (255, 0, int(255 * (6 - h)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='input text file containing lambda parameters')
    parser.add_argument('output', help='directory of output images')
    parser.add_argument('-d', '--dim', required=True, help='the order of Legendre Polynomials, ex) 4,3')
    parser.add_argument('-x', '--xaxis', type=int, default=0, help='')
    parser.add_argument('-y', '--yaxis', type=int, default=1, help='')
    parser.add_argument('--width', type=int, default=400)
    parser.add_argument('--height', type=int, default=300)

    args = parser.parse_args()
    try:
        dims = [int(i) + 1 for i in args.dim.split(',')]
        if not dims: raise RuntimeError()
    except:
        print('--dim argument must be comma-separated integer list. ex) --dim=4,3 or --dim=2,4,3')
        exit(-1)

    params = []
    found = False
    for line in open(args.input):
        line = line.strip()
        if line == '== Parameters ==': 
            found = True
            params.clear()
        elif found:
            line = line.strip(',')
            if line: params.append(np.array([float(f) for f in line.split(',')]))
            else: found = False
    
    print('Legendre Polynomial Dimensions:', ', '.join('{}'.format(d - 1) for d in dims))
    print('The input file "{}" has {} topics x {} lambda parameters'.format(args.input, len(params), len(params[0])))
    dsize = 1
    for d in dims: dsize *= d
    if dsize != len(params[0]):
        print('Lambda parameter size {} is not matched with dimension of Legendre Polynomials {}'.format(len(params[0]), ' x '.join('({} + 1)'.format(d - 1) for d in dims)))
        exit(-1)

    gd = GraphDrawer(params, dims, epsilon=1e-10)
    for n in range(gd.nTopics):
        print("Draw contour {}".format(n))
        img = gd.drawTopicContour(n, 0.01, args.width, args.height, (args.xaxis, args.yaxis), gray=False)
        with open('{}/{}.png'.format(args.output, n), 'wb') as f:
            img.save(f, 'PNG')

