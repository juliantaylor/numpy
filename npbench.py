
import timeit

def time(fun, dtype, size=10000, number=1000):
    t = timeit.timeit(setup="import numpy as np; d = np.ones((%s,), dtype=%s)" % (size, dtype),
                      stmt=fun, number=number)
    n = "%s %s#" % (dtype, fun)
    print "%s %s" % (n, ("%0.3g" % t).rjust(40 - len(n)))

if __name__ == "__main__":
    # non ieee compliant version in numpy outperforms vectorized one
    #time("np.multiply(d, d)", "np.complex64")
    #time("np.multiply(d, d)", "np.complex128")

    #last won't fit in L3 cache of most cpus
    for s in [5, 10000, 500000, 1000000]:
        print '%.1f, %.1f' % (s/1024. * 4, s/1024. * 8)
        time("d.byteswap()", "np.int16", s)
        for dt in ["np.float32", "np.float64"]:
            time("np.max(d)", dt, s)
            time("np.min(d)", dt, s)
            time("np.sum(d)", dt, s)
            time("np.prod(d)", dt, s)
            time("np.add(1, d)", dt, s)
            time("np.add(d, 1)", dt, s)
            time("np.add(d, 1, out=d)", dt, s)
            time("np.add(d, d, out=d)", dt, s)
            time("np.divide(d, 1)", dt, s)
            time("np.divide(d, d)", dt, s)
            time("np.add(d, d)", dt, s)
            time("np.multiply(d, d)", dt, s)
            time("np.sqrt(d)", dt, s)
            time("np.abs(d)", dt, s)
            time("np.abs(d, out=d)", dt, s)
