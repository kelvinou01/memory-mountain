#define MAXELEMS (1 << 26) /* 64M bytes / 8 = 8M doubles */

#include <mach/mach_time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

double data[MAXELEMS];

/* simple timer wrapper */
static double seconds(void (*f)(int, int), int elems, int stride)
{
    uint64_t t0 = mach_absolute_time();
    f(elems, stride);
    uint64_t t1 = mach_absolute_time();

    mach_timebase_info_data_t info;
    mach_timebase_info(&info);

    double ns = (double)(t1 - t0) * info.numer / info.denom;
    return ns * 1e-9;
}

void test(int elems, int stride)
{
    double result = 0.0;
    volatile double sink;
    for (int i = 0; i < elems; i += stride)
        result += data[i];
    sink = result;
}

double run(int size_bytes, int stride)
{
    int elems = size_bytes / sizeof(double);
    if (elems > MAXELEMS)
        return -1;

    test(elems, stride); /* warm up */
    double t = seconds(test, elems, stride);
    if (t == 0)
        return -1;

    long reads = elems / stride;
    double mb = (reads * 8.0) / (1024.0 * 1024.0);
    return mb / t; /* MB/s */
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "usage: %s <size_bytes> <stride_elems>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);
    int stride = atoi(argv[2]);
    double thr = run(size, stride);

    if (thr < 0)
        return 1;
    printf("%f\n", thr);
    return 0;
}
