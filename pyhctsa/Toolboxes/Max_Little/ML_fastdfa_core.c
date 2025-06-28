/* Performs fast detrended fluctuation analysis on a nonstationary input signal.

   Useage:
   Inputs
    x          - input signal: must be a row vector
   Optional inputs:
    intervals  - List of sample interval widths at each scale
                 (If not specified, then a binary subdivision is constructed)

   Outputs:
    intervals  - List of sample interval widths at each scale
    flucts     - List of fluctuation amplitudes at each scale

   (c) 2006 Max Little. If you use this code, please cite:
   M. Little, P. McSharry, I. Moroz, S. Roberts (2006),
   Nonlinear, biophysically-informed speech pathology detection
   in Proceedings of ICASSP 2006, IEEE Publishers: Toulouse, France.
*/

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define REAL double

/* Calculate accumulated sum signal */
static void cumulativeSum(
   unsigned long elements,
   REAL *x,
   REAL *y
)
{
   unsigned int i;
   REAL accum = 0.0f;
   for (i = 0; i < elements; i++)
   {
      accum += x[i];
      y[i] = accum;
   }
}

/* Calculate intervals if not specified */
static void calculateIntervals(
   unsigned long elements,
   unsigned long *N_scales,
   unsigned long **intervals
)
{
   unsigned long scales, subdivs;
   REAL idx_inc;
   long scale;

   scales = (unsigned long)(log10(elements) / log10(2.0));
   if (((REAL)(1 << (scales - 1))) > ((REAL)elements / 2.5f))
   {
      scales--;
   }
   *N_scales = scales;
   *intervals = (unsigned long *)calloc(scales, sizeof(unsigned long));
   for (scale = scales - 1; scale >= 0; scale--)
   {
      subdivs = 1 << scale;
      idx_inc = (REAL)elements / (REAL)subdivs;
      (*intervals)[scale] = (unsigned long)(idx_inc + 0.5f);
   }
}

/* Measure the fluctuations at each scale */
static void dfa(
   REAL *x,
   unsigned long elements,
   unsigned long *intervals,
   REAL *flucts,
   unsigned long N_scales
)
{
   unsigned long idx, i, start, end, iwidth, accum_idx;
   long scale;

   REAL Sy, Sxy;                   /* y and x-y components of normal equations */
   REAL Sx, Sxx;                   /* x-component of normal equations */
   REAL a, b;                      /* Straight-line fit parameters */
   REAL *trend;                    /* Trend vector */
   REAL diff, accum, delta;

   trend = (REAL *)calloc(elements, sizeof(REAL));

   for (scale = N_scales - 1; scale >= 0; scale--)
   {
      for (accum_idx = 0, idx = 0; idx < elements; idx += intervals[scale], accum_idx++)
      {
         start = idx;
         end = idx + intervals[scale] - 1;

         if (end >= elements)
         {
            for (i = start; i < elements; i++)
            {
               trend[i] = x[i];
            }
            break;
         }
         iwidth = end - start + 1;

         Sy = 0.0f;
         Sxy = 0.0f;
         for (i = start; i <= end; i++)
         {
            Sy += x[i];
            Sxy += x[i] * (REAL)i;
         }

         Sx = ((REAL)end + (REAL)start) * (REAL)iwidth / 2.0;
         Sxx = (REAL)iwidth * (2 * (REAL)end * (REAL)end + 2 * (REAL)start * (REAL)start +
                               2 * (REAL)start * (REAL)end + (REAL)end - (REAL)start) / 6.0;
         delta = (REAL)iwidth * Sxx - (Sx * Sx);

         b = (Sy * Sxx - Sx * Sxy) / delta;
         a = ((REAL)iwidth * Sxy - Sx * Sy) / delta;

         for (i = start; i <= end; i++)
         {
            trend[i] = a * (REAL)i + b;
         }
      }

      accum = 0.0f;
      for (i = 0; i < elements; i++)
      {
         diff = x[i] - trend[i];
         accum += diff * diff;
      }
      flucts[scale] = sqrt(accum / (REAL)elements);
   }

   free(trend);
}

/* Main C-callable entry point */
void fastdfa_core(
    const double *x,
    unsigned long elements,
    unsigned long **intervals, // pointer-to-pointer: will be allocated and filled
    double *flucts,
    unsigned long *N_scales
)
{
    double *y_in;
    unsigned long *intervals_local = NULL;
    unsigned long n_scales_local, i;

    y_in = (double *)calloc(elements, sizeof(double));
    if (!y_in) return;
    cumulativeSum(elements, (double *)x, y_in);

    if (*intervals == NULL) {
        calculateIntervals(elements, &n_scales_local, &intervals_local);
        *N_scales = n_scales_local;
        *intervals = (unsigned long *)calloc(n_scales_local, sizeof(unsigned long));
        if (!*intervals) {
            free(intervals_local);
            free(y_in);
            return;
        }
        for (i = 0; i < n_scales_local; i++) {
            (*intervals)[i] = intervals_local[i];
        }
        free(intervals_local);
    } else {
        n_scales_local = *N_scales;
    }

    for (i = 0; i < n_scales_local; i++) {
        if (((*intervals)[i] > elements) || ((*intervals)[i] < 3)) {
            free(y_in);
            return;
        }
    }

    dfa(y_in, elements, *intervals, flucts, n_scales_local);

    free(y_in);
}