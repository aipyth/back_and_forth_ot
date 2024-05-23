#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    int* indices;
    int hullCount;
} convex_hull;

convex_hull* convex_hull_create(int n) {
    convex_hull* hull = (convex_hull*)malloc(sizeof(convex_hull));
    hull->indices = (int*)malloc(n * sizeof(int));
    hull->hullCount = 0;
    return hull;
}

void convex_hull_destroy(convex_hull* hull) {
    free(hull->indices);
    free(hull);
}

typedef struct {
    int n1;
    int n2;
    double totalMass;
    double *xMap;
    double *yMap;
    double *rho;
    int *argmin;
    double *temp;
    convex_hull* hull;
} BFM;

BFM* BFM_create(int n1, int n2, double* mu) {
    BFM* bfm = (BFM*)malloc(sizeof(BFM));
    bfm->n1 = n1;
    bfm->n2 = n2;

    int n = fmax(n1, n2);
    bfm->hull = convex_hull_create(n);
    bfm->argmin = (int*)malloc(n * sizeof(int));
    bfm->temp = (double*)malloc(n1 * n2 * sizeof(double));

    bfm->xMap = (double*)malloc((n1 + 1) * (n2 + 1) * sizeof(double));
    bfm->yMap = (double*)malloc((n1 + 1) * (n2 + 1) * sizeof(double));

    for (int i = 0; i <= n2; i++) {
        for (int j = 0; j <= n1; j++) {
            double x = j / (n1 * 1.0);
            double y = i / (n2 * 1.0);
            bfm->xMap[i * (n1 + 1) + j] = x;
            bfm->yMap[i * (n1 + 1) + j] = y;
        }
    }

    bfm->rho = (double*)malloc(n1 * n2 * sizeof(double));
    memcpy(bfm->rho, mu, n1 * n2 * sizeof(double));

    bfm->totalMass = 0;
    for (int i = 0; i < n1 * n2; ++i) {
        bfm->totalMass += mu[i];
    }
    bfm->totalMass /= n1 * n2;

    return bfm;
}

void BFM_destroy(BFM* bfm) {
    free(bfm->xMap);
    free(bfm->yMap);
    free(bfm->rho);
    free(bfm->argmin);
    free(bfm->temp);
    convex_hull_destroy(bfm->hull);
    free(bfm);
}

void compute_2d_dual_inside(BFM* bfm, double* dual, double* u) {
    int pcount = bfm->n1 * bfm->n2;
    int n = fmax(bfm->n1, bfm->n2);
    memcpy(bfm->temp, u, pcount * sizeof(double));

    for (int i = 0; i < bfm->n2; i++) {
        compute_dual(&dual[i * bfm->n1], &bfm->temp[i * bfm->n1], bfm->argmin, bfm->hull, bfm->n1);
    }

    transpose_doubles(bfm->temp, dual, bfm->n1, bfm->n2);
    for (int i = 0; i < bfm->n1 * bfm->n2; i++) {
        dual[i] = -bfm->temp[i];
    }

    for (int j = 0; j < bfm->n1; j++) {
        compute_dual(&bfm->temp[j * bfm->n2], &dual[j * bfm->n2], bfm->argmin, bfm->hull, bfm->n2);
    }

    transpose_doubles(dual, bfm->temp, bfm->n2, bfm->n1);
}

void ctransform(BFM* bfm, double* dual, double* phi) {
    compute_2d_dual_inside(bfm, dual, phi);
}

void pushforward(BFM* bfm, double* rho, double* phi, double* nu) {
    calc_pushforward_map(bfm, phi);
    sampling_pushforward(bfm, nu);
    memcpy(rho, bfm->rho, bfm->n1 * bfm->n2 * sizeof(double));
}

double compute_w2(BFM* bfm, double* phi, double* dual, double* mu, double* nu) {
    int pcount = bfm->n1 * bfm->n2;
    double value = 0;

    for (int i = 0; i < bfm->n2; i++) {
        for (int j = 0; j < bfm->n1; j++) {
            double x = (j + .5) / (bfm->n1 * 1.0);
            double y = (i + .5) / (bfm->n2 * 1.0);
            value += .5 * (x * x + y * y) * (mu[i * bfm->n1 + j] + nu[i * bfm->n1 + j]) -
                     nu[i * bfm->n1 + j] * phi[i * bfm->n1 + j] -
                     mu[i * bfm->n1 + j] * dual[i * bfm->n1 + j];
        }
    }

    printf("value calculated");

    value /= pcount;
    return value;
}

void compute_dual(double* dual, double* u, int* dualIndicies, convex_hull* hull, int n) {
    get_convex_hull(u, hull, n);
    compute_dual_indices(dualIndicies, u, hull, n);

    for (int i = 0; i < n; i++) {
        double s = (i + .5) / (n * 1.0);
        int index = dualIndicies[i];
        double x = (index + .5) / (n * 1.0);
        double v1 = s * x - u[dualIndicies[i]];
        double v2 = s * (n - .5) / (n * 1.0) - u[n - 1];
        if (v1 > v2) {
            dual[i] = v1;
        } else {
            dualIndicies[i] = n - 1;
            dual[i] = v2;
        }
    }
}

int sgn(double x) {
    return (x > 0) - (x < 0);
}

void transpose_doubles(double* transpose, double* data, int n1, int n2) {
    for (int i = 0; i < n2; i++) {
        for (int j = 0; j < n1; j++) {
            transpose[j * n2 + i] = data[i * n1 + j];
        }
    }
}

void get_convex_hull(double* u, convex_hull* hull, int n) {
    hull->indices[0] = 0;
    hull->indices[1] = 1;
    hull->hullCount = 2;

    for (int i = 2; i < n; i++) {
        add_point(u, hull, i);
    }
}

void add_point(double* u, convex_hull* hull, int i) {
    if (hull->hullCount < 2) {
        hull->indices[1] = i;
        hull->hullCount++;
    } else {
        int hc = hull->hullCount;
        int ic1 = hull->indices[hc - 1];
        int ic2 = hull->indices[hc - 2];

        double oldSlope = (u[ic1] - u[ic2]) / (ic1 - ic2);
        double slope = (u[i] - u[ic1]) / (i - ic1);

        if (slope >= oldSlope) {
            hull->indices[hc] = i;
            hull->hullCount++;
        } else {
            hull->hullCount--;
            add_point(u, hull, i);
        }
    }
}

double interpolate_function(double* function, double x, double y, int n1, int n2) {
    int xIndex = fmin(fmax(x * n1 - .5, 0), n1 - 1);
    int yIndex = fmin(fmax(y * n2 - .5, 0), n2 - 1);

    double xfrac = x * n1 - xIndex - .5;
    double yfrac = y * n2 - yIndex - .5;

    int xOther = xIndex + sgn(xfrac);
    int yOther = yIndex + sgn(yfrac);

    xOther = fmax(fmin(xOther, n1 - 1), 0);
    yOther = fmax(fmin(yOther, n2 - 1), 0);

    double v1 = (1 - fabs(xfrac)) * (1 - fabs(yfrac)) * function[yIndex * n1 + xIndex];
    double v2 = fabs(xfrac) * (1 - fabs(yfrac)) * function[yIndex * n1 + xOther];
    double v3 = (1 - fabs(xfrac)) * fabs(yfrac) * function[yOther * n1 + xIndex];
    double v4 = fabs(xfrac) * fabs(yfrac) * function[yOther * n1 + xOther];

    double v = v1 + v2 + v3 + v4;

    return v;
}

void compute_dual_indices(int* dualIndicies, double* u, convex_hull* hull, int n) {
    int counter = 1;
    int hc = hull->hullCount;

    for (int i = 0; i < n; i++) {
        double s = (i + .5) / (n * 1.0);
        int ic1 = hull->indices[counter];
        int ic2 = hull->indices[counter - 1];

        double slope = n * (u[ic1] - u[ic2]) / (ic1 - ic2);
        while (s > slope && counter < hc - 1) {
            counter++;
            ic1 = hull->indices[counter];
            ic2 = hull->indices[counter - 1];
            slope = n * (u[ic1] - u[ic2]) / (ic1 - ic2);
        }
        dualIndicies[i] = hull->indices[counter - 1];
    }
}

void calc_pushforward_map(BFM* bfm, double* dual) {
    double xStep = 1.0 / bfm->n1;
    double yStep = 1.0 / bfm->n2;

    for (int i = 0; i <= bfm->n2; i++) {
        for (int j = 0; j <= bfm->n1; j++) {
            double x = j / (bfm->n1 * 1.0);
            double y = i / (bfm->n2 * 1.0);

            double dualxp = interpolate_function(dual, x + xStep, y, bfm->n1, bfm->n2);
            double dualxm = interpolate_function(dual, x - xStep, y, bfm->n1, bfm->n2);

            double dualyp = interpolate_function(dual, x, y + yStep, bfm->n1, bfm->n2);
            double dualym = interpolate_function(dual, x, y - yStep, bfm->n1, bfm->n2);

            bfm->xMap[i * (bfm->n1 + 1) + j] = .5 * bfm->n1 * (dualxp - dualxm);
            bfm->yMap[i * (bfm->n1 + 1) + j] = .5 * bfm->n2 * (dualyp - dualym);
        }
    }
}

void sampling_pushforward(BFM* bfm, double* mu) {
    int pcount = bfm->n1 * bfm->n2;
    memset(bfm->rho, 0, pcount * sizeof(double));

    for (int i = 0; i < bfm->n2; i++) {
        for (int j = 0; j < bfm->n1; j++) {
            double mass = mu[i * bfm->n1 + j];

            if (mass > 0) {
                double xStretch0 = fabs(bfm->xMap[i * (bfm->n1 + 1) + j + 1] - bfm->xMap[i * (bfm->n1 + 1) + j]);
                double xStretch1 = fabs(bfm->xMap[(i + 1) * (bfm->n1 + 1) + j + 1] - bfm->xMap[(i + 1) * (bfm->n1 + 1) + j]);

                double yStretch0 = fabs(bfm->yMap[(i + 1) * (bfm->n1 + 1) + j] - bfm->yMap[i * (bfm->n1 + 1) + j]);
                double yStretch1 = fabs(bfm->yMap[(i + 1) * (bfm->n1 + 1) + j + 1] - bfm->yMap[i * (bfm->n1 + 1) + j + 1]);

                double xStretch = fmax(xStretch0, xStretch1);
                double yStretch = fmax(yStretch0, yStretch1);

                int xSamples = fmax(bfm->n1 * xStretch, 1);
                int ySamples = fmax(bfm->n2 * yStretch, 1);

                double factor = 1 / (xSamples * ySamples * 1.0);

                for (int l = 0; l < ySamples; l++) {
                    for (int k = 0; k < xSamples; k++) {
                        double a = (k + .5) / (xSamples * 1.0);
                        double b = (l + .5) / (ySamples * 1.0);

                        double xPoint = (1 - b) * (1 - a) * bfm->xMap[i * (bfm->n1 + 1) + j] +
                                        (1 - b) * a * bfm->xMap[i * (bfm->n1 + 1) + j + 1] +
                                        b * (1 - a) * bfm->xMap[(i + 1) * (bfm->n1 + 1) + j] +
                                        a * b * bfm->xMap[(i + 1) * (bfm->n1 + 1) + j + 1];
                        double yPoint = (1 - b) * (1 - a) * bfm->yMap[i * (bfm->n1 + 1) + j] +
                                        (1 - b) * a * bfm->yMap[i * (bfm->n1 + 1) + j + 1] +
                                        b * (1 - a) * bfm->yMap[(i + 1) * (bfm->n1 + 1) + j] +
                                        a * b * bfm->yMap[(i + 1) * (bfm->n1 + 1) + j + 1];

                        double X = xPoint * bfm->n1 - .5;
                        double Y = yPoint * bfm->n2 - .5;

                        int xIndex = X;
                        int yIndex = Y;

                        double xFrac = X - xIndex;
                        double yFrac = Y - yIndex;

                        int xOther = xIndex + 1;
                        int yOther = yIndex + 1;

                        xIndex = fmin(fmax(xIndex, 0), bfm->n1 - 1);
                        xOther = fmin(fmax(xOther, 0), bfm->n1 - 1);

                        yIndex = fmin(fmax(yIndex, 0), bfm->n2 - 1);
                        yOther = fmin(fmax(yOther, 0), bfm->n2 - 1);

                        bfm->rho[yIndex * bfm->n1 + xIndex] += (1 - xFrac) * (1 - yFrac) * mass * factor;
                        bfm->rho[yOther * bfm->n1 + xIndex] += (1 - xFrac) * yFrac * mass * factor;
                        bfm->rho[yIndex * bfm->n1 + xOther] += xFrac * (1 - yFrac) * mass * factor;
                        bfm->rho[yOther * bfm->n1 + xOther] += xFrac * yFrac * mass * factor;
                    }
                }
            }
        }
    }

    double sum = 0;
    for (int i = 0; i < pcount; i++) {
        sum += bfm->rho[i] / pcount;
    }
    for (int i = 0; i < pcount; i++) {
        bfm->rho[i] *= bfm->totalMass / sum;
    }
}

int main() {
    // Example usage
    int n1 = 3, n2 = 3;
    double mu[] = {0.2, 0.3, 0.1, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    BFM* bfm = BFM_create(n1, n2, mu);


    printf("x map\n");
    for (int i = 0; i <= bfm->n2; i++) {
        for (int j = 0; j <= bfm->n1; j++) {
            // double x = j / (bfm->n1 * 1.0);
            // double y = i / (bfm->n2 * 1.0);
            // bfm->xMap[i * (bfm->n1 + 1) + j] = x;
            // bfm->yMap[i * (bfm->n1 + 1) + j] = y;
            printf("%f ", bfm->xMap[i * (bfm->n1 + 1) + j]);
        }
        printf("\n");
    }

    printf("y map\n");
    for (int i = 0; i <= bfm->n2; i++) {
        for (int j = 0; j <= bfm->n1; j++) {
            // double x = j / (bfm->n1 * 1.0);
            // double y = i / (bfm->n2 * 1.0);
            // bfm->xMap[i * (bfm->n1 + 1) + j] = x;
            // bfm->yMap[i * (bfm->n1 + 1) + j] = y;
            printf("%f ", bfm->yMap[i * (bfm->n1 + 1) + j]);
        }
        printf("\n");
    }

    // Define the phi and dual arrays
    double phi[] = {0., 0.25, 1.};
    double dual[n1 * n2];

    // Call ctransform
    ctransform(bfm, dual, phi);

    // Print the result
    printf("Dual array after ctransform:\n");
    for (int i = 0; i < n2; i++) {
        for (int j = 0; j < n1; j++) {
            printf("%f ", dual[i * n1 + j]);
        }
        printf("\n");
    }

    BFM_destroy(bfm);
    return 0;
}
