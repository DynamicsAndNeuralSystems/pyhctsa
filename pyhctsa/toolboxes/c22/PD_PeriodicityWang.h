//
//  PD_PeriodicityWang.h
//  C_polished
//
//  Created by Carl Henning Lubba on 28/09/2018.
//  Copyright © 2018 Carl Henning Lubba. All rights reserved.
//

#ifndef PD_PeriodicityWang_h
#define PD_PeriodicityWang_h

typedef struct {
    int th1;  // threshold 0
    int th2;  // threshold 0.01
    int th3;  // threshold 0.1
    int th4;  // threshold 0.2
    int th5;  // threshold 1/sqrt(N)
    int th6;  // threshold 5/sqrt(N)
    int th7;  // threshold 10/sqrt(N)
} PD_PeriodicityWang_Results;

PD_PeriodicityWang_Results PD_PeriodicityWang(
    const double *y,
    int size
);

#endif /* PD_PeriodicityWang_h */
