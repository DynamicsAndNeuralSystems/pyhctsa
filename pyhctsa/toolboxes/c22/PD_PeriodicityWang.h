/*
 * Part of catch22 (https://github.com/DynamicsAndNeuralSystems/catch22), Copyright (C) 2018-
 * Carl H. Lubba, Ben D. Fulcher and contributors, released under the GNU General Public License
 * v3.0 (see LICENSE and NOTICE in this directory). Any "All rights reserved" line below is Xcode
 * template text and does not restrict that grant. Modified for pyhctsa; distributed under
 * GPL-3.0-or-later.
 */
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
