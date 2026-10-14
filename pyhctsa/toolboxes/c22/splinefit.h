/*
 * Part of catch22 (https://github.com/DynamicsAndNeuralSystems/catch22), Copyright (C) 2018-
 * Carl H. Lubba, Ben D. Fulcher and contributors, released under the GNU General Public License
 * v3.0 (see LICENSE and NOTICE in this directory). Any "All rights reserved" line below is Xcode
 * template text and does not restrict that grant. Modified for pyhctsa; distributed under
 * GPL-3.0-or-later.
 * splinefit is based on SPLINEFIT, Copyright (c) 2010 Jonas Lundgren (BSD-2-Clause; see NOTICE).
 */
//
//  splinefit.h
//  C_polished
//
//  Created by Carl Henning Lubba on 27/09/2018.
//  Copyright © 2018 Carl Henning Lubba. All rights reserved.
//

#ifndef splinefit_h
#define splinefit_h

#include <stdio.h>

extern int splinefit(const double *y, const int size, double *yOut);

#endif /* splinefit_h */
