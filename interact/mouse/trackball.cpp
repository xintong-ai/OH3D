//
//     _____                   __  
//    / ___/__________  __  __/ /_ 
//    \__ \/ ____/ __ \/ / /   __/
//   ___/   /___/ /_/ / /_/ / /__ 
//  /____/\____/\____/\____/\___/
//
//       Visualization Team 
//     Advanced Computing Lab 
//  Los Alamos National Laboratory
// --------------------------------
// 
// $Id: Trackball.cpp,v 1.4 2005/09/02 09:49:30 groth Exp $
// $Date: 2005/09/02 09:49:30 $
// $Revision: 1.4 $
// Authors: A. McPherson, P. McCormick, J. Kniss, G. Roth
//
// *** Notes *** 
// 
//  Initially part of the ACL Visualization Library moved into 
//  scout to clean things up some and remove external 
//  dependencies. 
// 
//  Hacked into C++ from SGI's trackball.h---see copyright
//  at end.
// ============================================================
//

#include <string.h>
#include <limits.h>
#include <cmath>

#include "trackball.h" // Class definition
#include "Rotation.h"  // For member operations


//===========================================================================
//===========================================================================
// Static Initialization
//===========================================================================
//===========================================================================

//  This size should really be based on the distance from the center of
//  rotation to the point on the object underneath the mouse.  That
//  point would then track the mouse as closely as possible.  This is a
//  simple example, though, so that is left as an Exercise for the
//  Programmer.
float  Trackball::m_trackball_size = 0.9;
//float  Trackball::m_trackball_size = 1;

//===========================================================================
//===========================================================================
// Constructors/Destructor
//===========================================================================
//===========================================================================

Trackball::Trackball()
{
  //  Empty.

} // Trackball()


Trackball::~Trackball()
{
  //  Empty.

} // ~Trackball()

// ----------------------------------------------------------------------------
//!
//!  Ok, simulate a track-ball.  Project the points onto the virtual
//!  trackball, then figure out the axis of rotation, which is the cross
//!  product of P1 P2 and O P1 (O is the center of the ball, 0,0,0)
//!  Note:  This is a deformed trackball-- is a trackball in the center,
//!  but is deformed into a hyperbolic sheet of rotation away from the
//!  center.  This particular function was chosen after trying out
//!  several variations.
//!
//!  It is assumed that the arguments to this routine are in the range
//!  (-1.0 ... 1.0)
//!
const Rotation &Trackball::rotate(float fromX, float fromY, float toX, float toY)
{
  QuatVector axis;           //!< Axis of rotation
  QuatVector fromVec, toVec; //!< Destination and origin vectors

  //!  If no rotation, quit early
  if (fromX == toX && fromY == toY)
    {
      //! Zero rotation
      m_result.clear();
      return m_result;
    }

  //! First, figure out z-coordinates for projection of P1 and P2 to
  //! deformed sphere
  fromVec.set(fromX, fromY, project_to_sphere(m_trackball_size, fromX, fromY));
  toVec.set  (toX,   toY,   project_to_sphere(m_trackball_size, toX,   toY));

  //fromVec.set(0.0, fromY, project_to_sphere(m_trackball_size, fromX, fromY));
  //toVec.set(0.0, toY, project_to_sphere(m_trackball_size, toX, toY));

  return rotate_vectors(fromVec, toVec);

} // rotate()

// Differs from rotate() only in that the vector about which rotation occurs
// Is restricted to being coincident with the z-axis. Thus the 2D image can
// be rotated parallel to the screen.
// Much of the work could likely be simplified. Particularly the projection
// to a sphere, which could now be projected to a circle.
const Rotation &Trackball::spin(float fromX, float fromY, float toX, float toY)
{
  QuatVector axis;           //!< Axis of rotation
  QuatVector fromVec, toVec; //!< Destination and origin vectors

  //! Vectors are flat against plane parallel to screen
  fromVec.set(fromX, fromY, 0.0);
  toVec.set  (toX,   toY,   0.0);

  //! In degenerate case resulting from coincident vectors, quit early
  if(fromVec.coincident(toVec))
    {
      m_result.clear();
      return m_result;
    }

  return rotate_vectors(fromVec, toVec);

} // spin()

//============================================================================
// Protected Member Functions
//============================================================================

//-===========================================================================
// Private Member Functions
//============================================================================


#include <iostream>

//! Assumes error checking is done
const Rotation &Trackball::rotate_vectors(QuatVector &fromVec, QuatVector &toVec)
{
  QuatVector diffVec;//!< Difference vector
  float t;           //!< Position along vector

  //! Axis is the cross product of toVec and fromVec
  axis.cross(fromVec, toVec);

  //! Figure out how much to rotate around that axis.
  diffVec.sub(toVec, fromVec);
  t = diffVec.length() / (2.0 * m_trackball_size);

  //! Clamp out-of-control values...
  if (t > 1.0) t = 1.0;
  if (t < -1.0) t = -1.0;
  angle = 2.0 * asin(t);

  m_result.set(angle, axis);

  return m_result;

} // rotate_vectors()

// ---------------------------------------------------------------------------
//
//  Project an x,y pair onto a sphere of radius r OR a hyperbolic sheet
//  if we are away from the center of the sphere.
//
float Trackball::project_to_sphere(float r, float x, float y)
{
  float d, t, z;

  d = sqrt(x*x + y*y);
  if (d < r * 0.70710678118654752440) //! Inside  perspective sphere
    //if (d < r * r) //! Inside orthogonal sphere
    z = sqrt(r*r - d*d);
  else //! On hyperbola
    {
      t = r / 1.41421356237309504880;
      //t = r / (2.0*r);
      z = t*t / d;
    }

  return z;

} // project_to_sphere




/*
 * (c) Copyright 1993, 1994, Silicon Graphics, Inc.
 * ALL RIGHTS RESERVED
 * Permission to use, copy, modify, and distribute this software for
 * any purpose and without fee is hereby granted, provided that the above
 * copyright notice appear in all copies and that both the copyright notice
 * and this permission notice appear in supporting documentation, and that
 * the name of Silicon Graphics, Inc. not be used in advertising
 * or publicity pertaining to distribution of the software without specific,
 * written prior permission.
 *
 * THE MATERIAL EMBODIED ON THIS SOFTWARE IS PROVIDED TO YOU "AS-IS"
 * AND WITHOUT WARRANTY OF ANY KIND, EXPRESS, IMPLIED OR OTHERWISE,
 * INCLUDING WITHOUT LIMITATION, ANY WARRANTY OF MERCHANTABILITY OR
 * FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL SILICON
 * GRAPHICS, INC.  BE LIABLE TO YOU OR ANYONE ELSE FOR ANY DIRECT,
 * SPECIAL, INCIDENTAL, INDIRECT OR CONSEQUENTIAL DAMAGES OF ANY
 * KIND, OR ANY DAMAGES WHATSOEVER, INCLUDING WITHOUT LIMITATION,
 * LOSS OF PROFIT, LOSS OF USE, SAVINGS OR REVENUE, OR THE CLAIMS OF
 * THIRD PARTIES, WHETHER OR NOT SILICON GRAPHICS, INC.  HAS BEEN
 * ADVISED OF THE POSSIBILITY OF SUCH LOSS, HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, ARISING OUT OF OR IN CONNECTION WITH THE
 * POSSESSION, USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 * US Government Users Restricted Rights
 * Use, duplication, or disclosure by the Government is subject to
 * restrictions set forth in FAR 52.227.19(c)(2) or subparagraph
 * (c)(1)(ii) of the Rights in Technical Data and Computer Software
 * clause at DFARS 252.227-7013 and/or in similar or successor
 * clauses in the FAR or the DOD or NASA FAR Supplement.
 * Unpublished-- rights reserved under the copyright laws of the
 * United States.  Contractor/manufacturer is Silicon Graphics,
 * Inc., 2011 N.  Shoreline Blvd., Mountain View, CA 94039-7311.
 *
 * OpenGL(TM) is a trademark of Silicon Graphics, Inc.
 */
